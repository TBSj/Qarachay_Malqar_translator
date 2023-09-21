# Libraries
import os
import numpy as np
import pandas as pd
from tqdm.auto import tqdm, trange
import json
import random
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from transformers.optimization import Adafactor
# Consts
# MODEL_NAME = "facebook/nllb-200-distilled-600M"
ALL_SENTENCES_PATH = '1.Data/All_model.csv'
MODEL_PATH = "D:/Projects/Python/Models/NLLB_v1/"
# OLD_TOKENIZER_PATH = "".join([MODEL_PATH, 'old_tokenizer'])
# OLD_TOKENIZER_BPE_PATH = "".join([OLD_TOKENIZER_PATH, "/sentencepiece.bpe.model"])
LANG_UNICODE = 'krc_Cyrl'
MODEL_PATH_RAW = "".join([MODEL_PATH, 'nllb_krc_raw'])
SRC_LANG = "krc_Cyrl"
TRG_LANG = "rus_Cyrl"

SRC_LANG_DF = SRC_LANG.removesuffix("_Cyrl")
TRG_LANG_DF = TRG_LANG.removesuffix("_Cyrl")



# Model
# tokenizer = NllbTokenizer.from_pretrained(MODEL_PATH_RAW, rebuild=True)
tokenizer = NllbTokenizer.from_pretrained(MODEL_PATH_RAW)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH_RAW)

if torch.cuda.is_available():
    model.cuda()

# Load datasets
all_sentences = pd.read_csv(ALL_SENTENCES_PATH, sep = ';')
# all_sentences_krc = np.array(all_sentences.krc)
# all_sentences[:10]
# all_sentences[TRG_LANG_DF][:10]
all_pairs = list()
for i in range(len(all_sentences)):
    all_pairs.append(all_sentences.iloc[i].tolist())

all_pairs = all_pairs[:128]


SRC_LANG_DF_INDEX = all_sentences.columns.get_loc(SRC_LANG_DF)
TRG_LANG_DF_INDEX = all_sentences.columns.get_loc(TRG_LANG_DF)


# Preparation tokenizer
# Because language ids are added by hard-code on initialization, we need to manually fix them.
tokenizer.vocab_size
tokenizer.lang_code_to_id[LANG_UNICODE] = len(tokenizer)-1
tokenizer.id_to_lang_code[len(tokenizer)-1] = LANG_UNICODE
tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset

tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
if LANG_UNICODE not in tokenizer.additional_special_tokens:
    # tokenizer.additional_special_tokens.append(LANG_UNICODE)
    new_special_tokens = tokenizer.additional_special_tokens + [LANG_UNICODE]
    tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

for token_id in range(len(tokenizer.sp_model), len(tokenizer)):
    token = tokenizer.convert_ids_to_tokens(token_id)
    print(token_id, token)

# Train
for p in model.parameters():
    p.requires_grad = False
for p in model.model.shared.parameters():
    p.requires_grad = True
    
     
optimizer = Adafactor(
    [p for p in model.parameters() if p.requires_grad],
    scale_parameter=False,
    relative_step=False,
    lr=1e-5,
    clip_threshold=1.0
)

batch_size = 1 # 8
report_steps = 1
epochs = 1
losses = []

tokenizer.src_lang = SRC_LANG
tokenizer.tgt_lang = TRG_LANG

model.train()

for epoch in range(epochs):
    print('EPOCH', epoch)
    random.shuffle(all_pairs)
    for i in trange(0, int(len(all_pairs) / batch_size)):
        batch = all_pairs[i * batch_size: (i + 1) * batch_size]
        # кодируем вопрос и ответ
        x = tokenizer([p[SRC_LANG_DF_INDEX] for p in batch], return_tensors='pt', padding=True, truncation=True, max_length=256).to(model.device)
        with tokenizer.as_target_tokenizer():
            y = tokenizer([p[TRG_LANG_DF_INDEX] for p in batch], return_tensors='pt', padding=True, truncation=True, max_length=256).to(model.device)
        # -100 - специальное значение, позволяющее не учитывать токены
        y.input_ids[y.input_ids == 0] = -100
        # вычисляем функцию потерь
        try:
            loss = model(
                input_ids=x.input_ids,
                attention_mask=x.attention_mask,
                labels=y.input_ids,
                decoder_attention_mask=y.attention_mask,
                return_dict=True
            ).loss
            # делаем шаг градиентного спуска
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        except RuntimeError as e:
            print('error')
            loss = None
            optimizer.zero_grad(set_to_none=True)
            cleanup()
            continue

        # печатаем скользящее среднее значение функции потерь
        losses.append(loss.item())
        if i % report_steps == 0:
            print('step', i, 'loss', np.mean(losses[-report_steps:]))
