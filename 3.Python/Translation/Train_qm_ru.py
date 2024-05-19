# The training is done in three stages:  
# 1) Train on all data, we need to get the error to be less than 1.
# 2) Based on the sentence corpus, the best solution is calculated using validation data. Polishing the model.
# 3) Based on the corpus with 5 sentences (or paragraphs), the best solution is calculated using validation data. To teach nllb-200 to handle multiple sentences.
# 
# And then the best model is selected based on the metrics: ChRF++ and Blue

# 1.Libraries -----------------------------
# !pip install nltk
# !pip install matplotlib
import os
# import operator
import json
import random
import torch
import nltk
import gc
import unicodedata
import re
import sys
import typing as tp
import numpy as np
import pandas as pd
import sacrebleu
from sacremoses import MosesPunctNormalizer
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from transformers.optimization import Adafactor
from transformers import get_constant_schedule_with_warmup
from tqdm.auto import tqdm, trange

# 2.CONSTANTS -----------------------------
# MODEL_NAME = "facebook/nllb-200-distilled-600M"
DATA_PATH = "1.Data/"
ALL_SENTENCES_PATH     = "".join([DATA_PATH, 'All_model.csv'])
ONE_SENTENCE_PATH      = "".join([DATA_PATH, 'One_sentence_model.csv'])
SEVERAL_SENTENCES_PATH = "".join([DATA_PATH, 'Several_sentence_model.csv'])

MODEL_PATH = "D:/Projects/Python/Models/NLLB_v1/"
# OLD_TOKENIZER_PATH = "".join([MODEL_PATH, 'old_tokenizer'])
# OLD_TOKENIZER_BPE_PATH = "".join([OLD_TOKENIZER_PATH, "/sentencepiece.bpe.model"])


MODEL_PATH_RAW = "".join([MODEL_PATH, 'nllb_krc_raw'])
# MODEL_PATH_QM_EMBEDDING = "".join([MODEL_PATH, 'nllb_qm_v1_emb'])
MODEL_PATH_QM_ALL = "".join([MODEL_PATH, 'nllb_qm_v1_all'])
MODEL_PATH_QM_ONE = "".join([MODEL_PATH, 'nllb_qm_v1_one'])
MODEL_PATH_QM_SEV = "".join([MODEL_PATH, 'nllb_qm_v1_sev'])

LANG_UNICODE = 'krc_Cyrl'
SRC_LANG = "krc_Cyrl"
TRG_LANG = "rus_Cyrl"

SHARE_VAL = 0.25
SHARE_TEST = 0.05

MAX_LENGTH = 512

SRC_LANG_DF = SRC_LANG.removesuffix("_Cyrl")
TRG_LANG_DF = TRG_LANG.removesuffix("_Cyrl")

LANGS = [(TRG_LANG_DF, TRG_LANG), (SRC_LANG_DF, SRC_LANG)]


# 3.LOAD MODEL -----------------------------

# tokenizer = NllbTokenizer.from_pretrained(MODEL_PATH_RAW, rebuild=True)
tokenizer = NllbTokenizer.from_pretrained(MODEL_PATH_RAW)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH_RAW)

if torch.cuda.is_available():
    model.cuda()
    
bleu_calc = sacrebleu.BLEU()
chrf_calc = sacrebleu.CHRF(word_order=2)  # this metric is called ChrF++

# 4.LOAD DATASETS -----------------------------
all_sentences = pd.read_csv(ALL_SENTENCES_PATH, sep = ';')[:64]
one_sentence = pd.read_csv(ONE_SENTENCE_PATH, sep = ';')[:64]
several_sentences = pd.read_csv(SEVERAL_SENTENCES_PATH, sep = ';')[:64]
# all_sentences_krc = np.array(all_sentences.krc)
# all_sentences[:10]
# all_sentences[TRG_LANG_DF][:10]

# all_pairs = list()
# for i in range(len(all_sentences)):
#     all_pairs.append(all_sentences.iloc[i].tolist())

# all_pairs = pd.DataFrame(all_pairs[:128])
# all_pairs = (all_pairs[:128])
# all_pairs = (all_pairs[:64])


# SRC_LANG_DF_INDEX = all_sentences.columns.get_loc(SRC_LANG_DF)
# TRG_LANG_DF_INDEX = all_sentences.columns.get_loc(TRG_LANG_DF)


#  Train-test-split
# which = lambda lst:list(np.where(lst)[0])

def splitDataset(df):
  num_test_samples = round(SHARE_TEST * len(df))
  num_val_samples = round(SHARE_VAL * len(df))
  num_train_samples = len(df) - num_val_samples - num_test_samples
  
  vec_types = ["train"] * num_train_samples + ["test"] * num_test_samples +  ["val"] * num_val_samples
  pair_group = random.sample(vec_types, len(vec_types))
  
  
  
  # test_vec = which([(a == "test") for a in pair_group])
  # train_vec = which([(a == "train") for a in pair_group])
  # val_vec = which([(a == "val") for a in pair_group])
  
  # test_pairs = operator.itemgetter(*test_vec)(all_pairs)
  # test_pairs = [df[i] for i in test_vec]
  # train_pairs = [df[i] for i in train_vec]
  # valid_pairs = [df[i] for i in val_vec]
  
  test_pairs = df[[(a == "test") for a in pair_group]]
  train_pairs = df[[(a == "train") for a in pair_group]]
  valid_pairs = df[[(a == "val") for a in pair_group]]
  
  return (train_pairs, valid_pairs, test_pairs)


train_all_sentences = all_sentences # update embeddings all
train_all, valid_all, test_all = splitDataset(df = all_sentences)
train_one, valid_one, test_one = splitDataset(df = one_sentence)
train_sev, valid_sev, test_sev = splitDataset(df = several_sentences)


# 5.PREPARING TOKENIZER -----------------------------
# Because language ids are added by hard-code on initialization, we need to manually fix them.
def fix_tokenizer(tokenizer, new_lang=LANG_UNICODE):
    """
    Add a new language token to the tokenizer vocabulary
    (this should be done each time after its initialization)
    """
    old_len = len(tokenizer) - int(new_lang in tokenizer.added_tokens_encoder)
    tokenizer.lang_code_to_id[new_lang] = old_len-1
    tokenizer.id_to_lang_code[old_len-1] = new_lang
    # always move "mask" to the last position
    tokenizer.fairseq_tokens_to_ids["<mask>"] = len(tokenizer.sp_model) + len(tokenizer.lang_code_to_id) + tokenizer.fairseq_offset

    tokenizer.fairseq_tokens_to_ids.update(tokenizer.lang_code_to_id)
    tokenizer.fairseq_ids_to_tokens = {v: k for k, v in tokenizer.fairseq_tokens_to_ids.items()}
    if new_lang not in tokenizer._additional_special_tokens:
        tokenizer._additional_special_tokens.append(new_lang)
    # clear the added token encoder; otherwise a new token may end up there by mistake
    tokenizer.added_tokens_encoder = {}
    tokenizer.added_tokens_decoder = {}

fix_tokenizer(tokenizer)

for token_id in range(len(tokenizer.sp_model), len(tokenizer)):
    token = tokenizer.convert_ids_to_tokens(token_id)
    print(token_id, token)
    
    
# tokenizer.src_lang = SRC_LANG
# tokenizer.tgt_lang = TRG_LANG

# 6.FUNCTIONS -----------------------------
# 6.1.PREPROC -----------------------------
mpn = MosesPunctNormalizer(lang="en")
mpn.substitutions = [
    (re.compile(r), sub) for r, sub in mpn.substitutions
]


def get_non_printing_char_replacer(replace_by: str = " ") -> tp.Callable[[str], str]:
    non_printable_map = {
        ord(c): replace_by
        for c in (chr(i) for i in range(sys.maxunicode + 1))
        # same as \p{C} in perl
        # see https://www.unicode.org/reports/tr44/#General_Category_Values
        if unicodedata.category(c) in {"C", "Cc", "Cf", "Cs", "Co", "Cn"}
    }

    def replace_non_printing_char(line) -> str:
        return line.translate(non_printable_map)

    return replace_non_printing_char

replace_nonprint = get_non_printing_char_replacer(" ")

def preproc(text):
    clean = mpn.normalize(text)
    clean = replace_nonprint(clean)
    # replace 𝓕𝔯𝔞𝔫𝔠𝔢𝔰𝔠𝔞 by Francesca
    clean = unicodedata.normalize("NFKC", clean)
    return clean
  
# 6.2. OTHER -----------------------------
def cleanup():
    """Try to free GPU memory"""
    gc.collect()
    torch.cuda.empty_cache()

def getBatchPairs(batch_size, data, step=None):
    
    (l1, long1), (l2, long2) = random.sample(LANGS, 2)
    xx, yy = [], []
    if step == None:
      for _ in range(batch_size):
        item = data.iloc[random.randint(0, len(data)-1)]
        xx.append(preproc(item[l1]))
        yy.append(preproc(item[l2]))
    else:
      batch_data = data[step * batch_size: (step + 1) * batch_size]
      xx=preproc(batch_data[l1].to_list())
      yy=preproc(batch_data[l2].to_list())
    
    return xx, yy, long1, long2

print(getBatchPairs(4, data=train_all_sentences, step=4))

# 6.3.TRAIN FUNCTION -----------------------------
# batch_size                - Size of batch
# checkpoint_steps          - After how many iterations to check and report 
# n_non_improve_val_perplex - Number of subsequent training checkpoints if there are no improvements in validation perplexity
# val_amount_test           - Number of tests for validation
# steps                     - Number of steps for training
# data_train                - Train dataset
# data_validation           - Validation dataset
# model_path                - Path to save model
# random_batch              - Is it need to train all data or random one?
# You need to initialize: losses and val_losses - they are the global variable
def trainModel(batch_size, checkpoint_steps, steps, data_train, model_path, data_validation=None, n_non_improve_val_perplex=None, val_amount_test=None, random_batch=True):
  train_data = data_train.sample(frac = 1)
  x, y, loss, loss_val  = None, None, None, None
  last_val_best_loss_perpl = float("Inf")
  improve_iter = 0
  
  tq = trange(len(losses), steps)
  for i in tq:
    if random_batch == True:
      xx, yy, lang1, lang2 = getBatchPairs(batch_size, data = train_data)
    else:
      xx, yy, lang1, lang2 = getBatchPairs(batch_size, data = train_data, step=i)
      
    try:
        tokenizer.src_lang = lang1
        x = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH).to(model.device)
        tokenizer.src_lang = lang2
        y = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH).to(model.device)
        y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100

        loss = model(**x, labels=y.input_ids).loss
        loss.backward()
        losses.append(loss.item())

        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

    except RuntimeError as e:
        optimizer.zero_grad(set_to_none=True)
        x, y, loss = None, None, None
        cleanup()
        print('error', max(len(s) for s in xx + yy), e)
        continue
      
    if i % checkpoint_steps == 0:
      # if (data_validation != None) & (n_non_improve_val_perplex != None) & (val_amount_test != None):
      if (n_non_improve_val_perplex != None) & (val_amount_test != None):
        for _ in range(0, val_amount_test):
          xx, yy, lang1, lang2 = getBatchPairs(batch_size, data = data_validation)
          
          try:
            tokenizer.src_lang = lang1
            x_val = tokenizer(xx, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH).to(model.device)
            tokenizer.src_lang = lang2
            y_val = tokenizer(yy, return_tensors='pt', padding=True, truncation=True, max_length=MAX_LENGTH).to(model.device)
            y.input_ids[y.input_ids == tokenizer.pad_token_id] = -100
          
            loss_val = model(**x_val, labels=y_val.input_ids).loss
            val_losses.append(loss_val.item())
            
          except RuntimeError as e:
            optimizer.zero_grad(set_to_none=True)
            x, y, loss, loss_val  = None, None, None, None
            cleanup()
            print('error', max(len(s) for s in xx + yy), e)
            continue
          
        train_loss = np.mean(losses[-checkpoint_steps:])
        train_perp = np.mean(np.exp(losses[-checkpoint_steps:]))
        val_loss = np.mean(val_losses[-val_amount_test:])
        val_perp = np.mean(np.exp(val_losses[-val_amount_test:]))
        # val_blue = np.mean(blue_val)
        # val_chrf = np.mean(chrf_losses)
         
        print('Step:', i, '/', steps, '; loss:', train_loss, '; validation loss:', val_loss,'; perplexity:', train_perp, '; validation perplexity:', val_perp)
              
         
        # Update control values and save best result
        if val_perp < last_val_best_loss_perpl:
          last_val_best_loss_perpl = val_perp
          improve_iter = i
          print('Save model')
          # model.save_pretrained(model_path, force_download=True)
          # tokenizer.save_pretrained(model_path, force_download=True)
          model.save_pretrained(model_path)
          tokenizer.save_pretrained(model_path)
          
        # Stop cycle
        if (i - improve_iter) >= (n_non_improve_val_perplex * checkpoint_steps):
          print('Break')
          break
        
      else:
        train_loss = np.mean(losses[-checkpoint_steps:])
        train_perp = np.mean(np.exp(losses[-checkpoint_steps:]))
        print('Step:', i, '/', steps, '; loss:', train_loss, '; perplexity:', train_perp)
        
        model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)
        

# 7.TRAIN -----------------------------
optimizer = Adafactor(
    [p for p in model.parameters() if p.requires_grad],
    scale_parameter=False,
    relative_step=False,
    lr=1e-4,
    clip_threshold=1.0,
    weight_decay=1e-3,
)

scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)


model.train()



# 7.1.TRAINING ALL -----------------------------
# Here we need to get the error to be less than 1, so run one to several times
batch_size = 16 # 2
checkpoint_steps = 1000 # 1
n_non_improve_val_perplex = 7 # Number of subsequent training checkpoints if there is no improvement in validation perplexity
val_amount_test = 10 # Number of tests for validation
epoches = 3

for i in trange(0, epoches):
  print("epoch ", i, "/", epoches)
  losses = []
  val_losses = []
  trainModel(batch_size                = batch_size, 
             checkpoint_steps          = checkpoint_steps,
             steps                     = int(len(train_all_sentences) / batch_size),
             # steps                     = 10, 
             data_train                = train_all_sentences, 
             model_path                = MODEL_PATH_QM_ALL, 
             data_validation           = None, 
             n_non_improve_val_perplex = None, 
             val_amount_test           = None, 
             random_batch              = False)
           

pd.Series(losses).ewm(100).mean().plot();

# # 7.2.TRAINING ALL -----------------------------
# 
# cleanup()
# 
# tokenizer = NllbTokenizer.from_pretrained(MODEL_PATH_QM_EMBEDDING) # MODEL_PATH_RAW
# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH_QM_EMBEDDING) # MODEL_PATH_RAW
# 
# if torch.cuda.is_available():
#     model.cuda()
# 
# fix_tokenizer(tokenizer)
# print(tokenizer.convert_ids_to_tokens([262923, 262924, 262925])) # ['zul_Latn', 'krc_Cyrl', '<mask>']
# print(tokenizer.convert_tokens_to_ids(['zul_Latn', 'krc_Cyrl', '<mask>'])) # [262923, 262924, 262925]
# 
# optimizer = Adafactor(
#     [p for p in model.parameters() if p.requires_grad],
#     scale_parameter=False,
#     relative_step=False,
#     lr=1e-4,
#     clip_threshold=1.0,
#     weight_decay=1e-3,
# )
# 
# scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)
# 
# model.train()
# 
# 
# losses = []
# val_losses = []
# batch_size = 4 # 2
# checkpoint_steps = 3000 # 1
# n_non_improve_val_perplex = 8 # Number of subsequent training checkpoints if there is no improvement in validation perplexity
# 
# trainModel(batch_size                = batch_size, 
#            checkpoint_steps          = checkpoint_steps, 
#            n_non_improve_val_perplex = n_non_improve_val_perplex, # Number of subsequent training checkpoints if there is no improvement in validation perplexity
#            val_amount_test           = val_amount_test, # Number of tests for validation
#            # steps                     = 10,
#            steps                     = int(len(train_all) / batch_size),
#            data_train                = train_all, 
#            data_validation           = valid_all, 
#            model_path                = MODEL_PATH_QM_ALL, 
#            random_batch              = True)
#         
# 
# # pd.Series(losses).ewm(100).mean().plot();
# pd.Series(losses).plot();


# 7.3.TRAINING ONE -----------------------------

cleanup()

tokenizer = NllbTokenizer.from_pretrained(MODEL_PATH_QM_ALL) # MODEL_PATH_RAW
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH_QM_ALL) # MODEL_PATH_RAW

if torch.cuda.is_available():
    model.cuda()

fix_tokenizer(tokenizer)
print(tokenizer.convert_ids_to_tokens([262923, 262924, 262925])) # ['zul_Latn', 'krc_Cyrl', '<mask>']
print(tokenizer.convert_tokens_to_ids(['zul_Latn', 'krc_Cyrl', '<mask>'])) # [262923, 262924, 262925]

optimizer = Adafactor(
    [p for p in model.parameters() if p.requires_grad],
    scale_parameter=False,
    relative_step=False,
    lr=1e-4,
    clip_threshold=1.0,
    weight_decay=1e-3,
)

scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)

model.train()

losses = []
val_losses = []
batch_size = 16 # 2
checkpoint_steps = 1000 # 1
n_non_improve_val_perplex = 28 # Number of subsequent training checkpoints if there is no improvement in validation perplexity
multiplicator = 12

# int(len(train_one) / batch_size) * multiplicator / checkpoint_steps


trainModel(batch_size                = batch_size, 
           checkpoint_steps          = checkpoint_steps, 
           n_non_improve_val_perplex = n_non_improve_val_perplex, # Number of subsequent training checkpoints if there is no improvement in validation perplexity
           val_amount_test           = val_amount_test, # Number of tests for validation
           # steps                     = 10,
           steps                     = int(len(train_one) / batch_size) * multiplicator,
           data_train                = train_one, 
           data_validation           = valid_one, 
           model_path                = MODEL_PATH_QM_ONE, 
           random_batch              = True)
        

# pd.Series(losses).ewm(100).mean().plot();
pd.Series(losses).plot();


# 7.4.TRAINING SEVERAL -----------------------------

cleanup()

tokenizer = NllbTokenizer.from_pretrained(MODEL_PATH_QM_ONE) # MODEL_PATH_RAW
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH_QM_ONE) # MODEL_PATH_RAW

if torch.cuda.is_available():
    model.cuda()

fix_tokenizer(tokenizer)
print(tokenizer.convert_ids_to_tokens([262923, 262924, 262925])) # ['zul_Latn', 'krc_Cyrl', '<mask>']
print(tokenizer.convert_tokens_to_ids(['zul_Latn', 'krc_Cyrl', '<mask>'])) # [262923, 262924, 262925]

optimizer = Adafactor(
    [p for p in model.parameters() if p.requires_grad],
    scale_parameter=False,
    relative_step=False,
    lr=1e-4,
    clip_threshold=1.0,
    weight_decay=1e-3,
)

scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=1000)

model.train()

losses = []
val_losses = []
batch_size = 8 # 2
checkpoint_steps = 500 # 1
n_non_improve_val_perplex = 16 # Number of subsequent training checkpoints if there is no improvement in validation perplexity
multiplicator = 8

trainModel(batch_size                = batch_size, 
           checkpoint_steps          = checkpoint_steps, 
           n_non_improve_val_perplex = n_non_improve_val_perplex, # Number of subsequent training checkpoints if there is no improvement in validation perplexity
           val_amount_test           = val_amount_test, # Number of tests for validation
           # steps                     = 10,
           steps                     = int(len(train_sev) / batch_size)*multiplicator,
           data_train                = train_sev, 
           data_validation           = valid_sev, 
           model_path                = MODEL_PATH_QM_SEV, 
           random_batch              = True)
        

# pd.Series(losses).ewm(100).mean().plot();
pd.Series(losses).plot();

# 8.EVALUATION -----------------------------
# model_emb = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH_QM_EMBEDDING)
model_all = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH_QM_ALL)
model_one = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH_QM_ONE)
model_sev = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH_QM_SEV)


def translate(model_input,
    text, src_lang='rus_Cyrl', tgt_lang='eng_Latn', 
    a=32, b=3, max_input_length=1024, num_beams=4, **kwargs
):
    """Turn a text or a list of texts into a list of translations"""
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(
        text, return_tensors='pt', padding=True, truncation=True, 
        max_length=max_input_length
    )
    model_input.eval() # turn off training mode
    result = model_input.generate(
        **inputs.to(model_input.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams, **kwargs
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)
  

# Example usage:
t = 'Иш къолай болсун'
# translate(model_input=model_emb, text=t, src_lang='krc_Cyrl', tgt_lang='rus_Cyrl')
translate(model_input=model_all, text=t, src_lang='krc_Cyrl', tgt_lang='rus_Cyrl')
translate(model_input=model_one, text=t, src_lang='krc_Cyrl', tgt_lang='rus_Cyrl')
translate(model_input=model_sev, text=t, src_lang='krc_Cyrl', tgt_lang='rus_Cyrl')
# ['Да будет работа успешной']



# Analyze
def modelColumn(model_name, data):
  df_cur = pd.DataFrame(data = {'model': [model_name] * len(data)})
  df_cur.reset_index(drop=True, inplace=True)
  data.reset_index(drop=True, inplace=True)
  return pd.concat([df_cur, data], axis=1)


def uniteData(model_name, datas):
  return pd.concat([modelColumn(model_name = model_name, data = i) for i in datas], axis=0)

def translateByData(model_name, model_input, datas, lg1 = "krc", lg2 = 'rus', lg1_writing = "Cyrl", lg2_writing = "Cyrl"):
  df = uniteData(model_name, datas)
  from_to1 = "".join([lg1, '2', lg2])
  from_to2 = "".join([lg2, '2', lg1])
  lang1 = "".join([lg1, '_', lg1_writing])
  lang2 = "".join([lg2, '_', lg2_writing])
  
  df[from_to1] = [translate(model_input, t, lang1, lang2)[0] for t in tqdm(df[lg1])]
  df[from_to2] = [translate(model_input, t, lang2, lang1)[0] for t in tqdm(df[lg2])]
  
  return df

# all data
RANDOM_VALS = 2
data_vec = [test_all.copy().sample(RANDOM_VALS), test_one.copy().sample(RANDOM_VALS), test_sev.copy().sample(RANDOM_VALS)]

# df_emb = translateByData(model_name="model_emb", datas=data_vec, model_input=model_emb)
df_all = translateByData(model_name="model_all", datas=data_vec, model_input=model_all)
df_one = translateByData(model_name="model_one", datas=data_vec, model_input=model_one)
df_sev = translateByData(model_name="model_sev", datas=data_vec, model_input=model_sev)

# df_all_all = pd.concat([df_emb, df_all, df_one, df_sev], axis=0)  
df_all_all = pd.concat([df_all, df_one, df_sev], axis=0)  
df_all_all.reset_index(drop=True, inplace=True)

df_all_all["blue_rus"] = [bleu_calc.corpus_score([df_all_all["krc2rus"][x]], [[df_all_all['rus'][x]]]).score for x in df_all_all.index] 
df_all_all["blue_krc"] = [bleu_calc.corpus_score([df_all_all["rus2krc"][x]], [[df_all_all['krc'][x]]]).score for x in df_all_all.index] 
df_all_all["chrf_rus"] = [chrf_calc.corpus_score([df_all_all["krc2rus"][x]], [[df_all_all['rus'][x]]]).score for x in df_all_all.index] 
df_all_all["chrf_krc"] = [chrf_calc.corpus_score([df_all_all["rus2krc"][x]], [[df_all_all['krc'][x]]]).score for x in df_all_all.index] 

# Result
df_all_all.copy().groupby("model")[['blue_rus', 'blue_krc', "chrf_rus", "chrf_krc"]].median()
