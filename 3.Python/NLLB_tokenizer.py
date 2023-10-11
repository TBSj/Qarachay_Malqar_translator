# Libraries

# !pip install torch
# !pip install pandas
# !pip install tqdm
# !pip install transformers
# !pip install sacremoses


# !pip install --upgrade google-api-python-client
# !pip install --upgrade "protobuf<=3.20.1"

import torch
import random
import re
import pandas as pd
import numpy as np
import sentencepiece.sentencepiece_model_pb2 as sp_pb2_model
from collections import Counter, defaultdict
from copy import deepcopy
from heapdict import heapdict
from tqdm.auto import tqdm, trange
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline, NllbTokenizer
from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES
import sys
import unicodedata
from sacremoses import MosesPunctNormalizer
import sentencepiece as spm

# Consts
DATA_PATH = "1.Data/"
MODEL_NAME = "facebook/nllb-200-distilled-600M"
ALL_SENTENCES_PATH = "".join([DATA_PATH, 'All_model.csv'])
MODEL_PATH = "D:/Projects/Python/Models/NLLB_v1/"
OLD_TOKENIZER_PATH = "".join([MODEL_PATH, 'old_tokenizer'])
OLD_TOKENIZER_BPE_PATH = "".join([OLD_TOKENIZER_PATH, "/sentencepiece.bpe.model"])
LANG_UNICODE = 'krc_Cyrl'
MODEL_PATH_RAW = "".join([MODEL_PATH, 'nllb_krc_raw'])

ALL_TEXT_PLAIN = "".join([DATA_PATH, 'myv_texts_plain.txt'])
SPM_PREFIX = "".join([MODEL_PATH, 'SPM_QM/SPM_QM'])

NEW_SPM_NAME = "".join([MODEL_PATH, 'SPM_QM/spm_nllb_qm.model'])

# Model
tokenizer = NllbTokenizer.from_pretrained(MODEL_NAME)

# model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
# if torch.cuda.is_available():
#     model.cuda()
    

# Languages checking

# pd.Series([lang.split('_')[1] for lang in tokenizer.lang_code_to_id]).value_counts()
# pd.Series([lang.split('_')[1] for lang in tokenizer.lang_code_to_id])
# 
# # 'crh_Latn': 256039 - Къырымтатар тил
# # 'tat_Cyrl': 256171 - Татар тил
# # 'kaz_Cyrl': 256089, - Къазакъ тил
# 
# # Translate Fun 
# def translate(text, src_lang, tgt_lang, **kwargs):
#     tokenizer.src_lang = src_lang
#     tokenizer.tgt_lang = tgt_lang
#     inputs = tokenizer(text, return_tensors='pt')
#     result = model.generate(
#         **inputs.to(model.device), 
#         forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
#         **kwargs
#     )
#     return tokenizer.decode(result[0], skip_special_tokens=True)
#   
# 
# translate('Привет, как твои дела?', 'rus_Cyrl', 'spa_Latn')
# len(tokenizer.lang_code_to_id)


# Counting words 
PUNCT = '.,-—:)(»«!?–/;„"“…*́№Ёҥ[]”^%+І=і•_􏰀²|}{#‘■>⁠’á<°\§\''
  
all_sentences = pd.read_csv(ALL_SENTENCES_PATH, sep = ';')
all_sentences_krc = np.array(all_sentences.krc.replace(PUNCT, ""))
all_sentences["krc"][:10]
all_sentences_krc[:10]


all_pairs = list()
for i in range(len(all_sentences)):
    all_pairs.append(all_sentences.iloc[i].tolist())

all_pairs[:10]

# text = 'Аперимсе!!! :) :-\ :-D'
# tokenizer.vocab_size
# tokenizer.tokenize(text)
# 
# tokenizer.encode(text)
# 
# tokens = tokenizer.tokenize(text)
# tokens

# char_count = Counter()
# for text in tqdm(all_sentences_krc):
#     char_count.update(text)
#     
# print(char_count)

PUNCT = '.,-—:)(»«!?–/;„"“…*́№Ёҥ[]”^%+І=і•_􏰀²|}{#‘■>⁠’á<°\§\''
SPACE = '▁'

# for k, v in char_count.most_common(200):
#     if not re.match('[а-яА-Яa-zA-Z0-9ё\']', k):
#         print(k, end='')
        
# toks = tokenizer.tokenize(text)
# toks

def get_words(text, tokenizer, verbose=False):
    toks = tokenizer.tokenize(text)
    words = []
    word = []
    prev_punct = False
    for tok in toks:
        is_punct = tok.lstrip(SPACE) and all(c in PUNCT for c in tok.lstrip(SPACE))
        if tok.startswith(SPACE) or prev_punct != is_punct:
            if word:
                words.append(word)
            word = []
        word.append(tok)
        prev_punct = is_punct
    if word:
        words.append(word)
    if verbose:
        print(words)
    res = words
    # assert tokenizer.decode([tok for t in res for tok in t]) == text
    return res

# The number of words in our dataset
word_count = Counter()
word2toks = {}
for text in tqdm(all_sentences_krc):
    for word_toks in get_words(text, tokenizer):
        word = ''.join(word_toks)
        word_count[word] += 1
        word2toks[word] = word_toks
print(len(word_count))
print(len(word2toks))

# Deep copy
word_count2 = deepcopy(word_count)
word2toks2 = deepcopy(word2toks)

# Computing splits 
word_count = deepcopy(word_count2)
word2toks = deepcopy(word2toks2)


# for k, v in word_count.most_common(30):
#     if len(word2toks[k]) > 1:
#         print(word2toks[k])




pairs_count = Counter()
pair2word = defaultdict(set)
for w, c in tqdm(word_count.items(), total=len(word_count)):
    enc = word2toks[w]
    for pair in zip(enc[:-1], enc[1:]):
        pairs_count[pair] += c
        pair2word[pair].add(w)



# !pip install heapdict
hd = heapdict()
for w, c in pairs_count.items():
    hd[w] = -c

steps = 100_000
min_count = 50 # 30
# default:   0 new tokens, 30 lenght, 0% new tokens
# 100 mindf: 6.6k new tokens, 22 length, 47% new tokens (of sentence length)
# 30 mindf:  20k new tokens, 20 length, 58% new tokens
# 10 mindf: 50K new tokens, 18.5 length, 64% new tokens
extra_vocab = []
extra_counts = []
extra_pairs = []

def replace_pair(old_tokens, pair, new_token):
    result = []
    prev = old_tokens[0]
    for tok in old_tokens[1:]:
        if (prev, tok) == pair:
            result.append(new_token)
            prev = None
        else:
            if prev is not None:
                result.append(prev)
            prev = tok
    if prev is not None:
        result.append(prev)
    return result
  
# Create additional tokens
for _ in trange(steps):
    #pair, c = pairs_count.most_common(1)[0]  # это самая времязатратная операция
    pair, c = hd.peekitem()
    c = -c

    if c < min_count:
        break
    new_token = ''.join(pair) # instead of BERT-like pair[0] + pair[1][2:]
    extra_vocab.append(pair)
    extra_counts.append(c)
    extra_pairs.append(pair)

    # update the vocabulary
    #new_id = len(id2ids)
    #tok2id[new_token] = new_id
    #id2ids.append(id2ids[tok2id[pair[0]]] + id2ids[tok2id[pair[1]]])

    # calculate the delta for the heap
    delta = Counter()
    for word in list(pair2word[pair]):
        # calculate old and new ways to tokenize the word
        old_toks = word2toks[word]
        # new_toks = " ".join(old_toks).replace(' '.join(pair), new_token).split(" ")
        new_toks = replace_pair(old_toks, pair, new_token)
        word2toks[word] = new_toks
        wc = word_count[word]
        # update the index concerning the tokens of the word
        for old_pair in zip(old_toks[:-1], old_toks[1:]):
            #pairs_count[old_pair] -= wc
            delta[old_pair] -= wc
            if word in pair2word[old_pair]:
                pair2word[old_pair].remove(word)
        for new_pair in zip(new_toks[:-1], new_toks[1:]):
            # pairs_count[new_pair] += wc
            delta[new_pair] += wc
            pair2word[new_pair].add(word)
    # update the heap
    for a_pair, a_delta in delta.items():
        if a_delta == 0:
            continue
        if a_pair not in hd:
            hd[a_pair] = 0
        hd[a_pair] -= a_delta



len(extra_pairs)
extra_pairs[:10]
extra_pairs[-20:]
tokenizer.vocab_size
# adding token
# new_special_tokens = tokenizer.additional_special_tokens + [LANG_UNICODE]
# tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

tokenizer.save_pretrained(OLD_TOKENIZER_PATH)
# !pip install transformers sentencepiece -q
# !pip install sentencepiece
# !wget https://raw.githubusercontent.com/google/sentencepiece/master/src/sentencepiece_model.proto
# !curl https://raw.githubusercontent.com/google/sentencepiece/master/src/sentencepiece_model.proto
# ! protoc --python_out=. sentencepiece_model.proto
# import sentencepiece_model_pb2 as model


#  Add new extraaxted tokens
m = model.ModelProto()
m.ParseFromString(open(OLD_TOKENIZER_BPE_PATH, "rb").read())

scores = [p.score for p in m.pieces]
min_score = min(scores)
epsilon = 1e-4

tokenizer('къалайса')
type(m.pieces[37764-1].piece)


for i, pair in enumerate(extra_vocab):
    new_token = model.ModelProto().SentencePiece()
    new_token.piece = ''.join(pair)
    #print(''.join(pair))
    #print(pair)
    #print(new_token)
    #print("\n")
    new_token.score = min_score - epsilon * (i+1)
    m.pieces.append(new_token)



with open(OLD_TOKENIZER_BPE_PATH, 'wb') as f:
    f.write(m.SerializeToString())

tokenizer.vocab_size



new_tokenizer = NllbTokenizer.from_pretrained(
    # "".join([OLD_TOKENIZER_PATH, ":", "sentencepiece.bpe.model"]),
    OLD_TOKENIZER_BPE_PATH,
   # local_files_only=True,
    additional_special_tokens = tokenizer.additional_special_tokens
)

new_tokenizer.vocab_size

# new_tokenizer.added_tokens_decoder

# Add Language token
self = new_tokenizer
self.lang_code_to_id = {
    code: self.sp_model_size + i + self.fairseq_offset for i, code in enumerate(FAIRSEQ_LANGUAGE_CODES + [LANG_UNICODE])
} # new_tokenizer.lang_code_to_id.update(new_tokenizer.added_tokens_encoder)

self.id_to_lang_code = {v: k for k, v in self.lang_code_to_id.items()} # new_tokenizer.id_to_lang_code.update(new_tokenizer.added_tokens_decoder)
self.fairseq_tokens_to_ids["<mask>"] = len(self.sp_model) + len(self.lang_code_to_id) + self.fairseq_offset

self.fairseq_tokens_to_ids.update(self.lang_code_to_id)
self.fairseq_ids_to_tokens = {v: k for k, v in self.fairseq_tokens_to_ids.items()}

new_tokenizer.additional_special_tokens.append(LANG_UNICODE)
# Былай боллукъду
# new_special_tokens = new_tokenizer.additional_special_tokens + [LANG_UNICODE]
# new_tokenizer.add_special_tokens({'additional_special_tokens': new_special_tokens})

# old_vocab_size = len(tokenizer.sp_model) + 1
# 
# # Move added tokens to the end
# for old_token_id in range(old_vocab_size, len(tokenizer)):
#     old_token = tokenizer.convert_ids_to_tokens(old_token_id)
#     new_token_id = new_tokenizer.convert_tokens_to_ids(old_token)
#     # new_token = new_tokenizer.convert_ids_to_tokens(new_token_id-1)
#     new_token = new_tokenizer.convert_ids_to_tokens(new_token_id)
# 
#     print(old_token_id, old_token, new_token_id, new_token)




# Check
text = random.choice(all_sentences_krc)

print(tokenizer.tokenize(text))
print(new_tokenizer.tokenize(text))



random.seed(1)
sample = random.sample(list(all_sentences_krc), 10000)
sample[:10]

pd.DataFrame({
    'old': [len(tokenizer.tokenize(text)) for text in sample],
    'new': [len(new_tokenizer.tokenize(text)) for text in sample]
}).describe()


new_tot_len, tot_len = 0, 0
for text in sample:
    for tok in new_tokenizer.tokenize(text):
        s = len(tok)
        tot_len += s
        if new_tokenizer.convert_tokens_to_ids(tok) > len(tokenizer.sp_model):
            new_tot_len += s
            
print(new_tot_len / tot_len)



# Updating the model embeddings
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
if torch.cuda.is_available():
    model.cuda()

model.model.shared

model.resize_token_embeddings(len(new_tokenizer))

n_extra = len(extra_vocab)
print(n_extra)
old_vocab_size = len(tokenizer.sp_model) + 1

# Move added tokens to the end
for old_token_id in range(old_vocab_size, len(tokenizer)):
    old_token = tokenizer.convert_ids_to_tokens(old_token_id)
    new_token_id = new_tokenizer.convert_tokens_to_ids(old_token)
    new_token = new_tokenizer.convert_ids_to_tokens(new_token_id)

    print(old_token_id, old_token, new_token_id, new_token)
    # model.model.shared.weight.data[i + n_extra] = model.model.shared.weight.data[i]
    model.model.shared.weight.data[new_token_id] = model.model.shared.weight.data[old_token_id]

# Because we have added one more language, its id must be computed separately, e.g. as an average of related languages.
# 'crh_Latn': 256039 - Къырымтатар тил
# 'tat_Cyrl': 256171 - Татар тил
# 'kaz_Cyrl': 256089 - Къазакъ тил
model.model.shared.weight.data[new_tokenizer.convert_tokens_to_ids(LANG_UNICODE)] = (
    model.model.shared.weight.data[tokenizer.convert_tokens_to_ids('tat_Cyrl')] * 0.55
    + model.model.shared.weight.data[tokenizer.convert_tokens_to_ids('crh_Latn')] * 0.35
    + model.model.shared.weight.data[tokenizer.convert_tokens_to_ids('kaz_Cyrl')] * 0.1
)


# Compute embeddings for newly added tokens
token_priors = Counter()
token_to_others = defaultdict(Counter)

for qm, ru in tqdm(all_pairs):
    qm_toks = new_tokenizer.convert_tokens_to_ids(new_tokenizer.tokenize(qm))
    ru_toks = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ru))
    token_priors.update(ru_toks)
    for qm_tok in qm_toks:
        token_to_others[qm_tok].update(ru_toks)

def get_ru_toks(qm_tok):
    ru_toks = []
    ru_weights = []
    for t, w in token_to_others[qm_tok].items():
        ru_toks.append(t)
        ru_weights.append(w**2 / token_priors[t])
    ru_weights = np.array(ru_weights)
    ru_weights = ru_weights / (sum(ru_weights) + 1e-4)
    return ru_weights, ru_toks



for i in trange(n_extra):
    qm_tok = i + old_vocab_size
    ru_weights, ru_toks = get_ru_toks(qm_tok)
    if len(ru_toks) > 0:
        new_embedding = (model.model.shared.weight.data[ru_toks].T * ru_weights).sum(1)
        model.model.shared.weight.data[qm_tok] = new_embedding



# Save Model
model.save_pretrained(MODEL_PATH_RAW)
new_tokenizer.save_pretrained(MODEL_PATH_RAW)

model1 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH_RAW) 
tokenizer1 = NllbTokenizer.from_pretrained(MODEL_PATH_RAW)
# tokenizer1 = NllbTokenizer.from_pretrained(MODEL_PATH_RAW, rebuild=True)

model.model.shared
model1.model.shared

# Check that the model is still able to translate texts.
article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"

# translate Hindi to French
new_tokenizer.src_lang = "hin_Deva"
encoded_hi = new_tokenizer(article_hi, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_hi,
    forced_bos_token_id=new_tokenizer.lang_code_to_id["fra_Latn"],
    max_length=30,
    num_beams=1
)
new_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire dans la Syrie."

# Translation to Russian has broken, because some new qm tokens are very Russian-like, and they interere. 
generated_tokens = model.generate(
    **encoded_hi,
    forced_bos_token_id=new_tokenizer.lang_code_to_id["rus_Cyrl"],
    max_length=30,
    num_beams=5,
    repetition_penalty=30.0,
)
new_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire dans la Syrie."

# translate from Qrachay-Malqar! 
qm, ru = random.choice(all_pairs)
qm, ru

new_tokenizer.src_lang = LANG_UNICODE
encoded_hi = new_tokenizer(qm, return_tensors="pt")

generated_tokens = model.generate(
    **encoded_hi,
    forced_bos_token_id=new_tokenizer.lang_code_to_id["rus_Cyrl"],
    max_length=30,
    num_beams=5,
    repetition_penalty=30.0
)
new_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

