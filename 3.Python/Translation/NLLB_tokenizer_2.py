# 1.LIBRARIES ------------------------------------------------

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
from tqdm.auto import tqdm, trange
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
# from transformers.models.nllb.tokenization_nllb import FAIRSEQ_LANGUAGE_CODES
import sys
import unicodedata
from sacremoses import MosesPunctNormalizer
import sentencepiece as spm

# 2.CONSTANTS ------------------------------------------------
DATA_PATH = "1.Data/"
MODEL_NAME = "facebook/nllb-200-distilled-600M"
ALL_SENTENCES_PATH = "".join([DATA_PATH, 'All_model.csv'])
MODEL_PATH = "D:/Projects/Python/Models/NLLB_v1/"
# OLD_TOKENIZER_PATH = "".join([MODEL_PATH, 'old_tokenizer'])
# OLD_TOKENIZER_BPE_PATH = "".join([OLD_TOKENIZER_PATH, "/sentencepiece.bpe.model"])
LANG_UNICODE = 'krc_Cyrl'
MODEL_PATH_RAW = "".join([MODEL_PATH, 'nllb_krc_raw'])

ALL_TEXT_PLAIN = "".join([DATA_PATH, 'my_texts_plain.txt'])
SPM_PREFIX = "".join([MODEL_PATH, 'SPM_QM/SPM_QM'])
NEW_SPM_NAME = "".join([MODEL_PATH, 'SPM_QM/spm_nllb_qm.model'])

PUNCT = '.,-—:)(»«!?–/;„"“…*́№Ёҥ[]”^%+І=і•_􏰀²|}{#‘■>⁠’á<°\§\''
  
# 3.DOWNLOADING ------------------------------------------------

  
all_sentences = pd.read_csv(ALL_SENTENCES_PATH, sep = ';')
all_sentences_krc = np.array(all_sentences.krc.replace(PUNCT, ""))
all_sentences["krc"][:10]
all_sentences_krc[:10]

tokenizer = NllbTokenizer.from_pretrained(MODEL_NAME)

# 4.LOOKING AT THE DATA ------------------------------------------------
# How well does the data fit into a NLLB tokenizer?
def wordTokenize(text):
    # a very naive word tokenizer for languages with English-like orthography
    return re.findall('(\\w+|[^\\w\\s])', text)

# smpl = all_sentences.sample(10000, random_state=1)
smpl = all_sentences
smpl['rus_toks'] = smpl.rus.apply(tokenizer.tokenize)
smpl['krc_toks'] = smpl.krc.apply(tokenizer.tokenize)
smpl['rus_words'] = smpl.rus.apply(wordTokenize)
smpl['krc_words'] = smpl.krc.apply(wordTokenize)

print(smpl)


# Actually, we can compute precise statistics of this:

stats = smpl[
    ['rus_toks', 'krc_toks', 'rus_words', 'krc_words']
].applymap(len).describe()
print(stats.rus_toks['mean'] / stats.rus_words['mean'])  # 1.65
print(stats.krc_toks['mean'] / stats.krc_words['mean'])  # 2.38
stats


# There is enough big range between a number of tokens for russian and qarachay-malqar languages. So we need to add some tokens to nllb vocabulary

# CHECK UNK NUMBER 
texts_with_unk = [
    text for text in tqdm(all_sentences.krc) 
    if tokenizer.unk_token_id in tokenizer(text).input_ids
]
print(len(texts_with_unk))
s = random.sample(texts_with_unk, 5)

# tk = tokenizer('- Некди теnиз былай тузлу? – деb сейирсинnенди.').input_ids
# tokenizer.decode(tk)

# We can see that out of 260 602 texts, 21642 contain an “unknown symbol” after tokenization. 
# Most of these cases seem to be associated with non-standard punctuation marks, and there is a reason for that: 
#   the NLLB team preprocessed their texts before training the tokenizer and the model. 
#   The code for preprocessing (adapted from the Stopes repo) looks like this:

mpn = MosesPunctNormalizer(lang="en")
mpn.substitutions = [
    (re.compile(r), sub) for r, sub in mpn.substitutions
]

def get_non_printing_char_replacer(replace_by: str = " "):
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
  
  
texts_with_unk_normed = [
    text for text in tqdm(texts_with_unk) 
    if tokenizer.unk_token_id in tokenizer(preproc(text)).input_ids
]
print(len(texts_with_unk_normed))  # 0


# In this case we can say that unk tokens it is just a punctuation.


# 5.EXPANDING VOCABULARY ------------------------------------------------
chars_cnt = Counter(c for t in all_sentences_krc for c in t)

with open(ALL_TEXT_PLAIN, 'w') as f:
    for i, text in enumerate(all_sentences_krc):
        print(text, file=f)

# I chose the vocabulary size to be 8K intuitively,
# because such a number of tokens can potentially cover the most important roots and suffixes 
# in the language (to compare: NLLB vocabulary for 200 languages has 256000 tokens, but many of them are used by a lot of different languages). 
# All the other parameters are not very important.

spm.SentencePieceTrainer.train(
    input=ALL_TEXT_PLAIN,
    model_prefix=SPM_PREFIX,
    vocab_size=2**13,  # 8K
    character_coverage = 1,
    # num_threads=16,
    train_extremely_large_corpus=False,
    add_dummy_prefix=False,
    max_sentencepiece_length=512,
    max_sentence_length=4192*4,
    pad_id=0,
    eos_id=1,
    unk_id=2,
    bos_id=-1,
    required_chars=''.join([k for k, v in chars_cnt.most_common() if v >= 3 and k not in ' ']),
)

# After training a QM tokenizer, I perform a “surgical operation” 
# with it: extracting the sentencepiece model from the standard NLLB tokenizer and enriching it from all tokens 
# from the QM tokenizer that have been missing from the NLLB tokenizer (based on the example from the sentencepiece repo).

# reading the NLLB and the QM sentencepiece models into a native format
sp_trained = spm.SentencePieceProcessor(model_file=f'{SPM_PREFIX}.model')
added_spm = sp_pb2_model.ModelProto()
added_spm.ParseFromString(sp_trained.serialized_model_proto())
old_spm = sp_pb2_model.ModelProto()
old_spm.ParseFromString(tokenizer.sp_model.serialized_model_proto())

# adding the missing tokens to the NLLB sentencepiece model
nllb_tokens_set = {p.piece for p in old_spm.pieces}
print(len(nllb_tokens_set))

prev_min_score = old_spm.pieces[-1].score
for p in added_spm.pieces:
    piece = p.piece
    if piece not in nllb_tokens_set:
        new_p = sp_pb2_model.ModelProto().SentencePiece()
        new_p.piece = piece
        # for all new tokens, I'll set a lower score (priority)
        new_p.score = p.score + prev_min_score
        old_spm.pieces.append(new_p)
        
print(len(old_spm.pieces))

# saving the result to disk
with open(NEW_SPM_NAME, 'wb') as f:
    f.write(old_spm.SerializeToString())


# Finally, I need to update the neural network weights: 
#   add new embeddings for the freshly added tokens. 
#   In NLLB, the token embeddings reside in the parameter called shared. 
#   It is used both in the encoder and decoder input embeddings and in the last decoder layer that predicts the distribution of the next token.
# 
# By default, the embeddings for the new tokens are initialized randomly. 
# Instead, I re-initialize each one with the average of the embeddings of the old tokens that corresponded to the new token 
# (or if there are none, with the embedding of the <unk> token). This slightly improves the training speed, 
# because the newly created tokken embeddings are already informative.

# 6.UPDATING TOKENIZER VOCABLUARY ------------------------------------------------

tokenizer_old = NllbTokenizer.from_pretrained(MODEL_NAME)
print(len(tokenizer_old))
print(tokenizer_old.convert_ids_to_tokens([256202, 256203]))

tokenizer = NllbTokenizer.from_pretrained(MODEL_NAME, vocab_file=NEW_SPM_NAME)
print(len(tokenizer))
print(tokenizer.convert_ids_to_tokens([262923, 262924]))

n_new_tokens = len(tokenizer) - len(tokenizer_old)
print(n_new_tokens)


def fix_tokenizer(tokenizer, new_lang=LANG_UNICODE):
    """ Add a new language token to the tokenizer vocabulary (this should be done each time after its initialization) """
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


# Check
print(tokenizer.convert_ids_to_tokens([262923, 262924, 262925])) # ['zul_Latn', 'krc_Cyrl', '<mask>']
print(tokenizer.convert_tokens_to_ids(['zul_Latn', 'krc_Cyrl', '<mask>'])) # [262923, 262924, 262925]

print(len(tokenizer_old), len(tokenizer)) # 256204, 268559
added_vocab = set(tokenizer.get_vocab()).difference(set(tokenizer_old.get_vocab()))
print(len(added_vocab))  # 6721


text = random.choice(all_sentences_krc)

print(tokenizer_old.tokenize(text))
print(tokenizer.tokenize(text))



random.seed(1)
sample = random.sample(list(all_sentences_krc), 10000)
sample[:10]

pd.DataFrame({
    'old': [len(tokenizer_old.tokenize(text)) for text in sample],
    'new': [len(tokenizer.tokenize(text)) for text in sample]
}).describe()
stats

new_tot_len, tot_len = 0, 0
for text in sample:
    for tok in tokenizer.tokenize(text):
        s = len(tok)
        tot_len += s
        if tokenizer.convert_tokens_to_ids(tok) > len(tokenizer_old.sp_model):
            new_tot_len += s
            
print(new_tot_len / tot_len)



# 7. UPDATING THE MODEL EMBEDDINGS ------------------------------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
# if torch.cuda.is_available():
#     model.cuda()

model.model.shared

model.resize_token_embeddings(len(tokenizer))


moved_tokens = list(tokenizer_old.lang_code_to_id) + ['<mask>']

model.model.shared.weight.data[tokenizer.convert_tokens_to_ids(moved_tokens)] = model.model.shared.weight.data[tokenizer_old.convert_tokens_to_ids(moved_tokens)]



# Because we have added one more language, its id must be computed separately, e.g. as an average of related languages.
# 'crh_Latn': 256039 - Къырымтатар тил
# 'tat_Cyrl': 256171 - Татар тил
# 'kaz_Cyrl': 256089 - Къазакъ тил
model.model.shared.weight.data[tokenizer.convert_tokens_to_ids(LANG_UNICODE)] = (
    model.model.shared.weight.data[tokenizer_old.convert_tokens_to_ids('tat_Cyrl')] * 0.55
    + model.model.shared.weight.data[tokenizer_old.convert_tokens_to_ids('crh_Latn')] * 0.35
    + model.model.shared.weight.data[tokenizer_old.convert_tokens_to_ids('kaz_Cyrl')] * 0.1
)



# re-initializing the new embeddings
for t in tqdm(added_vocab):
    tt = tokenizer_old(t, add_special_tokens=False).input_ids
    if len(tt) == 0:
        tt = [tokenizer_old.unk_token_id]
    idx = tokenizer.convert_tokens_to_ids(t)
    model.model.shared.weight.data[idx] = model.model.shared.weight.data[tt].mean(0)

# Save Model
model.save_pretrained(MODEL_PATH_RAW)
tokenizer.save_pretrained(MODEL_PATH_RAW)

model1 = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH_RAW) 
tokenizer1 = NllbTokenizer.from_pretrained(MODEL_PATH_RAW)
# tokenizer1 = NllbTokenizer.from_pretrained(MODEL_PATH_RAW, rebuild=True)
fix_tokenizer(tokenizer1)


model.model.shared
model1.model.shared

# 8.CHECK MODEL TRANSLATION ABILITY ------------------------------------------------
article_hi = "संयुक्त राष्ट्र के प्रमुख का कहना है कि सीरिया में कोई सैन्य समाधान नहीं है"

# translate Hindi to French
tokenizer.src_lang = "hin_Deva"
encoded_hi = tokenizer(article_hi, return_tensors="pt")
generated_tokens = model.generate(
    **encoded_hi,
    forced_bos_token_id=tokenizer.lang_code_to_id["fra_Latn"],
    max_length=30,
    num_beams=1
)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire dans la Syrie."

# translate Hindi to Russian
generated_tokens = model.generate(
    **encoded_hi,
    forced_bos_token_id=tokenizer.lang_code_to_id["rus_Cyrl"],
    max_length=30,
    num_beams=5,
    repetition_penalty=30.0,
)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
# => "Le chef de l 'ONU affirme qu 'il n 'y a pas de solution militaire dans la Syrie."

# translate Qarachay-Malqar to Russian 
# qm, ru = random.choice(all_pairs)
df = all_sentences.sample(1)[['krc', 'rus']]
qm = df.krc.to_list()
ru = df.rus.to_list()

tokenizer.src_lang = LANG_UNICODE
encoded_hi = tokenizer(qm, return_tensors="pt")

generated_tokens = model.generate(
    **encoded_hi,
    forced_bos_token_id=tokenizer.lang_code_to_id["rus_Cyrl"],
    max_length=30,
    num_beams=5,
    repetition_penalty=30.0
)
tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

