# The first neural machine translation system for the Qarachay Malqar language on R
 
Authors: Bogdan Tewnalany, Ali Berberov

Created transformer model on R and trained it on 27235 parallel sentences between russian and Qarachay-Malqar languages. It is not enough for good prediction, result is bad. So, we used pre-trained model mbart-50 and fine-tuned it (TBSj/Qarachay_Malqar_translator_python).

Nowdays, we are collecting more sentences to improve our result.

Model for translation from russian to qarachay-malqar: https://huggingface.co/TSjB/mbart-large-52-ru-qm-v1

Model for translation from qarachay-malqar to russian: https://huggingface.co/TSjB/mbart-large-52-qm-ru-v1

Where to use:

https://huggingface.co/spaces/TSjB/QM_RU_translator

https://tsjb-qm-ru-translator.hf.space/

Telegram: https://t.me/QMKochBot
