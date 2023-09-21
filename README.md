# The first neural machine translation system for the Qarachay Malqar language on R
 
Authors: Bogdan Tewnalany, Ali Berberov

Created transformer model on R and trained it on 27235 parallel sentences between russian and Qarachay-Malqar languages. It is not enough for good prediction, result is bad. So, we used pre-trained model mbart-50 and fine-tuned it (TBSj/Qarachay_Malqar_translator_python).  
Nowdays, we are collecting more sentences to improve our result.

Model for translation from russian to qarachay-malqar: https://huggingface.co/TSjB/mbart-large-52-ru-qm-v1

Model for translation from qarachay-malqar to russian: https://huggingface.co/TSjB/mbart-large-52-qm-ru-v1

Where to use:

https://huggingface.co/spaces/TSjB/QM_RU_translator

https://tsjb-qm-ru-translator.hf.space/

Authors telegram:
Bogdan: https://t.me/bogdan_tewnalany
Ali: https://t.me/ali_berberov

# License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
