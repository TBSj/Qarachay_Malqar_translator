# The first neural machine translation system for the Qarachay Malqar language
 
Authors: Bogdan Tewunalany, Ali Berberov

We collect 260584 parallel sentences between russian and Qarachay-Malqar languages. There were two ways to make the translator:  
  * First, We made transformwer on Tensorflow/Keras on R. But fine-tuning of existing model is better.    
  * Second, We used pre-trained model nllb-200 and fine-tuned it.  

[Model for translation from russian to qarachay-malqar](https://huggingface.co/TSjB/mbart-large-52-ru-qm-v2)

[Model for translation from qarachay-malqar to russian](https://huggingface.co/TSjB/mbart-large-52-qm-ru-v2)

Where to use:
[HF](https://huggingface.co/spaces/TSjB/QM_RU_translator) and [site](https://tsjb-qm-ru-translator.hf.space/?)

Authors telegram:
[Bogdan](https://t.me/bogdan_tewunalany) and
[Ali](https://t.me/ali_bulat1990)  

# License
Shield: [![CC BY-NC-SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License][cc-by-nc-sa].

[![CC BY-NC-SA 4.0][cc-by-nc-sa-image]][cc-by-nc-sa]

[cc-by-nc-sa]: http://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-image]: https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg
