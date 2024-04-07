# The first neural machine translation system for the Qarachay Malqar language
 
Authors: Bogdan Tewunalany, Ali Berberov

We collect 288532 parallel sentences between russian and Qarachay-Malqar languages. There were two ways to make the translator:  
  * First, We made transformwer on Tensorflow/Keras on R. But fine-tuning of existing model is better.    
  * Second, We used pre-trained model nllb-200 and fine-tuned it (4.Jupyter_Colab).  

[Model](https://huggingface.co/TSjB/NLLB-201-600M-QM-V2)

Where to use:
[HF](https://huggingface.co/spaces/TSjB/Qarachay-Malqar_translator) and [site](https://tsjb-qarachay-malqar-translator.hf.space)

Parallel corpora:
[As is](https://huggingface.co/datasets/TSjB/qarachay-malqar_russian_parallel_corpora) and [Dialects free](https://huggingface.co/datasets/TSjB/qarachay-malqar_russian_parallel_corpora_dialectic-free)

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
