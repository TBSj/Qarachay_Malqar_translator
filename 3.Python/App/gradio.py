# 1. Libraries
import gradio as gr
import torch
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
import pandas as pd
from datasets import load_dataset

# 2. Constants
MODEL_PATH = 'TSjB/NLLB-201-600M-QM-V2'
DATA_PATH = "TSjB/dictionary_krc_rus"

# LANGUAGE = pd.DataFrame({"language": ["Къарачай-Малкъар тил", "Русский язык", "English", "Türk dili"], "token": ["krc_Cyrl", "rus_Cyrl", "eng_Latn", "tur_Latn"]})
LANGUAGE = pd.DataFrame({"language": ["Къарачай-Малкъар тил", "Русский язык"], "token": ["krc_Cyrl", "rus_Cyrl"]})
DIALECT = pd.DataFrame({"dialect": ["дж\ч", "ж\ч", "з\ц"], "short_name": ["qrc", "hlm", "mqr"]})
TYPE = pd.DataFrame({"krc": ["Кёчюрюўчю", "Сёзлюк"], "rus": ["Переводчик", "Словарь"], "eng": ["Translator", "Dictionary"], "tur": ["Çevirmen", "Sözlük."], "short_name": ["translator", "dictionary"]})

SYSTEM_LANG = "rus"
# NAMES = pd.DataFrame({
#    "id": ["title", "from", "to", "your_sent", "transl_sent", "dialect", "translate", "annotation"],
#    "krc": ["# Къарачай-Малкъар кёчюрюўчю", "тилден", "тилге", "Мында джаз...", "Кёчюрюлгени", "Къарачай-Малкъарны диалекти", "Кёчюр","Къарачай-Малкъар тилде биринчи кёчюрюўчюдю. [Богдан Теўуналаны](https://t.me/bogdan_tewunalany), [Али Берберлени](https://t.me/ali_berberov) къурагъандыла\n\nМодель Орус бла Къарачай-Малкъар тилледе юйрене тургъаны себебли, Къарачай-Малкъар кёчюрюў башха тиллеге да осал болургъа боллукъду."],
#    "rus": ["# Карачаево-Балкарский переводчик", "из", "на", "Напишите здесь...", "Переведённый текст", "Карачаево-Балкарский диалект", "Перевести","Первый переводчик на карачаево-балкарский язык. Создан [Богданом Теунаевым](https://t.me/bogdan_tewunalany), [Али Берберовым](https://t.me/ali_berberov)\n\nТак как модель обучалась на парах Русский и Карачаево-Балкарский, то Карачаево-Балкарский перевод для остальных языков может быть хуже."],
#    "tur": ["# Karaçay-Malkar tercümanı", "dilden", "dile", "Buraya yaz...", "Çevrilmiş metin burada", "Karaçay-Malkar lehçesi", "Tercüme edin", "İlk çevirmen. [Bogdan Tewunalanı](https://t.me/bogdan_tewunalany), [Ali Berberov](https://t.me/ali_berberov) tarafından oluşturuldu\n\nModel Rusça ve Karaçay-Malkar çiftleri halinde eğitildiğinden, diğer diller için Karaçay-Malkar çevirisi daha kötü olabilir."],
#    "eng": ["# Qarachay-Malqar translator", "from", "to", "Write here...", "Translated text is here", "Qarachay-Malqar dialect", "Translate", "The first translator. Created by [Bogdan Tewunalany](https://t.me/bogdan_tewunalany), [Ali Berberov](https://t.me/ali_berberov)\n\nSince the model was trained in pairs of Russian and Qarachay-Malqar, the Qarachay-Malqar translation for other languages may be worse."]
# })
NAMES = pd.DataFrame({
   "id": ["title", "type", "from", "to", "your_sent", "transl_sent", "dialect", "translate", "annotation", "word_absence"],
   "krc": ["# Къарачай-Малкъар сёзлюк бла кёчюрюўчю", "Тюрлюсю", "тилден", "тилге", "Мында джаз...", "Кёчюрюлгени", "Къарачай-Малкъарны диалекти", "Кёчюр","Къарачай-малкъар, орус тиллени арасында биринчи кёчюрюўчюдю. Сёзлюк да ичине салыннганды.\n\n[Богдан Теўуналаны](https://t.me/bogdan_tewunalany), [Али Берберлени](https://t.me/ali_berberov) къурагъандыла\n\nСоинвестированиени эмда спонсорлукъ болушлукъну юсюнден [Али Берберовгъа](https://t.me/ali_berberov) соругъуз", "Сорулгъаны сёзлюкде табылмагъанды."],
   "rus": ["# Карачаево-балкарский словарь и переводчик", "Тип", "из", "на", "Напишите здесь...", "Переведённый текст", "Карачаево-балкарский диалект", "Перевести","Первый переводчик между карачаево-балкарским и русским языками. Также встроен словарь для отдельных слов или коротких фраз.\n\nРазработчики: [Богдан Теунаев](https://t.me/bogdan_tewunalany), [Али Берберов](https://t.me/ali_berberov)\n\nПо вопросам соинвестирования и спонсорской поддержки обращайтесь к [Али Берберову](https://t.me/ali_berberov)", "Запрашиваемое в словаре не найдено."],
   "tur": ["# Karaçayca-Balkarca sözlük ve çevirmen", "Tür", "dilden", "dile", "Buraya yaz...", "Çevrilmiş metin burada", "Karaçay-Malkar lehçesi", "Tercüme edin", "Karaçay-Balkarca ve Rusça dilleri arasındaki ilk çevirmen. Tek tek kelimeler veya kısa ifadeler için bir sözlük de yerleşiktir.\n\nGeliştiriciler: [Bogdan Tewunalanı](https://t.me/bogdan_tewunalany), [Ali Berberov](https://t.me/ali_berberov)\n\nOrtak yatırım ve sponsorluk ile ilgili sorularınız için [Ali Berberov](https://t.me/ali_berberov) ile iletişime geçin", "Sorge sözlükte bulunmuyor."],
   "eng": ["# Qarachay-Malqar dictionary and translator", "Type", "from", "to", "Write here...", "Translated text is here", "Qarachay-Malqar dialect", "Translate", "The first translator between Qarachay-Malqar and Russian languages. A dictionary for individual words or short phrases is also built in.\n\nDevelopers: [Bogdan Tewunalany](https://t.me/bogdan_tewunalany), [Ali Berberov](https://t.me/ali_berberov)\n\nFor co-investment and sponsorship, please contact [Ali Berberov] (https://t.me/ali_berberov)", "The requested was not found in the dictionary."]
})


OUTPUT_ROW_BY_EVERY_DICTIONARY = 15

FILEPATH_SOURCE_PREPARED = "1.Data/Dictionary"
# dictionary = pd.read_csv("%s/dictionary.csv" % FILEPATH_SOURCE_PREPARED, sep = ";")

# 3. Upload
dictionary = load_dataset(DATA_PATH)
dictionary = pd.DataFrame(dictionary['train'])

dictionary["soz"] = dictionary.soz.str.upper()
dictionary["soz_l"] = dictionary.soz.str.lower()
dictionary["belgi_l"] = dictionary.belgi.str.lower()

dictionary_qm = dictionary[dictionary.til == "krc"]
dictionary_ru = dictionary[dictionary.til == "rus"]


tokenizer = NllbTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)

# 4. Fix tokenizer
def fixTokenizer(tokenizer, new_lang='krc_Cyrl'):
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

fixTokenizer(tokenizer)

# 5. Change letters

def fromModel(str, dialect = "qrc"):
  if dialect == "qrc":
      str = str.replace("тюйюл", "тюл") 
      str = str.replace("Тюйюл", "Тюл") 
      str = str.replace("уку", "гылын qуш") 
      str = str.replace("Уку", "Гылын qуш") 
      str = str.replace("хораз", "гугурукку") 
      str = str.replace("Хораз", "Гугурукку") 
      str = str.replace("юзмез", "qум") 
      str = str.replace("Юзмез", "Qум") 
      str = str.replace("jиля", "jыла") 
      str = str.replace("Jиля", "Jыла") 
      str = str.replace("ярабий", "арабин") 
      str = str.replace("арабий", "арабин") 
      str = str.replace("Ярабий", "Арабин") 
      str = str.replace("Арабий", "Арабин") 
      str = str.replace("нтта", "нтда") 
      str = str.replace("ртте", "ртде") 
      str = str.replace("jамауат", "jамаgат") 
      str = str.replace("jамаwат", "jамаgат") 
      str = str.replace("Jамауат", "Jамаgат") 
      str = str.replace("Jамаwат", "Jамаgат") 
      str = str.replace("шуёх", "шох") 
      str = str.replace("Шуёх", "Шох") 
      str = str.replace("шёндю", "бусаgат") 
      str = str.replace("Шёндю", "Бусаgат") 
      str = str.replace("уgай", "оgай") 
      str = str.replace("Уgай", "Оgай") 
      # str = str.replace("терк", "тез") 
      str = str.replace("саnа", "сенnе") 
      str = str.replace("сеnе", "сенnе") 
      str = str.replace("Саnа", "Сенnе") 
      str = str.replace("Сеnе", "Сенnе") 
      str = str.replace("маnа", "менnе") 
      str = str.replace("меnе", "менnе") 
      str = str.replace("Маnа", "Менnе") 
      str = str.replace("Меnе", "Менnе") 
      str = str.replace("аяq jол", "jахтана") 
      str = str.replace("Аяq jол", "Jахтана") 
      str = str.replace("сыbат", "сыфат") 
      str = str.replace("Сыbат", "Сыфат") 
      str = str.replace("b", "б") 
      str = str.replace("q", "къ") 
      str = str.replace("Q", "Къ") 
      str = str.replace("g", "гъ") 
      str = str.replace("G", "Гъ") 
      str = str.replace("j", "дж") 
      str = str.replace("J", "Дж") 
      str = str.replace("w", "ў") 
      str = str.replace("W", "Ў") 
      str = str.replace("n", "нг") 
      str = str.replace("N", "Нг")
  elif dialect == "hlm":
      str = str.replace("тюл", "тюйюл") 
      str = str.replace("Тюл", "Тюйюл") 
      str = str.replace("гылын qуш", "уку") 
      str = str.replace("Гылын qуш", "Уку") 
      str = str.replace("гугурукку", "хораз") 
      str = str.replace("Гугурукку", "Хораз") 
      str = str.replace("qум", "юзмез") 
      str = str.replace("Qум", "Юзмез") 
      str = str.replace("jыла", "jиля") 
      str = str.replace("Jыла", "Jиля") 
      str = str.replace("арабин", "ярабий") 
      str = str.replace("арабий", "ярабий") 
      str = str.replace("Арабин", "Ярабий") 
      str = str.replace("Арабий", "Ярабий") 
      str = str.replace("нтда", "нтта") 
      str = str.replace("ртде", "ртте") 
      str = str.replace("jамаgат", "jамаwат") 
      str = str.replace("Jамаgат", "Jамаwат") 
      str = str.replace("шох", "шуёх") 
      str = str.replace("Шох", "Шуёх") 
      str = str.replace("бусаgат", "шёндю") 
      str = str.replace("Бусаgат", "Шёндю") 
      str = str.replace("оgай", "уgай") 
      str = str.replace("Оgай", "Уgай") 
      str = str.replace("тез", "терк") 
      str = str.replace("сенnе", "саnа") 
      str = str.replace("сеnе", "саnа") 
      str = str.replace("Сенnе", "Саnа") 
      str = str.replace("Сеnе", "Саnа") 
      str = str.replace("менnе", "маnа") 
      str = str.replace("меnе", "маnа") 
      str = str.replace("Менnе", "Маnа") 
      str = str.replace("Меnе", "Маnа") 
      str = str.replace("jахтана", "аяq jол") 
      str = str.replace("Jахтана", "аяq jол") 
      str = str.replace("хо", "хаw") 
      str = str.replace("Хо", "Хаw") 
      str = str.replace("сыbат", "сыфат") 
      str = str.replace("Сыbат", "Сыфат") 
      str = str.replace("b", "п") 
      str = str.replace("q", "къ") 
      str = str.replace("Q", "Къ") 
      str = str.replace("g", "гъ") 
      str = str.replace("G", "Гъ") 
      str = str.replace("j", "ж") 
      str = str.replace("J", "Ж") 
      str = str.replace("w", "ў") 
      str = str.replace("W", "Ў") 
      str = str.replace("n", "нг") 
      str = str.replace("N", "Нг")
  elif dialect == "mqr":
      str = str.replace("тюл", "тюйюл") 
      str = str.replace("Тюл", "Тюйюл") 
      str = str.replace("гылын qуш", "уку") 
      str = str.replace("Гылын qуш", "Уку") 
      str = str.replace("гугурукку", "хораз") 
      str = str.replace("Гугурукку", "Хораз") 
      str = str.replace("qум", "юзмез") 
      str = str.replace("Qум", "Юзмез") 
      str = str.replace("jыла", "jиля") 
      str = str.replace("Jыла", "Jиля") 
      str = str.replace("арабин", "ярабий") 
      str = str.replace("арабий", "ярабий") 
      str = str.replace("Арабин", "Ярабий") 
      str = str.replace("Арабий", "Ярабий") 
      str = str.replace("нтда", "нтта") 
      str = str.replace("ртде", "ртте") 
      str = str.replace("jамаgат", "жамаwат") 
      str = str.replace("Jамаgат", "Жамаwат") 
      str = str.replace("шох", "шуёх") 
      str = str.replace("Шох", "Шуёх") 
      str = str.replace("бусаgат", "шёндю") 
      str = str.replace("Бусаgат", "Шёндю") 
      str = str.replace("оgай", "уgай") 
      str = str.replace("Оgай", "Уgай") 
      str = str.replace("тез", "терк") 
      str = str.replace("сенnе", "саnа") 
      str = str.replace("сеnе", "саnа") 
      str = str.replace("Сенnе", "Саnа") 
      str = str.replace("Сеnе", "Саnа") 
      str = str.replace("менnе", "маnа") 
      str = str.replace("меnе", "маnа") 
      str = str.replace("Менnе", "Маnа") 
      str = str.replace("Меnе", "Маnа") 
      str = str.replace("jахтана", "аяq jол") 
      str = str.replace("Jахтана", "аяq jол") 
      str = str.replace("хо", "хаw") 
      str = str.replace("Хо", "Хаw") 
      str = str.replace("сыbат", "сыфат") 
      str = str.replace("Сыbат", "Сыфат") 
      str = str.replace("b", "п") 
      str = str.replace("q", "къ") 
      str = str.replace("Q", "Къ") 
      str = str.replace("g", "гъ") 
      str = str.replace("G", "Гъ") 
      str = str.replace("j", "з") 
      str = str.replace("J", "З") 
      str = str.replace("w", "ў") 
      str = str.replace("W", "Ў") 
      str = str.replace("n", "нг") 
      str = str.replace("N", "Нг") 
      str = str.replace("ч", "ц") 
      str = str.replace("Ч", "Ц") 
      str = str.replace("п", "ф") 
      str = str.replace("П", "Ф") 
      str = str.replace("къ|гъ", "х")
  return str

    
def toModel(str):
    str = str.replace("дж", "j") 
    str = str.replace("Дж", "J") 
    str = str.replace("ДЖ", "J") 
    str = str.replace("ж", "j") 
    str = str.replace("Ж", "J")
    str = str.replace("себеп", "себеb") 
    str = str.replace("себеб", "себеb") 
    str = str.replace("Себеп", "Себеb") 
    str = str.replace("Себеб", "Себеb") 
    str = str.replace("тюйюл", "тюл") 
    str = str.replace("Тюйюл", "Тюл") 
    str = str.replace("уку", "гылын qуш") 
    str = str.replace("Уку", "Гылын qуш") 
    str = str.replace("хораз", "гугурукку") 
    str = str.replace("Хораз", "Гугурукку") 
    str = str.replace("юзмез", "qум") 
    str = str.replace("Юзмез", "Qум") 
    str = str.replace("арап", "араb") 
    str = str.replace("араб", "араb") 
    str = str.replace("Арап", "Араb")
    str = str.replace("Араб", "Араb")
    str = str.replace("jиля", "jыла") 
    str = str.replace("jыла", "jыла") 
    str = str.replace("jыла", "jыла") 
    str = str.replace("Jиля", "Jыла") 
    str = str.replace("Jыла", "Jыла") 
    str = str.replace("Jыла", "Jыла") 
    str = str.replace("ярабий", "арабин") 
    str = str.replace("арабий", "арабин") 
    str = str.replace("Ярабий", "Арабин") 
    str = str.replace("Арабий", "Арабин") 
    str = str.replace("нтта", "нтда") 
    str = str.replace("ртте", "ртде") 
    str = str.replace("jамагъат", "jамаgат") 
    str = str.replace("jамауат", "jамаgат") 
    str = str.replace("jамагъат", "jамаgат")
    str = str.replace("jамауат", "jамаgат")
    str = str.replace("Jамагъат", "Jамаgат") 
    str = str.replace("Jамауат", "Jамаgат") 
    str = str.replace("Jамагъат", "Jамаgат") 
    str = str.replace("Jамаўат", "Jамаgат") 
    str = str.replace("шуёх", "шох") 
    str = str.replace("Шуёх", "Шох") 
    str = str.replace("шёндю", "бусаgат") 
    str = str.replace("бусагъат", "бусаgат") 
    str = str.replace("Шёндю", "Бусаgат") 
    str = str.replace("Бусагъат", "Бусаgат") 
    str = str.replace("угъай", "оgай")
    str = str.replace("огъай", "оgай")
    str = str.replace("Угъай", "Оgай") 
    str = str.replace("Огъай", "Оgай") 
    # str = str.replace("терк", "тез") 
    # str = str.replace("терк", "тез") 
    str = str.replace("санга", "сенnе") 
    str = str.replace("сенге", "сенnе") 
    str = str.replace("сеннге", "сенnе") 
    str = str.replace("Санга", "Сенnе") 
    str = str.replace("Сеннге", "Сенnе") 
    str = str.replace("Сенге", "Сенnе") 
    str = str.replace("манга", "менnе") 
    str = str.replace("меннге", "менnе") 
    str = str.replace("менге", "менnе") 
    str = str.replace("Манга", "Менnе") 
    str = str.replace("Меннге", "Менnе") 
    str = str.replace("Менге", "Менnе") 
    str = str.replace("аякъ jол", "jахтана") 
    str = str.replace("аякъ jол", "jахтана") 
    str = str.replace("jахтана", "jахтана") 
    str = str.replace("jахтана", "jахтана") 
    str = str.replace("Аякъ jол", "Jахтана") 
    str = str.replace("Аякъ jол", "Jахтана") 
    str = str.replace("Jахтана", "Jахтана") 
    str = str.replace("Jахтана", "Jахтана") 
    str = str.replace("къамж", "qамыzh") 
    str = str.replace("къамыж", "qамыzh") 
    str = str.replace("Къамж", "Qамыzh") 
    str = str.replace("Къамыж", "Qамыzh") 
    str = str.replace("къымыж", "qымыzh") 
    str = str.replace("къымыж", "qымыzh") 
    str = str.replace("Къымыж", "Qымыzh") 
    str = str.replace("Къымыж", "Qымыzh") 
    str = str.replace("хау", "хо") 
    str = str.replace("хаў", "хо") 
    str = str.replace("Хау", "Хо") 
    str = str.replace("Хаў", "Хо") 
    str = str.replace("уа", "wa") 
    str = str.replace("ўа", "wa") 
    str = str.replace("Уа", "Wa") 
    str = str.replace("Ўа", "Wa") 
    str = str.replace("п", "b") 
    str = str.replace("б", "b") 
    str = str.replace("къ", "q") 
    str = str.replace("Къ", "Q") 
    str = str.replace("КЪ", "Q") 
    str = str.replace("гъ", "g") 
    str = str.replace("Гъ", "G") 
    str = str.replace("ГЪ", "G") 
    str = str.replace("ц", "ч") 
    str = str.replace("Ц", "Ч") 
    str = str.replace("ф", "п") 
    str = str.replace("сыпат", "сыфат") 
    str = str.replace("Сыпат", "Сыфат") 
    str = str.replace("Ф", "П") 
    str = str.replace("(?<=[аыоуэеиёюя])у(?=[аыоуэеиёюя])|(?<=[аыоуэеиёюя])ў(?=[аыоуэеиёюя])|(?<=[АЫОУЭЕИЁЮЯ])у(?=[АЫОУЭЕИЁЮЯ])|(?<=[АЫОУЭЕИЁЮЯ])ў(?=[АЫОУЭЕИЁЮЯ])", "w")
    str = str.replace("(?<=[аыоуэеиёюя])у|(?<=[аыоуэеиёюя])ў|(?<=[АЫОУЭЕИЁЮЯ])у|(?<=[АЫОУЭЕИЁЮЯ])ў", "w")
    # str = str.replace("у(?=[аыоуэеиёюя])|ў(?=[аыоуэеиёюя])|у(?=[АЫОУЭЕИЁЮЯ])|ў(?=[АЫОУЭЕИЁЮЯ])", "w") 
    # str = str.replace("У(?=[аыоуэеиёюя])|Ў(?=[аыоуэеиёюя])|У(?=[АЫОУЭЕИЁЮЯ])|Ў(?=[АЫОУЭЕИЁЮЯ])", "W") 
    str = str.replace("zh", "ж") 
    str = str.replace("нг", "n") 
    str = str.replace("Нг", "  N")
    str = str.replace("НГ", "  N")
    return str

# 6. Translate function
def translatePy(text, src_lang='rus_Cyrl', tgt_lang='krc_Cyrl',
    a=32, b=3, max_input_length=1024, num_beams=3, **kwargs
):
    """Turn a text or a list of texts into a list of translations"""
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang
    inputs = tokenizer(
        text, return_tensors='pt', padding=True, truncation=True,
        max_length=max_input_length
    )
    model.eval() # turn off training mode
    result = model.generate(
        **inputs.to(model.device),
        forced_bos_token_id=tokenizer.convert_tokens_to_ids(tgt_lang),
        max_new_tokens=int(a + b * inputs.input_ids.shape[1]),
        num_beams=num_beams, **kwargs
    )
    return tokenizer.batch_decode(result, skip_special_tokens=True)[0]


def translateProcess(text, from_, to, dialect):
    # print(from_)
    # print(to)
    # print(dialect)
    if from_ == 'krc_Cyrl':
      text = toModel(text)
    
    str_ = translatePy(text, src_lang = from_, tgt_lang = to)
    
    if to == 'krc_Cyrl':
      str_ = fromModel(str_, dialect = dialect)
    
    return str_
  
# 7. Dictionary function
def dictionaryDisp(from_, text):
  str_l = text.lower()
  filter_ = r"\W+" + str_l + r"|^" + str_l
  
  df_from_to = pd.DataFrame() 
  df_to_from = pd.DataFrame()
  
  if from_ == 'krc_Cyrl':
    df_from_to = dictionary_qm.copy()
    df_to_from = dictionary_ru.copy()
  elif from_ == 'rus_Cyrl':
    df_from_to = dictionary_ru.copy()
    df_to_from = dictionary_qm.copy()
    
  sozluk_1 = df_from_to[df_from_to.soz_l.str.startswith(str_l)]
  # Select rows based on the sequence and output
  sozluk_1 = sozluk_1.iloc[:OUTPUT_ROW_BY_EVERY_DICTIONARY]
  
  sozluk_2 = df_from_to[df_from_to.belgi_l.str.contains(filter_, regex=True)]
  sozluk_2 = sozluk_2.iloc[:OUTPUT_ROW_BY_EVERY_DICTIONARY]
  
  sozluk_3 = df_to_from[df_to_from.belgi_l.str.contains(filter_, regex=True)]
  sozluk_3 = sozluk_3.iloc[:OUTPUT_ROW_BY_EVERY_DICTIONARY]
  
  # Concatenate the DataFrames and drop duplicates
  sozluk = pd.concat([sozluk_1, sozluk_2, sozluk_3], ignore_index=True).drop_duplicates()[["soz", "belgi"]]
  sozluk = [x.soz + " ----- " + x.belgi + "\n\n----------\n\n" for x in sozluk.itertuples()]
  sozluk = "".join(sozluk)
  
  return sozluk
  # len(sozluk)
  
  
# 8. Output function 
def out(text, from_, to, dialect, type_):
  type_col = SYSTEM_LANG
  
  if dialect == "" or dialect is None:
    dialect = "дж\ч"
  if from_ == "" or from_ is None:
    from_ = "Русский язык"
  if to == "" or to is None:
    to = "Къарачай-Малкъар тил"
  if type_ == "" or type_ is None:
    type_ = "Кёчюрюўчю"
    type_col = "krc"
    
  from_ = "".join(LANGUAGE[LANGUAGE.language == from_].token.to_list())
  to = "".join(LANGUAGE[LANGUAGE.language == to].token.to_list())
  dialect = "".join(DIALECT[DIALECT.dialect == dialect].short_name.to_list())
  type_ = "".join(TYPE[TYPE[type_col] == type_].short_name.to_list())
    
    
  if type_ == "dictionary":
    str_ = dictionaryDisp(from_, text)
    if(len(str_) == 0):
      str_ = NAMES[NAMES.id == "word_absence"][SYSTEM_LANG].values[0]
  elif type_ == "translator":
    str_ = translateProcess(text, from_, to, dialect)
      # str_ = "myaf"
  
  return(str_)

# 9. Definition ui
_title = "".join(NAMES[NAMES.id == "title"][SYSTEM_LANG].to_list())
_type = "".join(NAMES[NAMES.id == "type"][SYSTEM_LANG].to_list())
_from = "".join(NAMES[NAMES.id == "from"][SYSTEM_LANG].to_list())
_to = "".join(NAMES[NAMES.id == "to"][SYSTEM_LANG].to_list())
_your_sent = "".join(NAMES[NAMES.id == "your_sent"][SYSTEM_LANG].to_list())
_transl_sent = "".join(NAMES[NAMES.id == "transl_sent"][SYSTEM_LANG].to_list())
_dialect = "".join(NAMES[NAMES.id == "dialect"][SYSTEM_LANG].to_list())
_translate = "".join(NAMES[NAMES.id == "translate"][SYSTEM_LANG].to_list())
_annotation = "".join(NAMES[NAMES.id == "annotation"][SYSTEM_LANG].to_list())
 
with gr.Blocks() as demo:
    gr.Markdown(_title)
    with gr.Row():
      
      with gr.Column():
        with gr.Row():            
            choice_type = gr.Dropdown(
              choices = TYPE[SYSTEM_LANG].to_list(), label=_type, value = TYPE[SYSTEM_LANG][0])
              
            choice_input = gr.Dropdown(
              choices = LANGUAGE.language.to_list(), label=_from, value = "Русский язык")
                           
            
              
      with gr.Column():
        with gr.Row():            
            choice_output = gr.Dropdown(
              choices = LANGUAGE.language.to_list(), label=_to, value = "Къарачай-Малкъар тил")
                           
            dialect = gr.Dropdown(
              choices = DIALECT.dialect.to_list(), label=_dialect, value = "дж\ч")
        
    with gr.Row():
      with gr.Column():
        text_input = gr.Textbox(lines=15, placeholder=_your_sent, label = "", show_copy_button=True)
        
      with gr.Column():
        text_output = gr.Textbox(lines=15, placeholder=_transl_sent, label = "", autoscroll=False, show_copy_button=True)
        
    text_button = gr.Button(_translate, variant = 'primary')
    
    text_button.click(out, inputs=[text_input, choice_input, choice_output, dialect, choice_type], outputs=[text_output]) # text, from, to, dialect
    
    gr.Markdown(_annotation)

# 10. Launch
# demo.launch(inbrowser=True)
demo.launch()



