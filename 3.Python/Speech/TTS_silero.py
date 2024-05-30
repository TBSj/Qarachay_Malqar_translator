# !pip install omegaconf

import torch
import random
import string

LANGUAGE_KRC_TTS = 'cyrillic'
LANGUAGE_RUS_TTS = 'ru'

MODEL_ID_KRC_TTS = 'v4_cyrillic'
MODEL_ID_RUS_TTS = 'v4_ru'

SAMPLE_RATE = 48000
SPEAKER_KRC = 'b_krc'
SPEAKER_RUS = 'random'

REPO_TTS = "snakers4/silero-models"
MODEL_TTS = "silero_tts"

device = torch.device('cpu')

FILE_PATH = "1.Data/Speech"


model_krc, _  = torch.hub.load(repo_or_dir = REPO_TTS,
                                     model = MODEL_TTS,
                                     language = LANGUAGE_KRC_TTS,
                                     speaker = MODEL_ID_KRC_TTS)
                                     
model_rus, _  = torch.hub.load(repo_or_dir = REPO_TTS,
                                     model = MODEL_TTS,
                                     language = LANGUAGE_RUS_TTS,
                                     speaker = MODEL_ID_RUS_TTS)


model_krc.to(device)  # gpu or cpu
model_rus.to(device)  # gpu or cpu

# model_krc.speakers
# model_rus.speakers

random_string = ''.join(random.choices(string.ascii_letters, k=8))

text = "Къалайса, Шохум? Не хапарынг барды?"

model_krc.save_wav(audio_path=f'{FILE_PATH}/{random_string}.wav',
            text=text,
            speaker=SPEAKER_KRC,
            sample_rate=SAMPLE_RATE,
            put_accent=True
        )
        
text = "Как дела, друг мой? Какие у тебя новости?"
        
model_rus.save_wav(audio_path=f'{FILE_PATH}/{random_string}.wav',
            text=text,
            speaker=SPEAKER_RUS,
            sample_rate=SAMPLE_RATE,
            put_accent=True
        )
