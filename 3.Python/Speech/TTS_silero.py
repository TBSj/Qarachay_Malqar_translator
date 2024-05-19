!pip install omegaconf

import torch
import random
import string

language = 'cyrillic'
model_id = 'v4_cyrillic'
sample_rate = 48000
speaker = 'b_krc'
device = torch.device('cpu')
FILE_PATH = "1.Data/Speech"


model, example_text = torch.hub.load(repo_or_dir='snakers4/silero-models',
                                     model='silero_tts',
                                     language=language,
                                     speaker=model_id)
model.to(device)  # gpu or cpu

random_string = ''.join(random.choices(string.ascii_letters, k=8))

text = "Къалайса, Шохум? Не хапарынг барды?"

model.save_wav(audio_path=f'{FILE_PATH}/{random_string}.wav',
            text=text,
            speaker=speaker,
            sample_rate=sample_rate
        )
