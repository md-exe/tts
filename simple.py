import os
import torch

device = torch.device('cpu')
torch.set_num_threads(16)
local_file = 'model.pt'

if not os.path.isfile(local_file):
    torch.hub.download_url_to_file('https://models.silero.ai/models/tts/ru/v3_1_ru.pt',
                                   local_file)  

model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
model.to(device)

ssml_sample = f"""
                      <speak>
                      <p>
                      Всем привет!
                      </p>
                      </speak>
                      """
sample_rate = 48000
speaker='baya'

audio_paths = model.save_wav(ssml_text=ssml_sample,
                             speaker=speaker,
                             sample_rate=sample_rate)