#%% md
# # This is a sample Jupyter Notebook
# 
# Below is an example of a code cell. 
# Put your cursor into the cell and press Shift+Enter to execute it and select the next one, or click 'Run Cell' button.
# 
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
# 
# To learn more about Jupyter Notebooks in PyCharm, see [help](https://www.jetbrains.com/help/pycharm/ipython-notebook-support.html).
# For an overview of PyCharm, go to Help -> Learn IDE features or refer to [our documentation](https://www.jetbrains.com/help/pycharm/getting-started.html).
#%%
import whisper
import torch
from datasets import load_dataset
import numpy as np
import os
from pydub import AudioSegment
import soundfile as sf

# Check if GPU is available
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


#%%
dataset = load_dataset("Nexdata/English_Emotional_Speech_Data_by_Microphone", split="train", streaming=True)
#%%
"""Function to convert audio to mp3"""
# for i, audio in enumerate(dataset):
#     # Step 1: Extract audio data
#     audio_array = audio['audio']['array']
#     sampling_rate = audio['audio']['sampling_rate']
#
#     # Step 2: Extract audio data from the dataset
#     audio_array = np.array(audio_array)
#
#     # Step 3: Save as WAV first
#     sf.write(f"audio/audio_{i}.wav", audio_array, sampling_rate)
#
#     # Step 4: Convert and save the file as mp3
#     audio = AudioSegment.from_wav(f"audio/audio_{i}.wav")
#     audio.export(os.path.join(path, f"audio_{i}.mp3"), format="mp3")
#     print('File saved!')
#
#     if i == 4:
#         break
#%%
model = whisper.load_model("tiny", device=device)
print(model)
#%%
path_for_texts = r"D:\Users\mlaudan\PycharmProjects\WhisperTask\texts_from_mp3"
path_for_audio = r"D:\Users\mlaudan\PycharmProjects\WhisperTask\audio_mp3"

def transcribe(audio_path):
    result = model.transcribe(audio_path)
    text = result["text"]
    return text


def save_text(text, path, name):
    output_path = os.path.join(path_for_texts, path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    file = os.path.join(output_path, f"{name}.txt")

    with open(file, "w", encoding="utf-8") as f:
        f.write(text)

    print(f'Text saved!: "{text}"')

    return True


#%%
for root, dirs, files in os.walk(path_for_audio):
    # print(f"Path: {root}, Files: {files}")
    for i in files:
        text = transcribe(os.path.join(root, i))
        save_text(text, os.path.basename(root), i)

