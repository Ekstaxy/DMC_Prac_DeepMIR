from email.mime import audio
import os
import sys
import pymixconsole as pymc
import numpy as np
import soundfile as sf
import librosa
from data.audio_effects import AudioEffect
import torchaudio

# y1, sr1 = torchaudio.load("data/ENST-drums-audio/ENST-drums-public/drummer_1/audio/wet_mix/021_hits_snare-drum_mallets_x5.wav", num_frames=44100*10)
# y2, sr2 = librosa.load("data/ENST-drums-audio/ENST-drums-public/drummer_1/audio/wet_mix/021_hits_snare-drum_mallets_x5.wav", sr=44100, mono=False, duration=10.0)
# y3, sr3 = sf.read("data/ENST-drums-audio/ENST-drums-public/drummer_1/audio/wet_mix/021_hits_snare-drum_mallets_x5.wav", dtype='float32', frames=44100*10)

# channel = pymc.channel.Channel(block_size=512, sample_rate=44100)
# reverb = pymc.processors.ConvolutionalReverb(block_size=512, sample_rate=44100)
# channel.processors.add(reverb)
# processors = channel.get_all_processors()

# processed_audio = np.empty(shape=(y3.shape[0],2))
# print("process_audio_shape: ", y3.shape)
# print("block_size: ", y3[0:512, :].shape)

# for n in range(y3.shape[0]//512):
#     start = n * 512
#     stop = start + 512

#     processed_audio[start:stop] = channel.process(y3[start:stop])

# print("processed_audio shape: ", processed_audio.shape)

channel = pymc.channel.Channel(block_size=512, sample_rate=44100)
reverb = pymc.processors.ConvolutionalReverb(block_size=512, sample_rate=44100)
channel.processors.add(reverb)
processors = channel.get_all_processors()

for p in processors:
    print(p.name)
    print(p.parameters)