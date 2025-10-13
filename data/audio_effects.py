"""
Audio effects for data augmentation and transformation learning.

This module provides classes to apply various audio effects to stems
and return both the processed audio and the effect parameters.
"""
import sys
import os
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import random
sys.path.insert(0, '/kaggle/working/pymixconsole/pymixconsole')  # Adjust the path as necessary
import pymixconsole as pymc
import pymixconsole as pymc
from torch import eq


class AudioEffect:
    def __init__(self, block_size: int = 512, sample_rate: int = 44100):
        self.channel = pymc.channel.Channel(block_size=block_size, sample_rate=sample_rate)
        reverb = pymc.processors.ConvolutionalReverb(block_size=512, sample_rate=44100)
        self.channel.processors.add(reverb)
        self.processors = self.channel.get_all_processors()

        self.pre_gain = self.processors[0]
        self.polarity_invert = self.processors[1]
        self.eq = self.channel.processors.get("eq")
        self.compressor = self.channel.processors.get("compressor")
        self.reverb = self.channel.processors.get("reverb")
        self.post_gain = self.processors[5]
        self.panner = self.processors[6]

    def apply(self, audio: np.ndarray, sample_rate: int, parameters: List, randomnize: bool = True) -> Tuple[np.ndarray, List]:
        """
        Apply the audio effect to the input audio.

        Args:
            audio (np.ndarray): Input audio array of shape (num_samples, num_channels).
            sample_rate (int): Sample rate of the input audio.
            parameters (Dict[str, float]): Dictionary of effect parameters.

        Returns:
            Tuple[np.ndarray, Dict[str, float]]: Processed audio and applied parameters.
        """
        if randomnize:
            self.channel.randomize(shuffle=False)
            parameters = [
                self.pre_gain.parameters.gain.value,
                self.polarity_invert.parameters.invert.value,
                self.eq.parameters.low_shelf_gain.value,
                self.eq.parameters.low_shelf_freq.value,
                self.eq.parameters.first_band_gain.value,
                self.eq.parameters.first_band_freq.value,
                self.eq.parameters.first_band_q.value,
                self.eq.parameters.second_band_gain.value,
                self.eq.parameters.second_band_freq.value,
                self.eq.parameters.second_band_q.value,
                self.eq.parameters.third_band_gain.value,
                self.eq.parameters.third_band_freq.value,
                self.eq.parameters.third_band_q.value,
                self.eq.parameters.high_shelf_gain.value,
                self.eq.parameters.high_shelf_freq.value,
                self.compressor.parameters.threshold.value,
                self.compressor.parameters.attack_time.value,
                self.compressor.parameters.release_time.value,
                self.compressor.parameters.ratio.value,
                self.compressor.parameters.makeup_gain.value,
                self.reverb.parameters.type.value,
                self.reverb.parameters.decay.value,
                self.reverb.parameters.dry_mix.value,
                self.reverb.parameters.wet_mix.value,
                self.post_gain.parameters.gain.value,
                self.panner.parameters.pan.value
            ]
        else:
            # 'sm-room', 'md-room', 'lg-room', 'hall', 'plate'
            if parameters[20] == 0:
                parameters[20] = 'sm-room'
            elif parameters[20] == 1:
                parameters[20] = 'md-room'
            elif parameters[20] == 2:
                parameters[20] = 'lg-room'
            elif parameters[20] == 3:
                parameters[20] = 'hall'
            elif parameters[20] == 4:
                parameters[20] = 'plate'

            self.pre_gain.parameters.gain.value = parameters[0]

            self.polarity_invert.parameters.invert.value = parameters[1]

            self.eq.parameters.low_shelf_gain.value = parameters[2]
            self.eq.parameters.low_shelf_freq.value = parameters[3]
            self.eq.parameters.first_band_gain.value = parameters[4]
            self.eq.parameters.first_band_freq.value = parameters[5]
            self.eq.parameters.first_band_q.value = parameters[6]
            self.eq.parameters.second_band_gain.value = parameters[7]
            self.eq.parameters.second_band_freq.value = parameters[8]
            self.eq.parameters.second_band_q.value = parameters[9]
            self.eq.parameters.third_band_gain.value = parameters[10]
            self.eq.parameters.third_band_freq.value = parameters[11]
            self.eq.parameters.third_band_q.value = parameters[12]
            self.eq.parameters.high_shelf_gain.value = parameters[13]
            self.eq.parameters.high_shelf_freq.value = parameters[14]

            self.compressor.parameters.threshold.value = parameters[15]
            self.compressor.parameters.attack_time.value = parameters[16]
            self.compressor.parameters.release_time.value = parameters[17]
            self.compressor.parameters.ratio.value = parameters[18]
            self.compressor.parameters.makeup_gain.value = parameters[19]

            self.reverb.parameters.type.value = parameters[20]
            self.reverb.parameters.decay.value = parameters[21]
            self.reverb.parameters.dry_mix.value = parameters[22]
            self.reverb.parameters.wet_mix.value = parameters[23]
            
            self.post_gain.parameters.gain.value = parameters[24]
            self.panner.parameters.pan.value = parameters[25]

        # 'sm-room', 'md-room', 'lg-room', 'hall', 'plate'
        if parameters[20] == 'sm-room':
            parameters[20] = 0
        elif parameters[20] == 'md-room':
            parameters[20] = 1
        elif parameters[20] == 'lg-room':
            parameters[20] = 2
        elif parameters[20] == 'hall':
            parameters[20] = 3
        elif parameters[20] == 'plate':
            parameters[20] = 4

        if audio.shape[1] == 1:
            # Mono to stereo
            audio = np.repeat(audio, 2, axis=1)
        processed_audio = np.empty(shape=(audio.shape[0],2))

        for n in range(audio.shape[0]//512):
            start = n * 512
            stop = start + 512

            processed_audio[start:stop] = self.channel.process(audio[start:stop])

        return processed_audio, parameters


        