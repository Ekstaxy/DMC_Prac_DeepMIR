"""
Dataset loading and saving utilities.
"""

import os
from cv2 import transform
import torch
from torch.utils.data import Dataset
from pathlib import Path
import glob
from typing import Dict, List, Optional, Union, Tuple
import numpy as np
import random
import torchaudio
import librosa
import soundfile as sf
from data.audio_effects import AudioEffect

class ENSTDrumsDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        length: int,
        sample_rate: float,
        drummers: List[int] = [1, 2],
        track_names: List[str] = [
            "kick",
            "snare",
            "hi-hat",
            "overhead_L",
            "overhead_R",
            "tom_1",
            "tom_2",
            "tom_3",
        ],
        indices: Tuple[int, int] = [0, 1],
        wet_mix: bool = False,
        hits: bool = False,
        num_examples_per_epoch: int = 1000,
        seed: int = 42,
        remove_silence: bool = False,
        silence_threshold_db: float = 20,
        apply_effects: bool = False,
        audio_effect: Optional[AudioEffect] = None,
        train_target: str = "mixing",  # "mixing" or "transformer"
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.length = length
        self.sample_rate = sample_rate
        self.drummers = drummers
        self.track_names = track_names
        self.indices = indices
        self.wet_mix = wet_mix
        self.hits = hits
        self.num_examples_per_epoch = num_examples_per_epoch
        self.max_num_tracks = 8  # this is fixed for this dataset
        self.remove_silence = remove_silence
        self.silence_threshold_db = silence_threshold_db
        self.apply_effects = apply_effects
        self.audio_effect = audio_effect
        self.train_target = train_target

        # find all mixes
        self.mix_filepaths = []
        for drummer in drummers:
            search_path = os.path.join(
                root_dir,
                f"drummer_{drummer}",
                "audio",
                "wet_mix" if wet_mix else "dry_mix",
                "*.wav",
            )
            self.mix_filepaths += glob.glob(search_path)

        # remove any mixes that are shorter than the requested length
        self.mix_filepaths = [
            fp
            for fp in self.mix_filepaths
            if torchaudio.info(fp).num_frames > self.length
        ]

        # remove any mixes that have "norm" in the filename
        self.mix_filepaths = [fp for fp in self.mix_filepaths if not "norm" in fp]

        # remove any mixes that are just hits
        if not self.hits:
            self.mix_filepaths = [fp for fp in self.mix_filepaths if "hits" not in fp]

        random.Random(seed).shuffle(self.mix_filepaths)
        self.mix_filepaths = self.mix_filepaths[indices[0] : indices[1]]

        if len(self.mix_filepaths) < 1:
            raise RuntimeError(f"No files found in {self.root_dir}.")
        else:
            print(f"Found {len(self.mix_filepaths)} examples from drummers: {drummers}")

        # for now we assume all songs have all tracks

    def __len__(self):
        return self.num_examples_per_epoch

    def _remove_silence(self, audio: torch.Tensor, sr: int, min_silence_duration: float = 3.0) -> torch.Tensor:
        """Remove silence from audio and concatenate non-silent portions.

        Args:
            audio: Input audio tensor
            sr: Sample rate
            min_silence_duration: Minimum silence duration in seconds to remove (default: 5.0)

        Returns:
            Audio with silent regions >min_silence_duration removed and concatenated
        """
        # Convert to numpy for librosa
        audio_np = audio.squeeze().numpy()

        # Split audio into non-silent intervals
        intervals = librosa.effects.split(
            audio_np,
            top_db=self.silence_threshold_db,
        )

        if len(intervals) == 0:
            # If no non-silent portions found, return original
            return audio

        # Filter out silence gaps that are longer than min_silence_duration
        min_silence_samples = int(min_silence_duration * sr)
        filtered_segments = []

        for i, (start, end) in enumerate(intervals):
            # Always include the segment
            filtered_segments.append(audio_np[start:end])

            # Check if there's a gap to the next segment
            if i < len(intervals) - 1:
                next_start = intervals[i + 1][0]
                silence_gap = next_start - end

                # If silence gap is shorter than threshold, keep it
                if silence_gap < min_silence_samples:
                    filtered_segments.append(audio_np[end:next_start])

        # Concatenate all segments
        if len(filtered_segments) > 0:
            non_silent = np.concatenate(filtered_segments)
            return non_silent.reshape(-1, 1)  # Return as 2D tensor
        else:
            return audio

    def __getitem__(self, _):
        if self.train_target == "transformer":
            return self._get_transformer_item()
        else:
            return self._get_mixing_item()

    def _get_mixing_item(self):
        """Original mixing mode - returns all stems."""
        # select a mix at random
        mix_idx = np.random.randint(0, len(self.mix_filepaths))
        mix_filepath = self.mix_filepaths[mix_idx]
        example_id = os.path.basename(mix_filepath)
        drummer_id = os.path.normpath(mix_filepath).split(os.path.sep)[-4]

        md = torchaudio.info(mix_filepath)  # check length

        # load the chunk of the mix
        silent = True
        while silent:
            # get random offset
            offset = np.random.randint(0, md.num_frames - self.length - 1)

            y, sr = torchaudio.load(
                mix_filepath,
                frame_offset=offset,
                num_frames=self.length,
            )
            energy = (y**2).mean()
            if energy > 1e-8:
                silent = False

        y /= y.abs().max().clamp(1e-8)  # peak normalize

        # -------------------- load the tracks from disk --------------------
        x = torch.zeros((self.max_num_tracks, self.length))
        pad = [True] * self.max_num_tracks  # note which tracks are empty

        for tidx, track_name in enumerate(self.track_names):
            track_filepath = os.path.join(
                self.root_dir,
                drummer_id,
                "audio",
                track_name,
                example_id,
            )
            if os.path.isfile(track_filepath):
                x_s, sr = torchaudio.load(
                    track_filepath,
                    frame_offset=offset,
                    num_frames=self.length,
                )
                x_s /= x_s.abs().max().clamp(1e-6)
                x_s *= 10 ** (-12 / 20.0)

                # Remove silence if enabled
                if self.remove_silence:
                    x_s = self._remove_silence(x_s, sr)
                    # Pad or trim to maintain consistent length
                    if x_s.shape[1] < self.length:
                        x_s = torch.nn.functional.pad(x_s, (0, self.length - x_s.shape[1]))
                    elif x_s.shape[1] > self.length:
                        x_s = x_s[:, :self.length]

                x[tidx, :] = x_s
                pad[tidx] = False

        # -------------------- apply effects if enabled --------------------
        if self.apply_effects and self.audio_effect is not None:
            # Apply effects to create targets
            y_effected = torch.zeros_like(x)
            effect_params = []

            for tidx in range(self.max_num_tracks):
                if not pad[tidx]:  # Only apply to non-empty tracks
                    # Extract single stem - convert to (num_samples, num_channels) format
                    stem = x[tidx].numpy()  # Shape: (length,)
                    # Reshape to (num_samples, 1) for mono audio
                    stem_reshaped = stem.reshape(-1, 1)

                    # Apply audio effect and get parameters
                    effected_stem, params = self.audio_effect.apply(
                        stem_reshaped,
                        int(sr),
                        parameters=[],
                        randomnize=True
                    )

                    # effected_stem shape: (num_samples, 2) - stereo output
                    # Convert back to mono by averaging channels and flatten
                    effected_mono = effected_stem.mean(axis=1)

                    # Store effected stem and parameters
                    y_effected[tidx] = torch.from_numpy(effected_mono)
                    effect_params.append(params)
                else:
                    effect_params.append(None)

            return x, y_effected, torch.tensor(pad), effect_params
        else:
            return x, y, torch.tensor(pad)

    def _get_transformer_item(self):
        """Transformer mode - returns a single random stem with params, origin, and processed waveform."""
        # select a mix at random
        mix_idx = np.random.randint(0, len(self.mix_filepaths))
        mix_filepath = self.mix_filepaths[mix_idx]
        example_id = os.path.basename(mix_filepath)
        drummer_id = os.path.normpath(mix_filepath).split(os.path.sep)[-4]

        # Get available stems for this example
        available_stems = []
        for track_name in self.track_names:
            track_filepath = os.path.join(
                self.root_dir,
                drummer_id,
                "audio",
                track_name,
                example_id,
            )
            if os.path.isfile(track_filepath):
                available_stems.append((track_name, track_filepath))

        if len(available_stems) == 0:
            raise RuntimeError(f"No stems found for {example_id}")

        # Randomly select one stem
        _, selected_stem_path = available_stems[np.random.randint(0, len(available_stems))]
        md = torchaudio.info(selected_stem_path) 
        offset = np.random.randint(0, md.num_frames - self.length - 1)

        silent = True
        while silent:
            # get random offset
            offset = np.random.randint(0, md.num_frames - self.length - 1)

            origin, sr = torchaudio.load(
                selected_stem_path,
                channels_first=False,
                frame_offset=offset,
                num_frames=self.length,
            )
            energy = (origin**2).mean()
            if energy > 1e-8:
                silent = False

        # Peak normalize
        origin = origin / origin.abs().max().clamp(1e-6)

        # Process the stem: remove silence >3 seconds and concatenate
        if self.remove_silence:
            origin_no_silence = self._remove_silence(origin, sr, min_silence_duration=5.0)  
        else:
            origin_no_silence = origin

        # Ensure processed audio is not empty
        if origin_no_silence.shape[1] == 0:
            # If all silent, use original
            origin_no_silence = origin

        if origin_no_silence.shape[1] != 2:
            # If mono, convert to stereo by duplicating channel
            origin_no_silence = origin_no_silence.repeat(2, 1)

        # Trim or pad to the desired length
        if origin_no_silence.shape[0] > self.length:
            x = origin_no_silence[:self.length]
        elif origin_no_silence.shape[0] < self.length:
            padding = self.length - origin_no_silence.shape[0]
            x = np.pad(origin_no_silence, ((0, padding), (0, 0)), mode='constant')
        else:
            x = origin_no_silence

        # Apply effects if enabled
        if self.apply_effects and self.audio_effect is not None:

            # Apply audio effect with randomized parameters
            y, params = self.audio_effect.apply(
                x,
                int(sr),
                parameters=[],
                randomnize=True
            )
        else:
            # No effects applied
            params = None
            y = origin_no_silence

        x = x.T[np.newaxis, :, :]
        y = y.transpose()  # Ensure (num_samples, num_channels)
        params = torch.tensor(params, dtype=torch.float32).unsqueeze(0)

        print("original_shape:", x.shape)
        print("processed_audio shape:", y.shape)

        return x, y, params