"""Simple test for ENSTDrumsDataset"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.datasets import ENSTDrumsDataset
from data.audio_effects import AudioEffect
from torch.utils.data import DataLoader

# Update this path to your dataset
root_dir = "data/ENST-drums-audio/ENST-drums-public"

# Initialize dataset
audio_effect = AudioEffect(block_size=512, sample_rate=44100)
dataset = ENSTDrumsDataset(
    root_dir=root_dir,
    length=3*44100,
    sample_rate=44100,
    indices=[0, 10],  # Use first 10 samples for quick testing
    train_target="transformer",
    apply_effects=True,
    audio_effect=audio_effect,
    remove_silence=True,
    num_examples_per_epoch=10
)

# Create dataloader
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Get one batch
x, y, params = next(iter(dataloader))

print(f"Batch loaded successfully!")
print(f"x (origin) shape: {x.shape}")
print(f"y (processed) shape: {y.shape}")
print(f"params length: {len(params)}")
# print(f"params: {params}")
