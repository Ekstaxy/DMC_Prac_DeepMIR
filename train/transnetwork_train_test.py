"""Debug script to investigate NaN loss issues"""

import os
import sys
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.datasets import ENSTDrumsDataset
from data.audio_effects import AudioEffect
from models.model_DMC import TransformationNetwork

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

# Create dataset
audio_effect = AudioEffect(block_size=512, sample_rate=44100)
dataset = ENSTDrumsDataset(
    root_dir="data/ENST-drums-audio/ENST-drums-public",  # UPDATE THIS PATH
    length=65536,
    sample_rate=44100,
    train_target="transformer",
    apply_effects=True,
    audio_effect=audio_effect,
    remove_silence=True,
    num_examples_per_epoch=10
)

# Create model
model = TransformationNetwork(num_blocks=10, channels=128).to(device)
l1_loss = torch.nn.L1Loss()

print("Testing 10 samples...\n")
for i in range(10):
    print(f"{'='*60}")
    print(f"Sample {i+1}")
    print(f"{'='*60}")

    # Get single sample
    x, y, params = dataset[i]

    print(f"x shape: {x.shape}")
    print(f"x range: [{x.min():.6f}, {x.max():.6f}]")
    print(f"x mean: {x.mean():.6f}, std: {x.std():.6f}")
    print(f"x has NaN: {torch.isnan(x).any()}, has Inf: {torch.isinf(x).any()}")

    print(f"\ny shape: {y.shape}")
    print(f"y range: [{y.min():.6f}, {y.max():.6f}]")
    print(f"y mean: {y.mean():.6f}, std: {y.std():.6f}")
    print(f"y has NaN: {torch.isnan(y).any()}, has Inf: {torch.isinf(y).any()}")

    print(f"\nparams shape: {params.shape}")
    print(f"params range: [{params.min():.6f}, {params.max():.6f}]")
    print(f"params has NaN: {torch.isnan(params).any()}")

    # Add batch dimension and move to device (dataset returns single sample)
    x_batch = x.unsqueeze(0).to(device)  # Add batch dim: [1, ...]
    y_batch = y.unsqueeze(0).to(device)  # Add batch dim: [1, ...]
    params_batch = params.unsqueeze(0).to(device)  # Add batch dim: [1, ...]

    print(f"\nBatch x shape: {x_batch.shape}")
    print(f"Batch y shape: {y_batch.shape}")
    print(f"Batch params shape: {params_batch.shape}")

    # Forward pass
    with torch.no_grad():
        y_pred, _ = model(x_batch, params_batch)

    print(f"\ny_pred shape: {y_pred.shape}")
    print(f"y_pred range: [{y_pred.min():.6f}, {y_pred.max():.6f}]")
    print(f"y_pred mean: {y_pred.mean():.6f}, std: {y_pred.std():.6f}")
    print(f"y_pred has NaN: {torch.isnan(y_pred).any()}, has Inf: {torch.isinf(y_pred).any()}")

    # Crop if needed
    if y_pred.shape[-1] != y_batch.shape[-1]:
        diff = y_batch.shape[-1] - y_pred.shape[-1]
        if diff > 0:
            start = diff // 2
            y_crop = y_batch[..., start:start + y_pred.shape[-1]]
        else:
            y_crop = y_batch
    else:
        y_crop = y_batch

    print(f"\ny_crop shape: {y_crop.shape}")

    loss = l1_loss(y_pred, y_crop)

    print(f"\nLoss: {loss.item():.6f}")
    print(f"Loss is NaN: {torch.isnan(loss)}")
    print(f"Loss is Inf: {torch.isinf(loss)}")

    if torch.isnan(loss) or loss.item() > 1000:
        print("\n⚠️ PROBLEM DETECTED!")
        print("Saving problematic tensors...")
        torch.save({
            'x': x, 'y': y, 'params': params,
            'y_pred': y_pred.cpu(), 'loss': loss.item()
        }, f'debug_sample_{i}.pt')

    print()
