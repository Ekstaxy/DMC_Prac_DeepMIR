"""Debug script to identify when model outputs extreme values during training"""

import os
import sys
import torch
from torch.optim import Adam

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.datasets import ENSTDrumsDataset
from data.audio_effects import AudioEffect
from models.model_DMC import TransformationNetwork
from torch.utils.data import DataLoader

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
    num_examples_per_epoch=20
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
print(f'Dataset: {len(dataset)} samples, Batch size: 4\n')

# Create model
model = TransformationNetwork(num_blocks=10, channels=128).to(device)
optimizer = Adam(model.parameters(), lr=3e-4)
l1_loss = torch.nn.L1Loss()

print("Starting small training test (5 epochs, 5 batches per epoch)\n")
print("="*80)

for epoch in range(5):
    print(f"\nEPOCH {epoch+1}/5")
    print("-"*80)

    model.train()
    epoch_loss = 0

    for batch_idx, (x, y, params) in enumerate(dataloader):
        if batch_idx >= 5:  # Only process 5 batches per epoch
            break

        x = x.to(device)
        y = y.to(device)
        params = params.to(device)

        print(f"\nBatch {batch_idx+1}/5:")
        print(f"  x: {x.shape}, range=[{x.min():.3f}, {x.max():.3f}], mean={x.mean():.3f}")
        print(f"  y: {y.shape}, range=[{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}")
        print(f"  params: {params.shape}, range=[{params.min():.3f}, {params.max():.3f}]")

        # Check input validity
        if torch.isnan(x).any() or torch.isinf(x).any():
            print("  ⚠️ WARNING: Input x has NaN/Inf!")
        if torch.isnan(y).any() or torch.isinf(y).any():
            print("  ⚠️ WARNING: Target y has NaN/Inf!")

        optimizer.zero_grad()
        y_pred, _ = model(x, params)

        print(f"  y_pred: {y_pred.shape}, range=[{y_pred.min():.3f}, {y_pred.max():.3f}], mean={y_pred.mean():.3f}")

        # Check prediction validity
        if torch.isnan(y_pred).any():
            print("  ❌ ERROR: y_pred has NaN!")
            print("  Stopping training - model is outputting NaN")
            exit()
        if torch.isinf(y_pred).any():
            print("  ❌ ERROR: y_pred has Inf!")
            print("  Stopping training - model is outputting Inf")
            exit()
        if y_pred.abs().max() > 1000:
            print(f"  ⚠️ WARNING: y_pred has extreme values (max={y_pred.abs().max():.1f})")

        # Center crop
        diff = y.shape[-1] - y_pred.shape[-1]
        if diff > 0:
            start = diff // 2
            y_crop = y[..., start:start + y_pred.shape[-1]]
        else:
            y_crop = y

        loss = l1_loss(y_pred, y_crop)

        print(f"  Loss: {loss.item():.6f}", end="")

        if torch.isnan(loss):
            print(" ❌ NaN!")
            print("  Stopping training - loss is NaN")
            exit()
        elif torch.isinf(loss):
            print(" ❌ Inf!")
            print("  Stopping training - loss is Inf")
            exit()
        elif loss.item() > 100:
            print(f" ⚠️ Very high loss!")
        else:
            print(" ✓")

        loss.backward()

        # Check gradients
        max_grad = 0
        for param in model.parameters():
            if param.grad is not None:
                max_grad = max(max_grad, param.grad.abs().max().item())
        print(f"  Max gradient: {max_grad:.3f}", end="")
        if max_grad > 100:
            print(" ⚠️ Large gradients!")
        else:
            print()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        epoch_loss += loss.item()

    avg_loss = epoch_loss / 5
    print(f"\nEpoch {epoch+1} avg loss: {avg_loss:.6f}")
    print("="*80)

print("\n✅ Training test completed successfully - no NaN/Inf detected!")
