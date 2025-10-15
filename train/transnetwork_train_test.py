"""Test pipeline matching real training with full debugging"""

import os
import sys
import torch
from torch.optim import Adam
import auraloss

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.datasets import ENSTDrumsDataset
from data.audio_effects import AudioEffect
from models.model_DMC import TransformationNetwork
from torch.utils.data import DataLoader

def center_crop(y_true, y_pred):
    diff = y_true.shape[-1] - y_pred.shape[-1]
    if diff > 0:
        start = diff // 2
        return y_true[..., start:start + y_pred.shape[-1]]
    return y_true

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Device: {device}\n')

# Create dataset
audio_effect = AudioEffect(block_size=512, sample_rate=44100)
dataset = ENSTDrumsDataset(
    root_dir="data/ENST-drums-audio/ENST-drums-public",  # UPDATE THIS
    length=65536,
    sample_rate=44100,
    train_target="transformer",
    apply_effects=True,
    audio_effect=audio_effect,
    remove_silence=True,
    num_examples_per_epoch=20
)

dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
print(f'Dataset: {len(dataset)} samples, Batch size: 4')
print(f'Batches per epoch: {len(dataloader)}\n')

# Create model
model = TransformationNetwork(num_blocks=10, channels=128).to(device)
optimizer = Adam(model.parameters(), lr=3e-4)
l1_loss = auraloss.freq.SumAndDifferenceSTFTLoss(
    fft_sizes = [512, 1024, 2048],  # Multiple scales for multi-resolution
    hop_sizes = [128, 256, 512],    # 25% overlap (hop = fft_size/4)
    win_lengths = [512, 1024, 2048], # Same as fft_sizes for full window
    window = "hann_window",
    w_sum = 1.0,
    w_diff = 1.0,
    output = "loss",
    sample_rate = 44100,
    perceptual_weighting=True,
    scale = 'mel',
    n_bins = 64
)

accumulation_steps = 4
print(f"Gradient accumulation: {accumulation_steps} steps")
print(f"Effective batch size: {4 * accumulation_steps}")
print("\nStarting training test (5 epochs)\n")
print("="*80)

for epoch in range(5):
    print(f"\nEPOCH {epoch+1}/5")
    print("-"*80)

    model.train()
    total_loss = 0
    optimizer.zero_grad()

    for batch_idx, (x, y, params) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        params = params.to(device)

        print(f"\nBatch {batch_idx+1}/{len(dataloader)}:")

        # Check input
        print(f"  x: {x.shape}, range=[{x.min():.3f}, {x.max():.3f}], mean={x.mean():.3f}, std={x.std():.3f}")
        if torch.isnan(x).any():
            print("  ❌ INPUT x has NaN!")
        if torch.isinf(x).any():
            print("  ❌ INPUT x has Inf!")
        if x.abs().max() > 10:
            print(f"  ⚠️ INPUT x extreme value: {x.abs().max():.3f}")

        # Check target
        print(f"  y: {y.shape}, range=[{y.min():.3f}, {y.max():.3f}], mean={y.mean():.3f}, std={y.std():.3f}")
        if torch.isnan(y).any():
            print("  ❌ TARGET y has NaN!")
        if torch.isinf(y).any():
            print("  ❌ TARGET y has Inf!")
        if y.abs().max() > 10:
            print(f"  ⚠️ TARGET y extreme value: {y.abs().max():.3f}")

        # Check params
        print(f"  params: {params.shape}, range=[{params.min():.3f}, {params.max():.3f}]")
        if torch.isnan(params).any():
            print("  ❌ PARAMS has NaN!")

        # Forward pass
        y_pred, _ = model(x, params)

        # Check prediction
        print(f"  y_pred: {y_pred.shape}, range=[{y_pred.min():.3f}, {y_pred.max():.3f}], mean={y_pred.mean():.3f}, std={y_pred.std():.3f}")
        if torch.isnan(y_pred).any():
            print("  ❌ PREDICTION has NaN! Stopping...")
            exit()
        if torch.isinf(y_pred).any():
            print("  ❌ PREDICTION has Inf! Stopping...")
            exit()
        if y_pred.abs().max() > 1000:
            print(f"  ⚠️ PREDICTION extreme value: {y_pred.abs().max():.3f}")

        # Center crop
        y_crop = center_crop(y, y_pred)
        print(f"  y_crop: {y_crop.shape}")

        # Compute loss
        loss = l1_loss(y_pred, y_crop)
        print(f"  Loss (before scaling): {loss.item():.6f}", end="")

        if torch.isnan(loss):
            print(" ❌ Loss is NaN! Stopping...")
            exit()
        if torch.isinf(loss):
            print(" ❌ Loss is Inf! Stopping...")
            exit()
        if loss.item() > 100:
            print(f" ⚠️ Very high loss!")
        else:
            print()

        # Scale loss (gradient accumulation)
        loss = loss / accumulation_steps
        print(f"  Loss (after scaling): {loss.item():.6f}")
        loss.backward()

        # Check gradients
        max_grad = 0
        nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"  ❌ NaN gradient in {name}")
                    nan_grad = True
                max_grad = max(max_grad, param.grad.abs().max().item())

        if nan_grad:
            print("  Stopping due to NaN gradients...")
            exit()

        print(f"  Max gradient: {max_grad:.3f}", end="")
        if max_grad > 100:
            print(" ⚠️ Large!")
        else:
            print()

        # Update weights every accumulation_steps
        if (batch_idx + 1) % accumulation_steps == 0:
            print(f"  → Updating weights (accumulated {accumulation_steps} batches)")
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps

    # Final step if needed
    if (batch_idx + 1) % accumulation_steps != 0:
        print(f"  → Final update (accumulated {(batch_idx + 1) % accumulation_steps} batches)")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataloader)
    print(f"\nEpoch {epoch+1} avg loss: {avg_loss:.6f}")
    print("="*80)

print("\n✅ Training test completed - no crashes!")
