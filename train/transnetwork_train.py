"""
Training script for TCN-based Transformation Network
"""

import os
import sys
import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import auraloss

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.datasets import ENSTDrumsDataset
from data.audio_effects import AudioEffect
from models.model_DMC import TransformationNetwork

def center_crop(y_true, y_pred):
    """Crop center of y_true to match y_pred size"""
    diff = y_true.shape[-1] - y_pred.shape[-1]
    if diff > 0:
        start = diff // 2
        return y_true[..., start:start + y_pred.shape[-1]]
    return y_true

def train_epoch(model, dataloader, optimizer, device, accumulation_steps=4):
    model.train()
    total_loss = 0
    valid_batches = 0
    recent_losses = []  # Track recent losses for spike detection
    sum_and_diff_STFT_loss = auraloss.freq.SumAndDifferenceSTFTLoss(
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

    optimizer.zero_grad()

    for batch_idx, (x, y, params) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        params = params.to(device)

        if batch_idx == 0:
            print(f"Input x: {x.shape}, range: [{x.min():.3f}, {x.max():.3f}]")
            print(f"Target y: {y.shape}, range: [{y.min():.3f}, {y.max():.3f}]")
            print(f"Params: {params.shape}")

        y_pred, _ = model(x, params)

        if batch_idx == 0:
            print(f"Predicted y_pred: {y_pred.shape}, range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")

        y_crop = center_crop(y, y_pred)

        if batch_idx == 0:
            print(f"Cropped y_crop: {y_crop.shape}")

        loss = sum_and_diff_STFT_loss(y_pred, y_crop)

        # Enhanced loss validation
        if torch.isnan(loss) or torch.isinf(loss) or loss.item() < 0:
            print(f"Invalid loss at batch {batch_idx}: {loss.item()}, skipping")
            continue

        # Spike detection using recent loss history
        if len(recent_losses) >=3:
            recent_avg = sum(recent_losses) / len(recent_losses)
            recent_std = (sum((x - recent_avg) ** 2 for x in recent_losses) / len(recent_losses)) ** 0.5
            
            # Skip if loss is more than 3 standard deviations above recent average
            # or more than 10x the recent average
            if (loss.item() > recent_avg + 3 * recent_std and loss.item() > recent_avg * 3) or loss.item() > 1e5:
                print(f"Loss spike detected at batch {batch_idx}: {loss.item():.6f} (avg: {recent_avg:.6f}, std: {recent_std:.6f}), skipping")
                continue

        # Update recent losses (keep last 10)
        recent_losses.append(loss.item())
        if len(recent_losses) > 10:
            recent_losses.pop(0)

        # Scale loss by accumulation steps
        loss = loss / accumulation_steps
        loss.backward()

        # Update weights every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        valid_batches += 1

    # Final step if batches don't divide evenly
    if (batch_idx + 1) % accumulation_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()

    return total_loss / max(valid_batches, 1)  # Avoid division by zero

def val_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0
    sum_and_diff_STFT_loss = auraloss.freq.SumAndDifferenceSTFTLoss(
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

    with torch.no_grad():
        for x, y, params in dataloader:
            x = x.to(device)
            y = y.to(device)
            params = params.to(device)

            y_pred, _ = model(x, params)

            # Check for Inf in prediction
            if torch.isinf(y_pred).any():
                print(f"Inf in y_pred at validation, skipping")
                continue

            y_crop = center_crop(y, y_pred)

            loss = sum_and_diff_STFT_loss(y_pred, y_crop)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"NaN/Inf loss at validation batch, skipping")
                print(f"loss: {loss.item() if not torch.isnan(loss) else 'NaN'}")
                print(f"y_pred_has_nan: {torch.isnan(y_pred).any()}")
                print(f"y_pred_has_inf: {torch.isinf(y_pred).any()}")
                print(f"y_has_nan: {torch.isnan(y).any()}")
                print(f"y_has_inf: {torch.isinf(y).any()}")
                continue

            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--tcn_blocks', type=int, default=10, choices=[10, 20, 30])
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sample_rate', type=int, default=44100)
    parser.add_argument('--length', type=int, default=44100*1.5)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--accumulation_steps', type=int, default=4)
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    # Ensure the script utilizes GPU if available, else defaults to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create model
    model = TransformationNetwork(
        num_blocks=args.tcn_blocks,
        channels=128,
        kernel_size=15,
        num_params=26,
        cglobal_dim=128
    )

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)
    print(f'Device: {device}')

    # Create results directory
    results_dir = Path(args.results_dir) / f'TCN-{args.tcn_blocks}'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f'Config saved')

    # Create datasets
    audio_effect = AudioEffect(block_size=512, sample_rate=args.sample_rate)

    train_dataset = ENSTDrumsDataset(
        root_dir=args.data_dir,
        length=args.length,
        sample_rate=args.sample_rate,
        train_target="transformer",
        apply_effects=True,
        audio_effect=audio_effect,
        remove_silence=True,
        num_examples_per_epoch=32,
        drummers=[1, 2, 3],
        indices=[0, 168]
    )

    val_dataset = ENSTDrumsDataset(
        root_dir=args.data_dir,
        length=args.length,
        sample_rate=args.sample_rate,
        train_target="transformer",
        apply_effects=True,
        audio_effect=audio_effect,
        remove_silence=True,
        num_examples_per_epoch=32,
        drummers=[1, 2, 3],
        indices=[168, 189]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)}')

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience)

    # Training loop
    best_val_loss = float(10e8)
    train_losses = []
    val_losses = []

    print('Training started')
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, accumulation_steps=args.accumulation_steps)
        val_loss = val_epoch(model, val_loader, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        print(f'Epoch {epoch+1}/{args.epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}')

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), results_dir / 'best_model.pth')
            print(f'Best model saved')

    # Save final results
    torch.save(model.state_dict(), results_dir / 'final_model.pth')

    # Save metrics
    metrics = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss
    }
    with open(results_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'Metrics saved')

    # Plot losses
    plt.figure(figsize=(10, 6))
    
    # Calculate reasonable y-axis limits by filtering out extreme outliers
    all_losses = train_losses + val_losses
    # Remove extreme outliers (values above 95th percentile)
    percentile_95 = np.percentile(all_losses, 95)
    filtered_losses = [loss for loss in all_losses if loss <= percentile_95]
    
    # Set y-axis limit to 110% of the 95th percentile of filtered data
    if filtered_losses:
        y_max = max(filtered_losses) * 1.1
    else:
        y_max = max(all_losses) * 1.1
    
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.title(f'TCN-{args.tcn_blocks} Training')
    plt.ylim(0, y_max)  # Set y-axis limit to suppress spikes
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / 'loss_curve.png', dpi=150)
    print(f'Loss curve saved')

    print('Training complete')

if __name__ == '__main__':
    main()
