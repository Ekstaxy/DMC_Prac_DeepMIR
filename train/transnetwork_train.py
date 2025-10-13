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

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0

    for x, y, params in dataloader:
        x = x.to(device)  # [batch, num_tracks, channel=2, length]
        y = y.to(device)  # [batch, channel=2, length]
        params = params.to(device)  # [batch, 1, 26]

        optimizer.zero_grad()
        y_pred, _ = model(x, params)

        # Center crop ground truth to match prediction
        y_crop = center_crop(y, y_pred)

        # L1 loss (MAE)
        loss = nn.L1Loss()(y_pred, y_crop)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def val_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for x, y, params in dataloader:
            x = x.to(device)
            y = y.to(device)
            params = params.to(device)

            y_pred, _ = model(x, params)
            y_crop = center_crop(y, y_pred)

            loss = nn.L1Loss()(y_pred, y_crop)
            total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--tcn_blocks', type=int, default=10, choices=[10, 20, 30])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--sample_rate', type=int, default=44100)
    parser.add_argument('--length', type=int, default=65536)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--results_dir', type=str, default='results')
    args = parser.parse_args()

    # Ensure the script utilizes GPU if available, else defaults to CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        num_examples_per_epoch=1000,
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
        num_examples_per_epoch=200,
        drummers=[1, 2, 3],
        indices=[168, 189]
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f'Train: {len(train_dataset)} | Val: {len(val_dataset)}')

    # Create model
    model = TransformationNetwork(
        num_blocks=args.tcn_blocks,
        channels=128,
        kernel_size=15,
        num_params=26,
        cglobal_dim=128
    ).to(device)
    print(f'TCN-{args.tcn_blocks} created')

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=args.patience, verbose=True)

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    print('Training started')
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
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
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('L1 Loss')
    plt.title(f'TCN-{args.tcn_blocks} Training')
    plt.legend()
    plt.grid(True)
    plt.savefig(results_dir / 'loss_curve.png', dpi=150)
    print(f'Loss curve saved')

    print('Training complete')

if __name__ == '__main__':
    main()
