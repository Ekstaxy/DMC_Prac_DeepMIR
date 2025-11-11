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
from models.model_DMC import TransformationNetwork, DifferentiableMixingConsole
if len(sys.argv) > 1:
    HF_TOKEN = sys.argv[1]
else:
    HF_TOKEN = ""  # ← Or paste your token here

if not HF_TOKEN:
    print("❌ Please provide token: python script.py YOUR_TOKEN")
    sys.exit(1)

# Login
from huggingface_hub import login
login(token=HF_TOKEN)
from diffusers import AutoencoderOobleck

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
