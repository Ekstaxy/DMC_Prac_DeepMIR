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

def center_crop(y_true, y_pred):
    """Crop center of y_true to match y_pred size"""
    diff = y_true.shape[-1] - y_pred.shape[-1]
    if diff > 0:
        start = diff // 2
        return y_true[..., start:start + y_pred.shape[-1]]
    return y_true

def train_epoch(model, dataloader, optimizer, device, accumulation_steps=4, use_amp=False):
    model.train()
    total_loss = 0
    valid_batches = 0
    recent_losses = []  # Track recent losses for spike detection
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
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

    for batch_idx, (x, y, pad_mask) in enumerate(dataloader):
        x = x.to(device)
        y = y.to(device)
        pad_mask = pad_mask.to(device)

        if batch_idx == 0:
            print(f"Input x: {x.shape}, range: [{x.min():.3f}, {x.max():.3f}]")
            print(f"Target y: {y.shape}, range: [{y.min():.3f}, {y.max():.3f}]")
            print(f"Pad mask: {pad_mask.shape}")

        # Mixed precision forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                y_pred, params_pred = model(x)
        else:
            y_pred, params_pred = model(x)

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

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Update weights every accumulation_steps batches
        if (batch_idx + 1) % accumulation_steps == 0:
            # Gradient clipping
            if use_amp:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        valid_batches += 1

    # Final step if batches don't divide evenly
    if (batch_idx + 1) % accumulation_steps != 0:
        if use_amp:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
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
        for x, y, pad_mask in dataloader:
            x = x.to(device)
            y = y.to(device)
            pad_mask = pad_mask.to(device)

            y_pred, params_pred = model(x)

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
    parser = argparse.ArgumentParser(description="Train DifferentiableMixingConsole")
    parser.add_argument("--data_dir", type=str, default="HW2/data", help="Path to data directory")
    parser.add_argument("--results_dir", type=str, default="results/dmc_training", help="Directory to save results")
    parser.add_argument("--encoder", type=str, default="VGGish", choices=["VGGish", "StableAudio"],
                        help="Encoder type (default: VGGish)")
    parser.add_argument("--arch", type=str, default="Original", choices=["Original", "Simple"],
                        help="Transformation network architecture (default: Original)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to pretrained transformation network checkpoint")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace token (required for StableAudio encoder)")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--accumulation_steps", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training")
    parser.add_argument("--use_amp", action="store_true",
                        help="Use automatic mixed precision (FP16) to reduce memory")

    args = parser.parse_args()

    # Login to HuggingFace if using StableAudio encoder
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')
    if args.encoder == "StableAudio":
        if hf_token:
            print("Logging in to HuggingFace...")
            from huggingface_hub import login
            login(token=hf_token)
            print("✓ HuggingFace login successful\n")
        else:
            print("⚠ Warning: StableAudio encoder requires HuggingFace authentication.")
            print("Provide --hf_token or set HF_TOKEN environment variable\n")

    # Create results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(results_dir / 'config.json', 'w') as f:
        json.dump(vars(args), f, indent=4)

    print("=" * 70)
    print("Training DifferentiableMixingConsole")
    print("=" * 70)
    print(f"Configuration:")
    for key, value in vars(args).items():
        print(f"  {key}: {value}")
    print()

    # Initialize model
    print("Initializing model...")
    model = DifferentiableMixingConsole(
        sample_rate=44100,
        transformation_arch=args.arch,
        encoder_type=args.encoder
    )

    model = model.to(args.device)
    print(f"✓ Model initialized on {args.device}")

    # Load checkpoint if provided (after moving to device)
    if args.checkpoint:
        print(f"Loading checkpoint from: {args.checkpoint}")
        model.load_transformation_checkpoint(args.checkpoint, device=args.device)

    # Initialize datasets
    print("\nInitializing datasets...")
    try:
        # Assuming 80/20 train/val split
        # First, get total number of available examples
        import glob
        all_mixes = []
        for drummer in [1, 2, 3]:  # Default drummers
            search_path = os.path.join(args.data_dir, f"drummer_{drummer}", "audio", "dry_mix", "*.wav")
            all_mixes.extend(glob.glob(search_path))

        total_mixes = len([f for f in all_mixes if "norm" not in f and "hits" not in f])
        train_split = int(0.8 * total_mixes)

        print(f"Total mixes found: {total_mixes}")
        print(f"Train split: 0-{train_split}, Val split: {train_split}-{total_mixes}")

        train_dataset = ENSTDrumsDataset(
            root_dir=args.data_dir,
            length=44100,  # ~1 second at 44100 Hz
            sample_rate=44100,
            drummers=[1, 2, 3],
            indices=[0, train_split],
            num_examples_per_epoch=1000,  # Adjust as needed
        )
        val_dataset = ENSTDrumsDataset(
            root_dir=args.data_dir,
            length=44100,
            sample_rate=44100,
            drummers=[1, 2, 3],
            indices=[train_split, total_mixes],
            num_examples_per_epoch=200,  # Adjust as needed
        )
        print(f"✓ Train dataset: {len(train_dataset)} samples per epoch")
        print(f"✓ Val dataset: {len(val_dataset)} samples per epoch")
    except Exception as e:
        print(f"✗ Dataset initialization failed: {e}")
        import traceback
        traceback.print_exc()
        print("Please check your data directory and dataset implementation")
        return

    # Initialize data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False
    )

    # Initialize optimizer and scheduler
    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(args.epochs):
        print(f"\n{'='*70}")
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"{'='*70}")

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, args.device, args.accumulation_steps, args.use_amp)
        train_losses.append(train_loss)
        print(f"Train Loss: {train_loss:.6f}")

        # Validate
        val_loss = val_epoch(model, val_loader, args.device)
        val_losses.append(val_loss)
        print(f"Val Loss: {val_loss:.6f}")

        # Update scheduler
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': vars(args)
            }
            torch.save(checkpoint, results_dir / 'best_model.pth')
            print(f"✓ Best model saved (val_loss: {val_loss:.6f})")

        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'config': vars(args)
        }
        torch.save(checkpoint, results_dir / 'latest_model.pth')

        # Plot losses
        plt.figure(figsize=(10, 5))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Progress')
        plt.legend()
        plt.grid(True)
        plt.savefig(results_dir / 'training_loss.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Save final model
    torch.save(model.state_dict(), results_dir / 'final_model.pth')
    print(f"\n{'='*70}")
    print(f"✓ Training complete!")
    print(f"  Best val loss: {best_val_loss:.6f}")
    print(f"  Results saved to: {results_dir}")
    print(f"{'='*70}")

if __name__ == "__main__":
    main()
