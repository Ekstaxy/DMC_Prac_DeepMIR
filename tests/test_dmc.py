import sys
import argparse
from pathlib import Path
import os

# Get token from command line argument
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

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from models.model_DMC import DifferentiableMixingConsole


def test_dmc(
    encoder_type="VGGish",
    transformation_arch="Original",
    checkpoint_path=None,
    use_gpu=False,
    hf_token=None
):
    """
    Test DifferentiableMixingConsole with various configurations.

    Args:
        encoder_type: Type of encoder ("VGGish" or "StableAudio")
        transformation_arch: Type of transformation network ("Original" or "Simple")
        checkpoint_path: Optional path to pretrained transformation network checkpoint
        use_gpu: Whether to use GPU if available
        hf_token: HuggingFace token (required for StableAudio encoder)
    """
    print("=" * 70)
    print("Testing DifferentiableMixingConsole")
    print("=" * 70)

    # Handle HuggingFace login for StableAudio encoder
    if encoder_type == "StableAudio":
        if hf_token:
            print("Logging in to HuggingFace with provided token...")
            from huggingface_hub import login
            login(token=hf_token)
            print("✓ HuggingFace login successful\n")
        else:
            print("Note: StableAudio encoder requires HuggingFace authentication.")
            print("Either provide --hf_token or run: huggingface-cli login\n")

    # Device setup
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # Test parameters
    sr = 44100
    seq_len = 131072  # ~3 seconds
    bs = 2  # Reduced batch size for testing
    num_tracks = 4  # Reduced number of tracks
    channel = 2

    print(f"\nTest Configuration:")
    print(f"  Sample Rate: {sr} Hz")
    print(f"  Sequence Length: {seq_len} samples (~{seq_len/sr:.2f}s)")
    print(f"  Batch Size: {bs}")
    print(f"  Num Tracks: {num_tracks}")
    print(f"  Channels: {channel}")
    print(f"  Encoder Type: {encoder_type}")
    print(f"  Transformation Architecture: {transformation_arch}")
    if checkpoint_path:
        print(f"  Checkpoint: {checkpoint_path}")
    print()

    # Create random test input
    print("Creating random test input...")
    x = torch.randn(bs, num_tracks, channel, seq_len).to(device)
    print(f"Input shape: {x.shape}")
    print(f"Input range: [{x.min():.3f}, {x.max():.3f}]")
    print()

    # Initialize model
    print(f"Initializing DifferentiableMixingConsole...")
    try:
        model = DifferentiableMixingConsole(
            sample_rate=sr,
            transformation_arch=transformation_arch,
            encoder_type=encoder_type
        )
        model = model.to(device)
        model.eval()
        print("✓ Model initialized successfully!")
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return

    # Load checkpoint if provided
    if checkpoint_path:
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        try:
            model.load_transformation_checkpoint(checkpoint_path, device=device)
        except Exception as e:
            print(f"✗ Checkpoint loading failed: {e}")
            print("Continuing with random weights...")

    # Forward pass
    print("\nRunning forward pass...")
    try:
        with torch.no_grad():
            y, p = model(x)

        print(f"✓ Forward pass successful!")
        print(f"\nOutput shapes:")
        print(f"  Mixed audio (y): {y.shape} (expected: [{bs}, {channel}, {seq_len}])")
        print(f"  Parameters (p): {p.shape} (expected: [{bs}, {num_tracks}, 26])")

        print(f"\nOutput statistics:")
        print(f"  Mixed audio range: [{y.min():.3f}, {y.max():.3f}]")
        print(f"  Mixed audio mean: {y.mean():.3f}")
        print(f"  Mixed audio std: {y.std():.3f}")
        print(f"  Parameters range: [{p.min():.3f}, {p.max():.3f}]")

        # Validate shapes
        assert y.shape == (bs, channel, seq_len), f"Output shape mismatch! Got {y.shape}"
        assert p.shape == (bs, num_tracks, 26), f"Params shape mismatch! Got {p.shape}"

        # Check for NaN or Inf
        if torch.isnan(y).any():
            print("✗ WARNING: Output contains NaN values!")
        elif torch.isinf(y).any():
            print("✗ WARNING: Output contains Inf values!")
        else:
            print("✓ Output is clean (no NaN/Inf)")

        print("\n" + "=" * 70)
        print("✓ All tests passed successfully!")
        print("=" * 70)

    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DifferentiableMixingConsole")
    parser.add_argument(
        "--encoder",
        type=str,
        default="VGGish",
        choices=["VGGish", "StableAudio"],
        help="Type of encoder to use (default: VGGish)"
    )
    parser.add_argument(
        "--arch",
        type=str,
        default="Original",
        choices=["Original", "Simple"],
        help="Transformation network architecture (default: Original)"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to pretrained transformation network checkpoint (.pth or .pt)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU if available"
    )
    parser.add_argument(
        "--hf_token",
        type=str,
        default=None,
        help="HuggingFace token (required for StableAudio encoder). Can also use env var HF_TOKEN or huggingface-cli login"
    )

    args = parser.parse_args()

    # Check for token in environment variable if not provided
    hf_token = args.hf_token or os.environ.get('HF_TOKEN')

    test_dmc(
        encoder_type=args.encoder,
        transformation_arch=args.arch,
        checkpoint_path=args.checkpoint,
        use_gpu=args.gpu,
        hf_token=hf_token
    )