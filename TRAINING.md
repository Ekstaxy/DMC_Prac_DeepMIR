# Training the Transformation Network

## Quick Start

Train TCN-10 (default):
```bash
python train/transnetwork_train.py --data_dir path/to/ENST-drums
```

Train TCN-20:
```bash
python train/transnetwork_train.py --data_dir path/to/ENST-drums --tcn_blocks 20
```

Train TCN-30:
```bash
python train/transnetwork_train.py --data_dir path/to/ENST-drums --tcn_blocks 30
```

## Arguments

- `--data_dir`: Path to ENST-drums dataset (required)
- `--tcn_blocks`: Number of TCN blocks: 10, 20, or 30 (default: 10)
- `--batch_size`: Batch size (default: 32)
- `--epochs`: Number of epochs (default: 200)
- `--lr`: Learning rate (default: 3e-4)
- `--sample_rate`: Sample rate (default: 44100)
- `--length`: Audio length in samples (default: 65536)
- `--patience`: LR scheduler patience (default: 20)
- `--results_dir`: Results directory (default: results)

## Output

Results are saved in `results/TCN-{blocks}/`:
- `best_model.pth`: Best validation checkpoint
- `final_model.pth`: Final checkpoint
- `config.json`: Training configuration
- `metrics.json`: Train/val losses
- `loss_curve.png`: Loss vs epoch plot
