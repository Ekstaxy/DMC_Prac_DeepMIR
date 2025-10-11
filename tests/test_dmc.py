import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from models.model_DMC import DifferentiableMixingConsole


sr = 44100
seq_len = 131072
bs = 3
num_tracks = 8
channel = 2


x = torch.randn(bs, num_tracks, channel, seq_len)
print(x.shape)

model = DifferentiableMixingConsole(sr)

y, p = model(x)
print(y.shape)
print(p.shape)