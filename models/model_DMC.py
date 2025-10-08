import torch
import torch.nn as nn

import torchaudio

class VGGishEncoder(nn.Module):
    def __init__(self, sample_rate, pretrained=True):
        super(VGGishEncoder, self).__init__()
        self.model = torch.hub.load("harritaylor/torchvggish", "vggish")
        self.model.eval()  # Set to evaluation mode
        self.sample_rate = sample_rate

        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        if len(x.shape) == 3:
            x = x.squeeze(1)  # Remove channel dimension
        batch_size, num_samples = x.shape
        if self.sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=self.sample_rate, new_freq=16000)
            x = resampler(x)

        z = []
        with torch.no_grad():
            for bidx in range(batch_size):
                x_item = x[bidx : bidx + 1, :]
                x_item = x_item.permute(1, 0)
                x_item = x_item.cpu().view(-1).numpy()
                z_item = self.model(x_item, fs=16000)
                z_item = z_item.mean(dim=0)  # mean across time frames
                z.append(z_item)

            if len(z) > 0:
                z = torch.stack(z, dim=0)   
            else:
                z = torch.empty((0,))

        return z
    
class PostProcessor(nn.Module):
    def __init__(self, input_dim=128, output_dim=10):
        super(PostProcessor, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.prelu1 = nn.PReLU()
        self.dropout1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.prelu2 = nn.PReLU()
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.prelu3 = nn.PReLU()
        self.dropout3 = nn.Dropout(0.1)

        self.fc4 = nn.Linear(32, output_dim)
        self.tanh = nn.Tanh()

    def forward(self, track_emb, context):
        x = torch.cat([track_emb, context], dim=-1)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.prelu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = self.prelu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        x = self.bn3(x)
        x = self.prelu3(x)
        x = self.dropout3(x)

        x = self.fc4(x)
        x = self.tanh(x)

        return x

class TCNBlock(nn.Module):
    """
    TCN block as described in the DMC paper (Fig. 2)
    """
    def __init__(self, channels, kernel_size=15, dilation=1, cglobal_dim=128):
        """
        Args:
            channels: number of channels (e.g., 128)
            kernel_size: convolution kernel size (default: 15)
            dilation: dilation factor (exponentially increasing)
            cglobal_dim: dimension of c_global vector (default: 128)
        """
        super(TCNBlock, self).__init__()
        
        # Dilated 1D Convolution
        padding = (kernel_size - 1) * dilation  # Causal padding
        self.conv1d = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            bias=False  # No bias because BatchNorm follows
        )
        
        # Batch Normalization (WITHOUT affine transformation)
        self.bn = nn.BatchNorm1d(channels, affine=False)
        
        # FiLM: Project c_global to channel dimension
        self.gamma_linear = nn.Linear(cglobal_dim, channels)
        self.beta_linear = nn.Linear(cglobal_dim, channels)
        
        # PReLU activation
        self.prelu = nn.PReLU()
        
        # Learnable gain for residual connection (g_n in diagram)
        self.residual_gain = nn.Parameter(torch.ones(1))
    
    def forward(self, x, cglobal):
        """
        Args:
            x: input features [batch, channels, length]
            cglobal: global conditioning [batch, cglobal_dim]
        
        Returns:
            output: [batch, channels, length]
        """
        # Save input for residual connection
        identity = x
        
        # 1. Conv1d (dilated convolution)
        out = self.conv1d(x)
        
        # Crop to match input length (causal)
        if out.shape[-1] != x.shape[-1]:
            out = out[..., :x.shape[-1]]
        
        # 2. BatchNorm (without affine)
        out = self.bn(out)
        
        # 3. FiLM modulation
        gamma = self.gamma_linear(cglobal)  # [batch, channels]
        beta = self.beta_linear(cglobal)    # [batch, channels]
        
        # Reshape for broadcasting: [batch, channels, 1]
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        
        # Apply FiLM: γ ⊙ x + β
        out = gamma * out + beta
        
        # 4. PReLU activation
        out = self.prelu(out)
        
        # 5. Residual connection with learnable gain (+ in diagram)
        out = out + self.residual_gain * identity
        
        return out

class ConditioningMLP(nn.Module):
    """
    3-layer MLP to generate c_global from effect parameters
    """
    def __init__(self, num_params=26, hidden_dim=128, output_dim=128):
        super(ConditioningMLP, self).__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(num_params, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, params):
        """
        Args:
            params: effect parameters [batch, num_params]
        
        Returns:
            cglobal: [batch, output_dim]
        """
        return self.mlp(params)
    
class TransformationNetwork(nn.Module):
    """
    Complete TCN-based transformation network
    Support TCN-10, TCN-20, TCN-30 configurations
    """
    def __init__(
            self, 
            num_blocks=10, 
            channels=128, 
            kernel_size=15, 
            num_params=26, 
            cglobal_dim=128,
            min_gain_dB=-48.0,
            max_gain_dB=24.0
        ):
        super(TransformationNetwork, self).__init__()
        self.min_gain_dB = min_gain_dB
        self.max_gain_dB = max_gain_dB

        # Define TCN blocks based on configuration
        self.tcn_blocks = nn.ModuleList()
        for layer_idx in range(num_blocks):
            dilation = 2 ** ((layer_idx) % 10)
            self.tcn_blocks.append(TCNBlock(
                channels=channels,
                kernel_size=kernel_size,
                dilation=dilation,
                cglobal_dim=cglobal_dim
            ))
        self.conditioning_mlp = ConditioningMLP(
            num_params=num_params, 
            hidden_dim=128, 
            output_dim=cglobal_dim
        )
    
    def forward(self, x, params):
        """
        Args:
            x: input features [batch, num_tracks, length]
            params: effect parameters [batch, num_params]
        
        Returns:
            output: transformed features [batch, num_tracks, length]
        """
        batch_size, num_tracks, length = x.shape

        gain_dB = params[:, 0]  # [batch]
        gain_dB = (gain_dB - self.min_gain_dB)/(self.max_gain_dB - self.min_gain_dB)
        gain_lin = 10 ** (gain_dB / 20.0)
        gain_lin = gain_lin.view(batch_size, num_tracks, 1)
        x = x * gain_lin

        x = x.view(batch_size, num_tracks, 1, -1)  # (bs, num_tracks, 1, seq_len)
        x = x.repeat(1, 1, 2, 1) 

        pan = params[:, 1]  # [batch]
        pan_theta = pan*(torch.pi/2)
        left_gain = torch.cos(pan_theta)
        right_gain = torch.sin(pan_theta)
        pan_gains_lin = torch.stack([left_gain, right_gain], dim=1)
        pan_gains_lin = pan_gains_lin.view(batch_size, num_tracks, 2, 1)
        x *= pan_gains_lin

        # Generate c_global from effect parameters
        cglobal = self.conditioning_mlp(params)  # [batch, cglobal_dim]
        
        # Pass through TCN blocks
        skip = torch.zeros_like(x)
        out = x
        for block in self.tcn_blocks:
            out = block(out, cglobal)
            skip = skip + out

        out = out + (skip/len(self.tcn_blocks))

        post_gain_dB = params[:, 24]  # [batch]
        post_gain_dB = (post_gain_dB - self.min_gain_dB)/(self.max_gain_dB - self.min_gain_dB)
        post_gain_lin = 10 ** (post_gain_dB / 20.0)
        post_gain_lin = post_gain_lin.view(batch_size, num_tracks, 1, 1)
        out = out * post_gain_lin

        post_pan = params[:, 25]  # [batch]
        post_pan_theta = post_pan*(torch.pi/2)
        post_left_gain = torch.cos(post_pan_theta)
        post_right_gain = torch.sin(post_pan_theta)
        post_pan_gains_lin = torch.stack([post_left_gain, post_right_gain], dim=1)
        post_pan_gains_lin = post_pan_gains_lin.view(batch_size, num_tracks, 2, 1)
        out = out * post_pan_gains_lin

        y = torch.sum(out, dim=1)  # Sum over tracks

        return y, params

class SimpleTransformationNetwork(nn.Module):
    def __init__(self, sample_rate: float, min_gain_dB: int = -48.0, max_gain_dB: int = 24.0):
        super(SimpleTransformationNetwork, self).__init__()
        self.sample_rate = sample_rate
        self.min_gain_dB = min_gain_dB
        self.max_gain_dB = max_gain_dB

    def forward(self, x: torch.Tensor, params: torch.Tensor):
        """Simplified transformation network to apply gain and panning.

        Args:
            x (torch.Tensor): Input waveform stems with shape (bs, num_tracks, seq_len).
            params (torch.Tensor): Mixing parameters with shape (bs, num_tracks, 2).

        Returns:
            torch.Tensor: Transformed waveform with shape (bs, 2, seq_len).
        """
        bs, num_tracks, seq_len = x.size()

        # Apply gain
        gain_dB = params[..., 0]  # Extract gain parameter
        gain_dB = (gain_dB - self.min_gain_dB) / (self.max_gain_dB - self.min_gain_dB)
        gain_lin = 10 ** (gain_dB / 20.0)  # Convert dB to linear scale
        gain_lin = gain_lin.view(bs, num_tracks, 1)
        x = x * gain_lin  # Apply gain

        # Apply panning
        x = x.view(bs, num_tracks, 1, -1).repeat(1, 1, 2, 1)  # Expand to stereo
        pan = params[..., 1]  # Extract pan parameter
        pan_theta = pan * torch.pi / 2
        left_gain = torch.cos(pan_theta)
        right_gain = torch.sin(pan_theta)
        pan_gains_lin = torch.stack([left_gain, right_gain], dim=-1).view(bs, num_tracks, 2, 1)
        x = x * pan_gains_lin  # Apply panning

        # Mix tracks
        y = torch.sum(x, dim=1)  # Sum tracks to create stereo mix

        return y
    
class DifferentiableMixingConsole(nn.Module):
    def __init__(self, sample_rate=44100, transformation_arch="Original")->None:
        super(DifferentiableMixingConsole, self).__init__()
        self.sample_rate = sample_rate
        if transformation_arch == "Original":
            self.transformation_network = TransformationNetwork()
        elif transformation_arch == "Simple":
            self.transformation_network = SimpleTransformationNetwork()

        self.encoder = VGGishEncoder(sample_rate=sample_rate, pretrained=True)
        self.post_processor = PostProcessor(input_dim=128+10, output_dim=10)
        # Additional initialization code can be added here
    
    def forward(self, x, track_mask=None):
        batch_size, num_tracks, seq_len = x.size()

        # if no track_mask supplied assume all tracks active
        if track_mask is None:
            track_mask = torch.zeros(batch_size, num_tracks).type_as(x).bool()

        # move tracks to the batch dimension to fully parallelize embedding computation
        x = x.view(batch_size * num_tracks, -1)

        # generate single embedding for each track
        e = self.encoder(x)
        e = e.view(batch_size, num_tracks, -1)  # (bs, num_tracks, d_embed)

        # generate the "context" embedding
        c = []
        for bidx in range(batch_size):
            c_n = e[bidx, ~track_mask[bidx, :], :].mean(
                dim=0, keepdim=True
            )  # (bs, 1, d_embed)
            c_n = c_n.repeat(num_tracks, 1)  # (bs, num_tracks, d_embed)
            c.append(c_n)
        c = torch.stack(c, dim=0)

        # fuse the track embs and context embs
        ec = torch.cat((e, c), dim=-1)  # (bs, num_tracks, d_embed*2)

        # estimate mixing parameters for each track (in parallel)
        p = self.post_processor(ec)  # (bs, num_tracks, num_params)

        # generate the stereo mix
        x = x.view(batch_size, num_tracks, -1)  # move tracks back from batch dim
        y, params = self.transformation_network(x, p)  # (bs, 2, seq_len) # and denormalized params

        return y, params
