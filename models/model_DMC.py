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

class StableAudioEncoder(nn.Module):
    """
    Encoder using Stable Audio Open VAE for audio embedding extraction.
    Similar to VGGishEncoder but uses the Stable Audio Open model.

    Note: Requires HuggingFace login for downloading the model.
    Make sure to run: huggingface-cli login
    """
    def __init__(self, sample_rate, pretrained=True, embedding_dim=128):
        super(StableAudioEncoder, self).__init__()
        self.sample_rate = sample_rate
        self.embedding_dim = embedding_dim

        # Load Stable Audio Open VAE encoder
        if pretrained:
            # Lazy import to avoid requiring diffusers when not using StableAudio encoder
            try:
                from diffusers import AutoencoderOobleck
            except ImportError:
                raise ImportError(
                    "diffusers package is required for StableAudioEncoder. "
                    "Install it with: pip install diffusers"
                )

            print("Loading Stable Audio Open VAE encoder...")
            print("Note: This requires HuggingFace authentication.")
            print("If download fails, run: huggingface-cli login")

            try:
                self.vae = AutoencoderOobleck.from_pretrained(
                    "stabilityai/stable-audio-open-1.0",
                    subfolder="vae"
                )
                self.vae.eval()
                print("Stable Audio Open VAE encoder loaded")
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Stable Audio Open model: {e}\n"
                    "Make sure you are logged in to HuggingFace: huggingface-cli login"
                )
        else:
            raise ValueError("Non-pretrained Stable Audio Open encoder not supported")

        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False

        # Projection layer to map latent features to embedding_dim (128 to match VGGish)
        # The VAE encoder outputs latents with 64 channels, we'll pool and project
        self.projection = nn.Linear(64, embedding_dim)

    def forward(self, x):
        """
        Args:
            x: input audio [batch_size, num_samples] (mono)

        Returns:
            z: embeddings [batch_size, embedding_dim] (default 128)
        """
        if len(x.shape) == 3:
            x = x.squeeze(1)  # Remove channel dimension if present

        batch_size, num_samples = x.shape

        # Resample to 44100 Hz if needed (Stable Audio Open expects 44100 Hz)
        if self.sample_rate != 44100:
            resampler = torchaudio.transforms.Resample(
                orig_freq=self.sample_rate,
                new_freq=44100
            ).to(x.device)
            x = resampler(x)
            num_samples = x.shape[1]

        # Convert mono to stereo (Stable Audio Open expects stereo input)
        x_stereo = x.unsqueeze(1).repeat(1, 2, 1)  # [batch_size, 2, num_samples]

        # Ensure minimum length for VAE (pad if necessary)
        min_length = 44100  # At least 1 second
        if num_samples < min_length:
            pad_length = min_length - num_samples
            x_stereo = torch.nn.functional.pad(x_stereo, (0, pad_length))

        z = []
        with torch.no_grad():
            for bidx in range(batch_size):
                x_item = x_stereo[bidx:bidx+1]  # [1, 2, num_samples]

                # Encode with VAE
                latent_dist = self.vae.encode(x_item).latent_dist
                z_item = latent_dist.sample()  # [1, channels, latent_time]

                # Pool across time dimension to get fixed-size representation
                z_item = z_item.mean(dim=-1)  # [1, channels]
                z_item = z_item.squeeze(0)     # [channels]

                z.append(z_item)

        if len(z) > 0:
            z = torch.stack(z, dim=0)  # [batch_size, channels]
        else:
            z = torch.empty((0, 64))

        # Project to embedding_dim (128 to match VGGish)
        z = self.projection(z)  # [batch_size, embedding_dim]

        return z

class PostProcessor(nn.Module):
    def __init__(self, input_dim=128, output_dim=26):
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
        batch_size, num_tracks, _ = track_emb.shape

        x = torch.cat([track_emb, context], dim=-1)  # [batch, num_tracks, input_dim]

        # Flatten for processing: [batch*num_tracks, input_dim]
        x = x.view(batch_size * num_tracks, -1)

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

        # Reshape back: [batch, num_tracks, output_dim]
        x = x.view(batch_size, num_tracks, -1)

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

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Conv1d: Kaiming initialization (good for ReLU-like activations)
        nn.init.kaiming_normal_(self.conv1d.weight, mode='fan_out', nonlinearity='relu')

        # FiLM gamma: Initialize close to 1 (identity)
        nn.init.normal_(self.gamma_linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.gamma_linear.bias, 1.0)

        # FiLM beta: Initialize close to 0 (no shift)
        nn.init.normal_(self.beta_linear.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.beta_linear.bias, 0.0)

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
        self.input_projection = nn.Conv1d(2, channels, kernel_size=1)
        self.output_projection = nn.Conv1d(channels, 2, kernel_size=1)
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
            x: input features [batch, num_tracks, channel, length]
            params: effect parameters [batch, num_tracks, num_params]
        
        Returns:
            output: transformed features [batch, num_tracks, channel, length]
        """
        batch_size, num_tracks, channel, length = x.shape
        if channel == 1:
            x = x.repeat(1, 1, 2, 1)  # Convert mono to stereo
            channel = 2  # Update channel count after conversion

        gain_dB = params[:, :, 0]  # [batch, num_tracks]
        gain_dB = (gain_dB - self.min_gain_dB)/(self.max_gain_dB - self.min_gain_dB)
        gain_lin = 10 ** (gain_dB / 20.0)
        gain_lin = gain_lin.view(batch_size, num_tracks, 1, 1)  # [batch, num_tracks, 1, 1]
        x = x * gain_lin

        pan = params[:, :, 1]  # [batch, num_tracks]
        pan_theta = pan*(torch.pi/2)
        left_gain = torch.cos(pan_theta)
        right_gain = torch.sin(pan_theta)
        pan_gains_lin = torch.stack([left_gain, right_gain], dim=2)  # [batch, num_tracks, 2]
        pan_gains_lin = pan_gains_lin.unsqueeze(-1)  # [batch, num_tracks, 2, 1]
        x *= pan_gains_lin

        # Reshape params for per-track processing: [batch*num_tracks, num_params]
        params_flat = params.view(batch_size * num_tracks, -1)

        # Generate c_global from effect parameters for each track
        cglobal = self.conditioning_mlp(params_flat)  # [batch*num_tracks, cglobal_dim]

        # Reshape x for per-track TCN processing: [batch*num_tracks, channels, length]
        x = x.view(batch_size * num_tracks, channel, length)
        x = self.input_projection(x)  # Project input to match TCN channels

        # Pass through TCN blocks (each track separately)
        skip = torch.zeros_like(x)
        out = x
        for block in self.tcn_blocks:
            out = block(out, cglobal)
            skip = skip + out

        out = out + (skip/len(self.tcn_blocks))

        # Project back to 2 channels: [batch*num_tracks, 128, length] → [batch*num_tracks, 2, length]
        out = self.output_projection(out)

        # Reshape back to [batch, num_tracks, channels, length]
        out = out.view(batch_size, num_tracks, channel, length)

        post_gain_dB = params[:, :, 24]  # [batch, num_tracks]
        post_gain_dB = (post_gain_dB - self.min_gain_dB)/(self.max_gain_dB - self.min_gain_dB)
        post_gain_lin = 10 ** (post_gain_dB / 20.0)
        post_gain_lin = post_gain_lin.view(batch_size, num_tracks, 1, 1)
        out = out * post_gain_lin

        post_pan = params[:, :, 25]  # [batch, num_tracks]
        post_pan_theta = post_pan*(torch.pi/2)
        post_left_gain = torch.cos(post_pan_theta)
        post_right_gain = torch.sin(post_pan_theta)
        post_pan_gains_lin = torch.stack([post_left_gain, post_right_gain], dim=2)  # [batch, num_tracks, 2]
        post_pan_gains_lin = post_pan_gains_lin.unsqueeze(-1)  # [batch, num_tracks, 2, 1]
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
            x (torch.Tensor): Input waveform stems with shape (batch, num_tracks, channel, seq_len).
            params (torch.Tensor): Mixing parameters with shape (batch, num_tracks, num_params).

        Returns:
            torch.Tensor: Transformed waveform with shape (batch, num_tracks, channel, seq_len).
        """
        batch, num_tracks, channel, seq_len = x.size()

        # Apply gain
        gain_dB = params[:, :, 0]  # Extract gain parameter
        gain_dB = (gain_dB - self.min_gain_dB) / (self.max_gain_dB - self.min_gain_dB)
        gain_lin = 10 ** (gain_dB / 20.0)  # Convert dB to linear scale
        gain_lin = gain_lin.view(batch, num_tracks, 1)
        x = x * gain_lin  # Apply gain

        pan = params[:, :, 1]  # Extract pan parameter
        pan_theta = pan * torch.pi / 2
        left_gain = torch.cos(pan_theta)
        right_gain = torch.sin(pan_theta)
        pan_gains_lin = torch.stack([left_gain, right_gain], dim=-1).view(batch, num_tracks, 2, 1)
        x = x * pan_gains_lin  # Apply panning

        # Mix tracks
        y = torch.sum(x, dim=1)  # Sum tracks to create stereo mix

        return y, params
    
class DifferentiableMixingConsole(nn.Module):
    def __init__(self, sample_rate=44100, transformation_arch="Original", encoder_type="VGGish")->None:
        """
        Args:
            sample_rate: audio sample rate
            transformation_arch: transformation network architecture ("Original" or "Simple")
            encoder_type: encoder to use for audio embeddings ("VGGish" or "StableAudio")
        """
        super(DifferentiableMixingConsole, self).__init__()
        self.sample_rate = sample_rate
        self.encoder_type = encoder_type

        # Initialize transformation network
        if transformation_arch == "Original":
            self.transformation_network = TransformationNetwork()
        elif transformation_arch == "Simple":
            self.transformation_network = SimpleTransformationNetwork(sample_rate=sample_rate)
        else:
            raise ValueError(f"Unknown transformation_arch: {transformation_arch}")

        # Initialize encoder based on encoder_type
        if encoder_type == "VGGish":
            self.encoder = VGGishEncoder(sample_rate=sample_rate, pretrained=True)
            embedding_dim = 128
        elif encoder_type == "StableAudio":
            self.encoder = StableAudioEncoder(sample_rate=sample_rate, pretrained=True, embedding_dim=128)
            embedding_dim = 128
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}. Choose 'VGGish' or 'StableAudio'")

        self.post_processor = PostProcessor(input_dim=embedding_dim+embedding_dim, output_dim=26)  # track_emb + context
        # Additional initialization code can be added here

    def load_transformation_checkpoint(self, checkpoint_path, device='cpu'):
        """
        Load a pretrained transformation network checkpoint.

        Args:
            checkpoint_path: Path to the checkpoint file (.pth)
            device: Device to load the checkpoint on
        """
        print(f"Loading transformation network checkpoint from: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)

        # Handle DataParallel wrapper prefix if present
        if any(key.startswith('module.') for key in state_dict.keys()):
            state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}

        self.transformation_network.load_state_dict(state_dict)
        print("✓ Checkpoint loaded successfully!")

    def forward(self, x, track_mask=None):
        """
        Args:
            x: input features [batch, num_tracks, channel, length]
        """
        batch_size, num_tracks, channel, seq_len = x.size()

        # if no track_mask supplied assume all tracks active
        if track_mask is None:
            track_mask = torch.zeros(batch_size, num_tracks).type_as(x).bool()

        # Flatten stereo tracks to mono for encoder: [batch*num_tracks, channel*seq_len]
        # VGGish expects mono audio, so we'll mix stereo to mono or process channels separately
        x_mono = x.mean(dim=2)  # Average stereo channels to mono: [batch, num_tracks, seq_len]

        # move tracks to the batch dimension to fully parallelize embedding computation
        x_mono = x_mono.view(batch_size * num_tracks, -1)  # [batch*num_tracks, seq_len]

        # generate single embedding for each track
        e = self.encoder(x_mono)
        e = e.view(batch_size, num_tracks, -1)  # (bs, num_tracks, d_embed)

        # generate the "context" embedding
        c = []
        for bidx in range(batch_size):
            c_n = e[bidx, ~track_mask[bidx, :], :].mean(
                dim=0, keepdim=True
            )  # (1, d_embed)
            c_n = c_n.repeat(num_tracks, 1)  # (num_tracks, d_embed)
            c.append(c_n)
        c = torch.stack(c, dim=0)  # (bs, num_tracks, d_embed)

        # estimate mixing parameters for each track (in parallel)
        p = self.post_processor(e, c)  # (bs, num_tracks, num_params)

        # generate the stereo mix using the original stereo input
        y, params = self.transformation_network(x, p)  # (bs, 2, seq_len)

        return y, params