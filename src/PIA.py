"""
PIA Model for IVIM

Self-supervised physics-informed autoencoder for IVIM parameter estimation.
Uses an MLP encoder with physics-based decoding via the bi-exponential model.
"""
import torch
from torch import nn
from torch.nn import functional as F
from typing import List


class PIA(nn.Module):
    def __init__(self,
                 number_of_signals=8,
                 f_mean=0.2025,
                 Dt_mean=0.0013,
                 Dstar_mean=0.0335,
                 f_delta=0.1975,
                 Dt_delta=0.0012,
                 Dstar_delta=0.0315,
                 b_values=[0, 5, 50, 100, 200, 500, 800, 1000],
                 hidden_dims: List = None,
                 predictor_depth=1,
                 device='cpu'):
        super(PIA, self).__init__()

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        self.number_of_signals = number_of_signals
        self.f_mean = torch.tensor(f_mean, device=device)
        self.Dt_mean = torch.tensor(Dt_mean, device=device)
        self.Dstar_mean = torch.tensor(Dstar_mean, device=device)
        self.f_delta = torch.tensor(f_delta, device=device)
        self.Dt_delta = torch.tensor(Dt_delta, device=device)
        self.Dstar_delta = torch.tensor(Dstar_delta, device=device)
        self.b_values = torch.tensor(b_values, dtype=torch.float32, device=device)
        self.device = device
        self.softmax = nn.Softmax(dim=1)
        self.relu = nn.ReLU()

        modules = []
        # Build The Encoder
        in_channels = number_of_signals
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(in_features=in_channels, out_features=h_dim),
                    nn.LeakyReLU()
                )
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules).to(device)

        # f predictor
        f_predictor = []
        for _ in range(predictor_depth):
            f_predictor.append(nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                nn.LeakyReLU()))
        f_predictor.append(nn.Linear(hidden_dims[-1], 1))
        self.f_predictor = nn.Sequential(*f_predictor).to(device)

        # Dt predictor
        Dt_predictor = []
        for _ in range(predictor_depth):
            Dt_predictor.append(nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                nn.LeakyReLU()))
        Dt_predictor.append(nn.Linear(hidden_dims[-1], 1))
        self.Dt_predictor = nn.Sequential(*Dt_predictor).to(device)

        # D* predictor
        star_predictor = []
        for _ in range(predictor_depth):
            star_predictor.append(nn.Sequential(
                nn.Linear(hidden_dims[-1], hidden_dims[-1]),
                nn.LeakyReLU()))
        star_predictor.append(nn.Linear(hidden_dims[-1], 1))
        self.Dstar_predictor = nn.Sequential(*star_predictor).to(device)

    def encode(self, x):
        result = self.encoder(x)

        f_var = self.f_delta * torch.tanh(self.f_predictor(result))
        f = self.f_mean + f_var

        Dt_var = self.Dt_delta * torch.tanh(self.Dt_predictor(result))
        Dt = self.Dt_mean + Dt_var

        Dstar_var = self.Dstar_delta * torch.tanh(self.Dstar_predictor(result))
        Dstar = self.Dstar_mean + Dstar_var

        return f, Dt, Dstar

    def decode(self, f, Dt, Dstar):
        f = f.view(-1, 1)
        Dt = Dt.view(-1, 1)
        Dstar = Dstar.view(-1, 1)

        b = self.b_values.view(1, -1)  # s/mm^2 native units

        S = (1 - f) * torch.exp(-b * Dt) + f * torch.exp(-b * Dstar)
        return S.to(self.device)

    def forward(self, x):
        f, Dt, Dstar = self.encode(x)
        S_pred = self.decode(f, Dt, Dstar)
        return S_pred, x, f, Dt, Dstar

    def loss_function(self, pred_signal, true_signal, weights=None):
        if weights is not None:
            loss = torch.mean(weights * (pred_signal - true_signal) ** 2)
        else:
            loss = F.mse_loss(pred_signal, true_signal)
        return loss


class CNN_PIA(nn.Module):
    """CNN-based Physics-Informed Autoencoder for IVIM (Stage 1).

    Unlike the voxel-wise MLP PIA, this encoder operates on 2D spatial
    patches via Conv2d layers, capturing neighbor context for more
    spatially coherent parameter estimates.

    Architecture: 4-layer Conv2d encoder [8->64->64->128->64] with LeakyReLU,
    followed by three 1x1 conv prediction heads with sigmoid-scaled
    bounded outputs enforcing biophysical parameter ranges.

    Checkpoint: checkpoints/pia_cnn_best.pt
    """

    # Default biophysical parameter bounds (matching PIA MLP)
    PARAM_BOUNDS = {
        'f':     {'min': 0.005,   'range': 0.395},    # [0.005, 0.40]
        'Dt':    {'min': 0.0001,  'range': 0.0024},    # [0.0001, 0.0025]
        'Dstar': {'min': 0.002,   'range': 0.063},     # [0.002, 0.065]
    }

    def __init__(self,
                 b_values=[0, 5, 50, 100, 200, 500, 800, 1000],
                 device='cpu'):
        super(CNN_PIA, self).__init__()
        self.b_values = torch.tensor(b_values, dtype=torch.float32, device=device)
        self.device = device
        n_b = len(b_values)

        self.features = nn.Sequential(
            nn.Conv2d(n_b, 64, 3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, padding=1), nn.LeakyReLU(),
        )
        self.f_pred = nn.Conv2d(64, 1, 1)
        self.Dt_pred = nn.Conv2d(64, 1, 1)
        self.Dstar_pred = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 8, H, W) tensor of b-value DWI signal maps.

        Returns:
            f, Dt, Dstar: each (B, 1, H, W) bounded parameter maps.
        """
        feat = self.features(x)
        f = torch.sigmoid(self.f_pred(feat)) * self.PARAM_BOUNDS['f']['range'] \
            + self.PARAM_BOUNDS['f']['min']
        Dt = torch.sigmoid(self.Dt_pred(feat)) * self.PARAM_BOUNDS['Dt']['range'] \
            + self.PARAM_BOUNDS['Dt']['min']
        Dstar = torch.sigmoid(self.Dstar_pred(feat)) * self.PARAM_BOUNDS['Dstar']['range'] \
            + self.PARAM_BOUNDS['Dstar']['min']
        return f, Dt, Dstar

    def decode(self, f, Dt, Dstar):
        """Physics decoder: bi-exponential IVIM signal model.

        Args:
            f, Dt, Dstar: (B, 1, H, W) parameter maps.

        Returns:
            S: (B, N_b, H, W) reconstructed signal.
        """
        b = self.b_values.view(1, -1, 1, 1)  # (1, N_b, 1, 1)
        S = (1 - f) * torch.exp(-b * Dt) + f * torch.exp(-b * Dstar)
        return S

    def loss_function(self, pred_signal, true_signal, weights=None):
        if weights is not None:
            loss = torch.mean(weights * (pred_signal - true_signal) ** 2)
        else:
            loss = F.mse_loss(pred_signal, true_signal)
        return loss


class ResBlock(nn.Module):
    """Residual block with GroupNorm used in the UNetRefiner.

    Two 3x3 Conv2d layers with GroupNorm and LeakyReLU, plus a learned
    1x1 shortcut when input and output channels differ.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.gn1 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.gn2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.leaky_relu(self.gn1(self.conv1(x)))
        x = self.gn2(self.conv2(x))
        return F.leaky_relu(x + residual)


class UNetRefiner(nn.Module):
    """Stage 2 Multi-Scale Spatial U-Net Refiner (3.49M parameters).

    Takes all 3 Stage 1 parameter maps (f, Dt, D*) as a 3-channel input
    and outputs refined parameter maps. Uses a 3-level encoder-decoder
    with ResBlock stages, MaxPool2d downsampling, bilinear upsampling
    with skip concatenation, and a 1x1 final convolution.

    Exploits multi-scale 2D spatial context to remove salt-and-pepper
    noise artifacts from Stage 1 estimates. Trained with tumor-weighted
    supervised loss to preserve malignant lesion boundaries.

    Checkpoint: checkpoints/refiner_cnn_e2e_best.pt (end-to-end fine-tuned)
    """

    # Normalization constants for input/output parameter maps
    NORM = {
        'f':     {'min': 0.005,   'range': 0.395},
        'Dt':    {'min': 0.0001,  'range': 0.0024},
        'Dstar': {'min': 0.002,   'range': 0.063},
    }

    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = ResBlock(3, 64)
        self.enc2 = ResBlock(64, 128)
        self.enc3 = ResBlock(128, 256)
        # Bottleneck
        self.bottleneck = ResBlock(256, 256)
        # Decoder (input channels = upsampled + skip concatenation)
        self.dec3 = ResBlock(512, 128)   # 256 + 256 skip
        self.dec2 = ResBlock(256, 64)    # 128 + 128 skip
        self.dec1 = ResBlock(128, 64)    # 64 + 64 skip
        # Output
        self.final_conv = nn.Conv2d(64, 3, 1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        """Forward pass.

        Args:
            x: (B, 3, H, W) normalized parameter maps [f, Dt, D*].
               Expected to be min-max normalized to ~[0, 1] using NORM constants.

        Returns:
            (B, 3, H, W) refined parameter maps (same normalization space).
        """
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))
        d3 = self.dec3(torch.cat([
            F.interpolate(b, e3.shape[2:], mode='bilinear', align_corners=False), e3
        ], dim=1))
        d2 = self.dec2(torch.cat([
            F.interpolate(d3, e2.shape[2:], mode='bilinear', align_corners=False), e2
        ], dim=1))
        d1 = self.dec1(torch.cat([
            F.interpolate(d2, e1.shape[2:], mode='bilinear', align_corners=False), e1
        ], dim=1))
        return self.final_conv(d1)
