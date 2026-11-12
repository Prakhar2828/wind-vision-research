import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
import math


class FCEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.GELU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_size, output_size * 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """
        u is a tensor of size (batch_size, s1, ..., sn)
        num_sensors = s1 * ... * sn
        we flatten u so it is of size (batch_size, num_sensors)
        """
        x_flat = x.flatten(start_dim=1)
        return self.net(x_flat)


class FCTrunk(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_outputs, num_hidden_layers
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_outputs = num_outputs
        self.num_hidden_layers = num_hidden_layers

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.GELU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_size, output_size * num_outputs))
        layers.append(nn.GELU())
        layers.append(Rearrange("b n (i f) -> b n i f", f=num_outputs))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class FCBranch(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_hidden_layers = num_hidden_layers

        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.GELU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ResidualBlock(nn.Module):
    """
    Residual block for improved gradient flow in deeper networks.
    Can be used to enhance your VAE decoder's capacity.
    """

    def __init__(self, input_dim, hidden_dim=None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = input_dim

        # Main branch
        self.net = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(input_dim),
            nn.Linear(hidden_dim, input_dim),
        )

        self.activation = nn.SiLU()

    def forward(self, x):
        # Residual connection: output = activation(x + F(x))
        return self.activation(x + self.net(x))


class EnhancedBranchNetwork(nn.Module):
    """
    Enhanced decoder branch network with residual connections
    for better gradient flow and increased expressivity.
    """

    def __init__(self, latent_dim, hidden_dim, output_dim, num_blocks=3):
        super().__init__()
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # Project latent vector to working dimension
        h = self.input_proj(z)
        for block in self.blocks:
            h = block(h)
        return self.output_proj(h)


class FlowModel(nn.Module):
    def __init__(
        self, latent_dim, hidden_size, num_hidden_layers, z_mean=None, z_std=None
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.z_mean = z_mean
        self.z_std = z_std

        layers = []
        layers.append(nn.Linear(latent_dim + 1, hidden_size))
        layers.append(nn.GELU())

        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_size, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None, :]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding


class EnhancedFlowModel(nn.Module):
    def __init__(
        self,
        latent_dim,
        hidden_size,
        num_hidden_layers,
        time_embed_dim=64,
        z_mean=None,
        z_std=None,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        # Time embedding
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )

        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_size)

        # Hidden layers with LayerNorm and residual connections
        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(
                nn.ModuleDict(
                    {
                        "norm": nn.LayerNorm(hidden_size),
                        "linear": nn.Linear(hidden_size, hidden_size),
                        "act": nn.GELU(),
                    }
                )
            )

        # Output projection
        self.output_norm = nn.LayerNorm(hidden_size)
        self.output_proj = nn.Linear(hidden_size, latent_dim)

        # Initialize output layer to near-zero for stable training start
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

        self.z_mean = z_mean
        self.z_std = z_std

    def forward(self, x, t):
        # t shape: (batch,) or (batch, 1)
        if t.dim() == 2:
            t = t.squeeze(-1)

        # Time embedding
        t_emb = self.time_mlp(self.time_embed(t))  # (batch, hidden_size)

        # Input projection
        h = self.input_proj(x)  # (batch, hidden_size)

        # Add time embedding
        h = h + t_emb

        # Hidden layers with residual connections
        for layer in self.layers:
            h_res = h
            h = layer["norm"](h)
            h = layer["linear"](h)
            h = layer["act"](h)
            h = h + h_res  # Residual connection

        # Output
        h = self.output_norm(h)
        return self.output_proj(h)
