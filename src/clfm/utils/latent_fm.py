import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchdyn.core import NeuralODE
from tqdm import tqdm

from clfm.nn.fully_connected_nets import FlowModel, EnhancedFlowModel
from clfm.utils import reparameterize


class torch_wrapper_simple(torch.nn.Module):
    """
    https://github.com/atong01/conditional-flow-matching/tree/main
    Wraps model to torchdyn compatible format.
    """

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        t = t.repeat(x.shape[0]).unsqueeze(-1)
        return self.model(torch.cat([x, t], dim=-1))


# For updated FM model w/ sinusoidal time embeddings
class torch_wrapper_enhanced(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, t, x, *args, **kwargs):
        t = t.expand(x.shape[0])
        return self.model(x, t)


def train_lfm(
    vae,
    train_data,
    latent_dim=64,
    hidden_layer_size=128,
    num_hidden_layers=3,
    num_epochs=1000,
    sigma_min=0.01,
    batch_size=256,
    lr=0.001,
    device="cuda",
    num_workers=0,
    use_enhanced_flow_model=False,
    time_embed_dim=64,
):
    """
    Trains a latent flow model (LFM) to learn the velocity field for generative modeling in the latent space of a pre-trained VAE.

    This function implements the training procedure where a neural network learns to
    predict the velocity field between random noise and encoded data points. The flow
    model is trained to predict the direction from noise to data samples in the latent
    space of the VAE.

    Parameters
    ----------
    vae : nn.Module
        Pre-trained variational autoencoder model used to encode training data into latent vectors
    train_data : Dataset
        PyTorch dataset containing training samples (pairs of data and coordinates)
    latent_dim : int, default=64
        Dimension of the latent space for both the VAE and flow model
    hidden_layer_size : int, default=128
        Size of hidden layers in the flow model
    num_hidden_layers : int, default=3
        Number of hidden layers in the flow model
    num_epochs : int, default=1000
        Number of training epochs
    sigma_min : float, default=0.01
        Minimum noise level added during the diffusion process
    batch_size : int, default=256
        Batch size for training
    lr : float, default=0.001
        Learning rate for the Adam optimizer
    device : str, default="cuda"
        Device to run the training on ("cpu", "cuda", "mps", ...)
    num_workers : int, default=0
        Number of workers for the data loader
    use_enhanced_flow_model : bool, default=False
        If True, use the EnhancedFlowModel with sinusoidal time embeddings
    time_embed_dim : int, default=64
        Dimension of time embedding (only used with enhanced flow model)

    Returns
    -------
    flow : FlowModel or EnhancedFlowModel
        The trained flow model with z_mean and z_std attributes
    loss_hist : list
        History of training losses (one value per epoch)

    Notes
    -----
    The training procedure follows a simplified diffusion model approach where:
    1. Data points are encoded into latent vectors z1 using the VAE encoder
    2. Latent statistics are computed for normalization
    3. Random noise vectors z0 are generated from a standard normal distribution
    4. Interpolated points are created between z0 and z1 at random timesteps t
    5. The flow model learns to predict the velocity field (z1 - z0) at these points
    """

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    # First pass: compute latent statistics for normalization
    print("Computing latent statistics...")
    all_z = []
    with torch.no_grad():
        for batch in train_loader:
            u, x = batch
            u = u.to(device)
            z = reparameterize(*vae.encode(u))
            all_z.append(z.cpu())
    all_z = torch.cat(all_z, dim=0)
    z_mean = all_z.mean(dim=0).to(device)
    z_std = all_z.std(dim=0).to(device) + 1e-6

    print("Latent mean:", z_mean)
    print("Latent std:", z_std)

    if use_enhanced_flow_model:
        flow = EnhancedFlowModel(
            latent_dim=latent_dim,
            hidden_size=hidden_layer_size,
            num_hidden_layers=num_hidden_layers,
            time_embed_dim=time_embed_dim,
        ).to(device)
    else:
        flow = FlowModel(
            latent_dim=latent_dim,
            hidden_size=hidden_layer_size,
            num_hidden_layers=num_hidden_layers,
        ).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)

    loss_hist = []

    for _ in tqdm(range(num_epochs)):
        epoch_losses = []
        for batch in train_loader:
            opt.zero_grad()
            u, x = batch
            u, x = u.to(device), x.to(device)

            with torch.no_grad():
                z1 = reparameterize(*vae.encode(u))
                # Normalize latent codes
                z1 = (z1 - z_mean) / z_std

            z0 = torch.randn_like(z1)
            t = torch.rand(z1.shape[0], device=device).unsqueeze(-1)
            mu_t = t * z1 + (1.0 - t) * z0
            zt = mu_t + sigma_min * torch.randn_like(mu_t)
            if use_enhanced_flow_model:
                vt = flow(zt, t)
            else:
                vt = flow(torch.cat([zt, t], dim=-1))
            loss = F.mse_loss(vt, z1 - z0)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(flow.parameters(), max_norm=1.0)

            opt.step()
            epoch_losses.append(loss.item())

        loss_hist.append(sum(epoch_losses) / len(epoch_losses))

    # Store normalization stats with the flow model
    flow.z_mean = z_mean
    flow.z_std = z_std

    return flow, loss_hist


def sample_lfm(
    num_samples,
    x_grid,
    flow_model,
    vae,
    num_time_steps=100,
    device="cuda",
    atol=1e-6,
    rtol=1e-6,
    use_enhanced_flow_model=False,
    decode_batch_size=None,
):
    """
    Generates samples using latent flow matching by simulating the learned flow in latent space and decoding the results.

    This function performs generative sampling through a two-step process:
    1. Uses a Neural ODE to simulate the learned flow, transforming random noise into meaningful latent vectors
    2. Decodes these latent vectors into function outputs using the VAE decoder at specified coordinates

    Parameters
    ----------
    num_samples : int
        Number of samples to generate
    x_grid : Tensor
        Tensor of shape (num_points, point_dim) specifying the coordinates at which to evaluate
        the generated functions
    flow_model : nn.Module
        Trained flow model that predicts the velocity field in the latent space
    vae : nn.Module
        Trained variational autoencoder with a decode method that maps from latent space to function space
    num_time_steps : int, default=100
        Number of integration steps for the Neural ODE solver
    device : str, default="cuda"
        Device on which to perform the computation ("cpu", "cuda", "mps", ...)
    atol : float, default=1e-6
        Absolute tolerance for the ODE solver
    rtol : float, default=1e-6
        Relative tolerance for the ODE solver
    use_enhanced_flow_model : bool, default=False
        If True, use the enhanced wrapper for models with sinusoidal time embeddings
    decode_batch_size : int, optional
        If set, decodes in chunks of this size to avoid OOM errors

    Returns
    -------
    y_gen : Tensor
        Generated function samples evaluated at the provided coordinates
        Shape: (num_samples, num_points, output_dim)
    z : Tensor
        Latent vectors used for decoding

    Notes
    -----
    The sampling process implements continuous normalizing flows in the latent space:
    1. Starting from random Gaussian noise in the latent space
    2. Integrating the learned velocity field using a Neural ODE solver
    3. Obtaining samples from the data distribution in latent space
    4. Decoding these samples to obtain function values at specified coordinates
    """

    node = NeuralODE(
        (
            torch_wrapper_enhanced(flow_model)
            if use_enhanced_flow_model
            else torch_wrapper_simple(flow_model)
        ),
        solver="dopri5",
        sensitivity="adjoint",
        atol=atol,
        rtol=rtol,
    ).to(device)

    with torch.no_grad():
        traj = node.trajectory(
            torch.randn(num_samples, flow_model.latent_dim, device=device),
            t_span=torch.linspace(0.0, 1.0, num_time_steps, device=device),
        )
    z = traj[-1, :, :]

    # Denormalize if stats are stored and it is not none
    if hasattr(flow_model, "z_mean") and flow_model.z_mean is not None:
        z = z * flow_model.z_std.to(device) + flow_model.z_mean.to(device)

    x_grid_device = x_grid.to(device)
    batch_size = decode_batch_size if decode_batch_size is not None else num_samples
    y_chunks = []
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            z_batch = z[i : i + batch_size]
            x_batch = x_grid_device[None, :, :].expand(z_batch.shape[0], -1, -1)
            y_chunks.append(vae.decode(z_batch, x_batch))
    y_gen = torch.cat(y_chunks, dim=0)

    return y_gen, z
