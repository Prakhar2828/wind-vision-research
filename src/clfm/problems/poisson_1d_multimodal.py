import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
from torch.distributions import Normal, Categorical
from scipy.integrate import solve_bvp
import numpy as np
from pathlib import Path

from clfm.utils.utils import grad
from clfm.utils.grid import RectangularGrid


def h_func(x, A):
    return 1 + A * x**2


def dh_dx(x, A):
    return 2 * A * x


def f_func(x):
    x = torch.tensor(x)
    return torch.sin(x)


def ode_system(x, y, A):
    _, v = y
    du_dx = v
    dv_dx = -f_func(x) / h_func(x, A) - (dh_dx(x, A) / h_func(x, A)) * v
    return np.vstack((du_dx, dv_dx))


def bc(ya, yb):
    return np.array([ya[0], yb[0]])  # u(0) = 0, u(pi) = 0


def compute_u(x_grid, A):
    u_sols = np.zeros((len(A), len(x_grid)), dtype=np.float32)
    u_init = np.zeros((2, len(x_grid)))
    for i, A_sample in enumerate(A):
        sol = solve_bvp(lambda x, u: ode_system(x, u, A_sample), bc, x_grid, u_init)
        u_sols[i, :] = sol.y[0]
    return u_sols


def sample_A(N):
    """Sample parameter A from a bimodal Gaussian mixture distribution."""
    mean1, std1 = 0.15, 0.015
    mean2, std2 = 0.35, 0.035
    weights = [0.3, 0.7]

    gaussian1 = Normal(torch.tensor(mean1), torch.tensor(std1))
    gaussian2 = Normal(torch.tensor(mean2), torch.tensor(std2))
    mixing_dist = Categorical(torch.tensor(weights))

    choices = mixing_dist.sample((N,))
    samples_gaussian1 = gaussian1.sample((N,))
    samples_gaussian2 = gaussian2.sample((N,))

    samples = torch.where(choices == 0, samples_gaussian1, samples_gaussian2)
    return samples


class Poisson1DMultimodalDataset(Dataset):

    def __init__(
        self,
        N: int = None,
        num_sensors: int = None,
        x_min: float = 0.0,
        x_max: float = torch.pi,
        samples_dir: str = None,
    ):
        super().__init__()
        self.x_min = x_min
        self.x_max = x_max
        self.grid = RectangularGrid(x_min, x_max)

        if samples_dir is None:
            A = sample_A(N)
            self.x_sensor = self.grid.dense(num_sensors).reshape(-1, 1)
            self.h_samples = torch.tensor(h_func(self.x_sensor, A)).T
            self.u_samples = torch.tensor(compute_u(self.x_sensor[:, 0], A))
        else:
            self.h_samples = torch.load(
                Path(samples_dir) / "h_samples.pt", weights_only=True
            )
            self.u_samples = torch.load(
                Path(samples_dir) / "u_samples.pt", weights_only=True
            )
            N, num_sensors = self.h_samples.shape
            self.x_sensor = self.grid.dense(num_sensors).reshape(-1, 1)

        # required attributes
        self.num_fields = 2
        self.N = N
        self.num_sensors = num_sensors

    def __len__(self):
        return self.N

    def __getitem__(self, idx: int):
        return (self.u_samples[idx], self.x_sensor)

    def dense_eval(self, idx: int, num_sensors: int = 100):
        x = torch.linspace(self.x_min, self.x_max, num_sensors).reshape(-1, 1)
        return self.u_samples[idx], x

    def store_samples(self, output_dir):
        torch.save(self.h_samples, Path(output_dir) / "h_samples.pt")
        torch.save(self.u_samples, Path(output_dir) / "u_samples.pt")


class Poisson1DMultimodalLoss(nn.Module):
    def __init__(self, num_colloc: int):
        super().__init__()
        self.num_colloc = num_colloc

    def reconstruction(self, vae, z: Tensor, x: Tensor, u_true: Tensor):
        f = vae.decode(z, x)
        u, _ = f.chunk(2, dim=2)
        u = u.squeeze(2)
        return torch.mean(torch.square(u - u_true))

    def residual(self, vae, z: Tensor, x: Tensor = None):
        if x is None:
            x = vae.grid.sample(z.shape[0] * self.num_colloc)
            x = x.reshape(z.shape[0], self.num_colloc, -1)

        x.requires_grad_()
        f_target = f_func(x)
        u_h_gen = vae.decode(z, x)
        u, h = u_h_gen.chunk(2, dim=2)
        du_dx = grad(u, x)
        h_du_dx = h * du_dx
        d_h_du_dx_dx = grad(h_du_dx, x)

        residual = torch.mean(torch.square(d_h_du_dx_dx + f_target))
        return residual, {"residual": residual.item()}

    def validate(self, vae, *args):
        return {}
