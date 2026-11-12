import torch
from torch import nn, Tensor
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

from clfm.utils.grid import RectangularGrid
from clfm.utils import reparameterize


def cov_func(x: Tensor, cov_len: float, variance: float):
    """
    Evaluate true covariance function where x is a column vector - tensor of shape (n_x, 1)
    """
    dx = x - x.T
    return variance * torch.exp(-(dx**2) / (2 * cov_len**2))


def log_normal_covariance(mu: Tensor, cov: Tensor, a=0.0, b=1.0):
    """
    Compute the covariance of the log-normal process from the Gaussian process covariance.

    Parameters:
        mu (Tensor): Mean of the original GP (shape (n_x, 1)).
        cov (Tensor): Covariance matrix of the original GP (shape (n_x, n_x)).
        a (float): Offset parameter of the log-normal transformation.
        b (float): Scale parameter of the log-normal transformation.

    Returns:
        Tensor: Covariance matrix of the log-normal process (shape (n_x, n_x)).
    """
    mu = mu.view(-1, 1)
    var_g = torch.diag(cov).view(1, -1)
    log_normal_cov = (
        b**2 * torch.exp(mu + mu.T + 0.5 * (var_g + var_g.T)) * (torch.exp(cov) - 1)
    )
    return log_normal_cov


class LogNormalDataset(Dataset):

    def __init__(
        self,
        N: int = 1000,
        num_sensors: int = 100,
        x_min: float = 0.0,
        x_max: float = 1.0,
        discretization: int = 100,
        cov_len: float = 0.25,
        variance: float = 1.0,
        a: float = 0.0,
        b: float = 1.0,
        samples_dir: str = None,
    ):
        super().__init__()
        self.variance = variance
        self.cov_len = cov_len
        self.a = a
        self.b = b

        # define grid on which to generate samples
        self.x_min = x_min
        self.x_max = x_max
        self.x = torch.linspace(x_min, x_max, discretization).reshape(-1, 1)
        self.sensor_indices = self._get_sensor_indices(num_sensors, discretization)

        if samples_dir is None:
            # compute the Cholesky factorization of the covariance matrix
            cov = cov_func(self.x, cov_len, self.variance)
            cov += 1e-5 * torch.eye(discretization)
            L = torch.tensor(np.linalg.cholesky(cov.numpy()))

            u = torch.randn(discretization, N)
            gp_samples = torch.matmul(L, u)
            log_normal_samples = a + b * torch.exp(gp_samples)

            self.samples = log_normal_samples.T  # transpose to (N, n_x)
        else:
            self.samples = torch.load(
                Path(samples_dir) / "samples.pt", weights_only=True
            )

        # required attributes
        self.grid = RectangularGrid(self.x_min, self.x_max)
        self.num_fields = 1
        self.num_sensors = num_sensors

    def __len__(self) -> int:
        return self.samples.shape[0]

    def __getitem__(self, idx: int):
        return (self.samples[idx, self.sensor_indices], self.x[self.sensor_indices, :])

    def _get_sensor_indices(self, num_sensors, discretization):
        if num_sensors == 1:
            sensor_indices = torch.tensor([round(discretization / 2)], dtype=torch.long)
        elif num_sensors == 2:
            sensor_indices = torch.tensor(
                [round(discretization / 3), round(2 * discretization / 3)],
                dtype=torch.long,
            )
        else:
            sensor_indices = torch.linspace(
                0, discretization - 1, num_sensors, dtype=torch.long
            )
        return sensor_indices

    def dense_eval(self, idx: int, n_x: int = None):
        return (self.samples[idx], self.x[:, :])

    def store_samples(self, output_dir):
        output_file = Path(output_dir) / "samples.pt"
        torch.save(self.samples, output_file)


class LogNormalLoss(nn.Module):

    def __init__(
        self,
        train_data: LogNormalDataset,
        num_colloc: int,
        positivity_weight: float = 1.0,
        residual_weight: float = 0.1,
    ):
        super().__init__()
        self.cov_len = train_data.cov_len
        self.variance = train_data.variance
        self.a = train_data.a
        self.b = train_data.b
        self.num_colloc = num_colloc
        self.pos_weight = positivity_weight
        self.res_weight = residual_weight

    def reconstruction(self, vae, z: Tensor, x: Tensor, u_true: Tensor):
        u_pred = vae.decode(z, x)
        return torch.mean(torch.square(u_pred.squeeze() - u_true.squeeze()))

    def residual(self, vae, z: Tensor, x: Tensor = None):
        if x is None:
            x = vae.grid.sample(self.num_colloc).unsqueeze(0).repeat(z.shape[0], 1, 1)

        u_pred = vae.decode(z, x).squeeze()
        positivity_loss = torch.mean(torch.nn.functional.relu(-u_pred))

        if self.res_weight > 0:
            # assuming all x in the batch are on the same grid.
            gp_cov = cov_func(x[0, :, :], self.cov_len, self.variance)
            log_normal_cov = log_normal_covariance(
                torch.zeros_like(x[0, :, :]), gp_cov, self.a, self.b
            )
            sigma_hat = torch.cov(u_pred.T)
            cov_res = torch.mean(torch.square(log_normal_cov - sigma_hat))
            residual = self.res_weight * cov_res + self.pos_weight * positivity_loss
        else:
            residual = self.pos_weight * positivity_loss
        return residual, {"residual": residual.item()}

    def validate(self, vae, u: Tensor, x_data: Tensor):
        NUM_VAL_PTS = 100
        z = reparameterize(*vae.encode(u))
        x = vae.grid.sample(NUM_VAL_PTS).unsqueeze(0).repeat(z.shape[0], 1, 1)
        y_pred = vae.decode(z, x).squeeze()
        # assuming all x in the batch are on the same grid.
        gp_cov = cov_func(x[0, :, :], self.cov_len, self.variance)
        log_normal_cov = log_normal_covariance(
            torch.zeros_like(x[0, :, :]), gp_cov, self.a, self.b
        )
        cov_gen = torch.cov(y_pred.T)
        positivity_loss = self.pos_weight * torch.mean(
            torch.nn.functional.relu(-y_pred)
        )
        cov_mse = torch.mean(torch.square(log_normal_cov - cov_gen))
        val_loss = cov_mse + positivity_loss

        return {
            "val_cov_mse": cov_mse.item(),
            "val_loss": val_loss.item(),
            "val_pos_loss": positivity_loss.item(),
        }
