import torch
import torch.nn as nn
from typing import Optional

try:
    from torchdiffeq import odeint
except Exception:
    odeint = None


class LiquidBioCellODE(nn.Module):
    """Liquid cell compatible with torchdiffeq (ODE interface).

    The callable signature is f(t, y) and we capture a time-varying input sequence
    via closure in the kernel that calls this cell with the appropriate input.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.tau = nn.Parameter(torch.ones(hidden_dim))
        self.weight_in = nn.Linear(input_dim, hidden_dim)
        self.weight_rec = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, t: float, y: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        # x_t is the input at time t provided by the kernel closure
        innovation = torch.tanh(self.weight_in(x_t) + self.weight_rec(y))
        dydt = (-y / self.tau) + innovation
        return dydt


class SPI_OmniKernelODE(nn.Module):
    """Omni-kernel that integrates liquid cell dynamics using torchdiffeq.odeint.

    process_bio_stream accepts an array-like x_stream of shape [T, input_dim]
    and a t_span of length T. The implementation uses a closure to provide
    x_t to the LiquidBioCellODE by indexing into the provided stream.
    """

    def __init__(self, bio_input_dim: int, latent_dim: int, sentiment_dim: int):
        super().__init__()
        self.liquid_layer = LiquidBioCellODE(bio_input_dim, latent_dim)
        self.semantic_mapper = nn.Linear(latent_dim, sentiment_dim)

    def process_bio_stream(self, x_stream: torch.Tensor, t_span: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
        if odeint is None:
            raise RuntimeError('torchdiffeq.odeint is required but not installed.')

        if device is None:
            device = x_stream.device

        x_stream = x_stream.to(device)
        t_span = t_span.to(device)

        # initial state
        y0 = torch.zeros(self.liquid_layer.weight_rec.out_features, device=device)

        def func(t, y):
            # t is a scalar tensor; index the closest timestep
            idx = int(torch.clamp(t.round().long(), 0, x_stream.shape[0] - 1).item())
            x_t = x_stream[idx]
            return self.liquid_layer(t, y, x_t)

        # integrate over the requested time span
        out = odeint(func, y0, t_span)

        # out shape: [T, latent_dim]
        final = out[-1]
        return self.semantic_mapper(final)


if __name__ == '__main__':
    # quick self-check when run directly
    import numpy as np

    T = 10
    input_dim = 8
    latent_dim = 16
    sentiment_dim = 4

    kernel = SPI_OmniKernelODE(input_dim, latent_dim, sentiment_dim)
    x = torch.tensor(np.random.randn(T, input_dim).astype('float32'))
    t = torch.arange(0, T, dtype=torch.float32)
    if odeint is not None:
        out = kernel.process_bio_stream(x, t)
        print('Output shape:', out.shape)
    else:
        print('torchdiffeq not available; install it to run this example')
