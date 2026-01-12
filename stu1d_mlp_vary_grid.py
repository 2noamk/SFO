import math
import os
from pathlib import Path
from typing import Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import hilbert


# =========================================================
# 1) STU basis in frequency domain
# =========================================================

@torch.no_grad()
def build_stu_basis_rfft(
    grid_size: int,
    k: int,
    basis: str = "hilbert",
    cache_dir: str | Path = "./basis_cache",
) -> torch.Tensor:
    """
    Build top-l eigenvectors of the Hilbert matrix in the frequency domain.

    Returns
    -------
    Phi_f : (l=num_hilbert_functions, n_r) complex64
        RFFT of the selected eigenvectors, where n_r = grid_size//2 + 1.
    """
    grid_size = int(grid_size)
    assert 1 <= k <= grid_size
    basis = basis.lower()
    assert basis == "hilbert"

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{basis}_rfft_L{grid_size}.pt"

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        w = payload["eigvals"]
        V_f = payload["eigvecs_rfft"]
        grid_size_r = grid_size // 2 + 1
        assert isinstance(w, torch.Tensor) and w.ndim == 1 and w.numel() == grid_size
        assert isinstance(V_f, torch.Tensor) and V_f.shape == (grid_size, grid_size_r)
    else:
        Z = torch.tensor(hilbert(grid_size), dtype=torch.float64)
        w, V = torch.linalg.eigh(Z)           # ascending λ
        V_t = V.T                             # (grid_size, grid_size)
        V_f = torch.fft.rfft(V_t, n=grid_size, dim=-1)  # (grid_size, grid_size_r)
        payload = {
            "eigvals": w.cpu(),
            "eigvecs_rfft": V_f.to(torch.complex64).cpu()
        }
        torch.save(payload, cache_path)

    idx = torch.argsort(w, descending=True)[:k]
    Phi_f = V_f[idx]  # (l = num of hilbert functions, grid_size_r)
    return Phi_f.to(torch.complex64)


# =========================================================
# 2) Core STU spatial operator
# =========================================================
class STULayer(nn.Module):
    def __init__(self, lifting_dim: int, k: int):
        super().__init__()

        self.lifting_dim = lifting_dim

        self.Theta = nn.Parameter(
            torch.empty(k, lifting_dim, lifting_dim)
        )
        nn.init.kaiming_uniform_(self.Theta, a=math.sqrt(5))

        self.norm = nn.LayerNorm(lifting_dim)

        # ---- Flash STU MLP block ----
        hidden_dim = 2 * lifting_dim  # expansion ratio like Flash-STU
        self.mlp_fc1 = nn.Conv1d(lifting_dim, hidden_dim, kernel_size=1)
        self.mlp_fc2 = nn.Conv1d(hidden_dim, lifting_dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, Phi_f: torch.Tensor) -> torch.Tensor:
        batch_size, lifting_dim, grid_size = x.shape
        assert lifting_dim == self.lifting_dim

        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        
        z = self.norm(x.transpose(1, 2)).transpose(1, 2)
        Xf = torch.fft.rfft(z, dim=-1)  # (B, d, R) complex
        Uf = Xf.unsqueeze(2) * Phi_f.conj().unsqueeze(0).unsqueeze(0)  # (B, d, L, R) complex

        # Uf = A + iB
        A = Uf.real  # (B, d, L, R) float
        B = Uf.imag  # (B, d, L, R) float

        # Mix d -> h using Theta (real), and sum over l directly
        Sf_re = torch.einsum("bdlr,lhd->bhr", A, self.Theta)  # (B, h, R) float
        Sf_im = torch.einsum("bdlr,lhd->bhr", B, self.Theta)  # (B, h, R) float

        Sf = torch.complex(Sf_re, Sf_im)  # (B, h, R) complex
        S = torch.fft.irfft(Sf, n=grid_size, dim=-1)     # (B, h, N) float

        # ---- Flash-STU MLP ----
        H = self.mlp_fc1(S)
        H = self.act(H)
        out = self.mlp_fc2(H)

        return x + out

# =========================================================
# 3) Local lifting + grid addition
# =========================================================

class LocalLift1D(nn.Module):
    """
    Local lifting from (batch_size, in_channel, grid_size) → (batch_size, lifting_dim, grid_size).
    """
    def __init__(
        self,
        in_channel: int,
        lifting_dim: int,
        kernel_size: int = 1,
        dilation: int = 1,
        norm: Optional[str] = None,  # "layer" | "group" | None
    ):
        super().__init__()
        assert in_channel > 0 and lifting_dim > 0
        assert kernel_size >= 1 and kernel_size % 2 == 1
        assert dilation >= 1

        self.in_channel = in_channel
        self.lifting_dim = lifting_dim
        self.kernel_size = kernel_size
        self.dilation = dilation

        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Conv1d(
            in_channels=in_channel,
            out_channels=lifting_dim,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            padding_mode="circular",
            bias=True,
        )
        
        if norm is None:
            self.norm = nn.Identity()
        elif norm.lower() == "layer":
            self.norm = nn.GroupNorm(num_groups=1, num_channels=lifting_dim, affine=True)
        elif norm.lower() == "group":
            # choose a divisor of c_out
            num_groups = 8
            while lifting_dim % num_groups != 0 and num_groups > 1:
                num_groups //= 2
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=lifting_dim, affine=True)
        else:
            raise ValueError(f"Unknown norm: {norm}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return self.norm(y)


class add_grid(nn.Module):
    """
    Add 1D grid
    """
    def __init__(self, grid_range: float = 1.0):
        super().__init__()
        self.grid_range = grid_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channel, grid_size)
        batch_size, in_channel, grid_size = x.shape
        device, dtype = x.device, x.dtype
        coords = torch.linspace(0, self.grid_range, grid_size, device=device, dtype=dtype) # cordsnstart at 0
        coords = coords.expand(batch_size, 1, grid_size)
        return torch.cat([x, coords], dim=1)

# =========================================================
# 4) Stacked spatial-only STU for 1D
# =========================================================


class StackedSTU1D(nn.Module):
    """
    Spatial-only STU stack.

    I/O:
      forward(a): a (batch_size, in_channel, grid_size) -> (batch_size, out_dim, grid_size)
    """
    def __init__(
        self,
        grid_size: int,
        grid_range: float,
        in_channel: int,
        out_dim: int,
        *,
        lifting_dim: int = 64,
        depth_space: int = 4,
        k_space: int = 8,
        basis_cache: str | Path = "./basis_cache",
        lift_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.grid_size = int(grid_size)
        self.grid_range = float(grid_range)
        self.in_channel = int(in_channel)
        self.out_dim = int(out_dim)
        self.lifting_dim = int(lifting_dim)
        self.depth_space = int(depth_space)
        self.k_space = int(k_space)

        # Add grid
        self.concat_grid = add_grid(grid_range=self.grid_range)
        self.basis_cache = basis_cache

        # Lifting
        lift_cfg = lift_cfg or {}
        self.lift = LocalLift1D(
            in_channel=self.in_channel + 1,  # +1 for grid
            lifting_dim=self.lifting_dim,
            kernel_size=lift_cfg.get("kernel_size", 1),
            dilation=lift_cfg.get("dilation", 1),
            norm=lift_cfg.get("norm", None),
        )

        self.layers = nn.ModuleList([
            STULayer(
                lifting_dim=self.lifting_dim,
                k=self.k_space,
                ) for _ in range(self.depth_space)
        ])
        
        # Projection to output channels
        self.head = nn.Conv1d(self.lifting_dim, self.out_dim, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 3, f"Expected 3D input, got {x.shape}"
        batch_size, in_channel, grid_size = x.shape
        assert in_channel == self.in_channel, f"in_channel mismatch: {in_channel} vs {self.in_channel}"

        # Build Phi_f for THIS grid_size (train-time grid or test-time grid)
        Phi_f = build_stu_basis_rfft(
            grid_size=grid_size,
            k=self.k_space,
            basis="hilbert",
            cache_dir=self.basis_cache,   # store this on self in __init__
        ).to(x.device)

        h = self.concat_grid(x)
        h = self.lift(h)
        for layer in self.layers:
            h = layer(h, Phi_f)
        y = self.head(h)
        return y


# =========================================================
# 5) Wrapper with SVD1d-like API
# =========================================================

class STU1d(nn.Module):
    """
    Wrapper that mimics the SVD1d constructor so it can be used
    with the existing build_and_train_model() pipeline.

    Parameters match SVD1d.__init__ (extra ones are mostly ignored):

      in_channel    : input channel dimension (matches in_dim in DS code)
      num_eigenfunc : mapped to k_space (number of STU modes)
      H             : ignored (for use in SVD setting)
      N_H           : ignored (number of basis functions in SVD setting)
      lifting_dim   : latent width d
      grid_range    : (not used inside STU; kept for logging)
      grid_size     : spatial grid length n
      kernel_layers : number of stacked STU layers
      out_dim       : output channel dimension
      mlp_type      : for lifting kind ("linear" or "mlp")
      device        : device string, just stored; actual .to(device) is done outside
    """
    def __init__(
        self,
        in_channel: int,
        num_eigenfunc: int,
        H: int = 8,
        N_H: int = 1,
        lifting_dim: int = 64,
        grid_range: float = 1.0,
        grid_size: int = 1024,
        kernel_layers: int = 4,
        out_dim: int = 0,
        mlp_type="linear",
        device: str = "cuda",
    ):
        super().__init__()
        self.in_channel = int(in_channel)
        self.out_dim = int(out_dim) if out_dim > 0 else 1
        self.grid_size = int(grid_size)
        self.num_eigenfunc = int(num_eigenfunc)
        self.lifting_dim = int(lifting_dim)
        self.grid_range = float(grid_range)

        # Reasonable defaults tying SVD hyper-params to STU:
        k_space = num_eigenfunc      # number of spatial modes

        basis_cache = os.path.join(
            os.path.dirname(__file__),
            "basis_cache"
        )

        self.core = StackedSTU1D(
            grid_size=self.grid_size,
            grid_range=self.grid_range,
            in_channel=self.in_channel,
            out_dim=self.out_dim,
            lifting_dim=lifting_dim,
            depth_space=kernel_layers,
            k_space=k_space,
            basis_cache=basis_cache,
            lift_cfg={
                "kind": mlp_type,
                "kernel_size": 1,
                "hidden": [lifting_dim, lifting_dim],
                "norm": "group",
            },
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x comes from the pipeline as:
            (batch, grid_size, in_dim == in_channel)

        We convert to (batch_size, in_channel=1, grid_size=n), apply STU, and convert back:
            out: (batch_size, grid_size, out_dim)
        """
        assert x.ndim == 3, f"Expected (batch_size, grid_size, in_dim), got {x.shape}"
        batch_size, grid_size, in_channel = x.shape
        assert in_channel == self.in_channel, f"in_channel mismatch: {in_channel} vs {self.in_channel}"

        x_perm = x.permute(0, 2, 1).contiguous()  # (batch_size, in_channel = 1, n = grid_size)
        y = self.core(x_perm)                     # (batch_size, out_dim, n)
        y = y.permute(0, 2, 1).contiguous()       # (batch_size, grid_size, out_dim)
        return y
