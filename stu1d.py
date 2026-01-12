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
    L = int(grid_size)
    assert 1 <= k <= L
    basis = basis.lower()
    assert basis == "hilbert"

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{basis}_rfft_L{L}.pt"

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        w = payload["eigvals"]
        V_f = payload["eigvecs_rfft"]
        grid_size_r = L // 2 + 1
        assert isinstance(w, torch.Tensor) and w.ndim == 1 and w.numel() == L
        assert isinstance(V_f, torch.Tensor) and V_f.shape == (L, grid_size_r)
    else:
        Z = torch.tensor(hilbert(L), dtype=torch.float64)
        w, V = torch.linalg.eigh(Z)           # ascending λ
        V_t = V.T                             # (L, L)
        V_f = torch.fft.rfft(V_t, n=L, dim=-1)  # (L, grid_size_r)
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
    """
    Single STU layer on a 1D ring (circular conv via FFT).

    Input / Output:
      x: (batch_size, lifting_dim, T)
    """
    def __init__(self, lifting_dim: int, grid_size: int, Phi_f: torch.Tensor):
        super().__init__()
        assert Phi_f.ndim == 2
        num_hilbert_functions, grid_size_r = Phi_f.shape

        self.lifting_dim = int(lifting_dim)
        self.grid_size = int(grid_size)
        self.num_hilbert_functions = int(num_hilbert_functions)
        self.grid_size_r = int(grid_size_r)

        # complex basis in frequency domain
        self.register_buffer("Phi_f", Phi_f.clone())

        # Theta
        self.Theta = nn.Parameter(torch.empty(self.num_hilbert_functions, self.lifting_dim, self.lifting_dim))
        nn.init.kaiming_uniform_(self.Theta, a=math.sqrt(5))

        # gated linear unit on the aggregated signal
        self.glu_val = nn.Conv1d(self.lifting_dim, self.lifting_dim, kernel_size=1)
        self.glu_gate = nn.Conv1d(self.lifting_dim, self.lifting_dim, kernel_size=1)

        # simple per-position normalization
        self.norm = nn.LayerNorm(self.lifting_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, lifting_dim, grid_size)
        """
        batch_size, lifting_dim, grid_size = x.shape
        assert lifting_dim == self.lifting_dim and grid_size == self.grid_size

        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # LayerNorm over channels, keep shape (batch_size, lifting_dim, grid_size)
        z = self.norm(x.transpose(1, 2)).transpose(1, 2)

        # RFFT over time dimension
        Xf = torch.fft.rfft(z, dim=-1)  # (batch_size, lifting_dim, grid_size_r)

        # Project onto l = num_hilbert_functions modes with Phi_f:
        #   Xf (batch_size, lifting_dim = d, 1, grid_size_r) * Phi_f^* (1, 1, num_hilbert_functions = l, grid_size_r)
        Uf = Xf.unsqueeze(2) * self.Phi_f.conj().unsqueeze(0).unsqueeze(0)  # (batch_size, lifting_dim, num_hilbert_functions, grid_size_r)

        # Back to time domain (circular)
        U = torch.fft.irfft(Uf, n=self.grid_size, dim=-1)  # (batch_size, lifting_dim, num_hilbert_functions, grid_size)

        # b d l n , l h d -> b h l n
        U_mix = torch.einsum("bdln,lhd->bhln", U, self.Theta)

        # Aggregate over modes
        S = U_mix.sum(dim=2)  # (batch_size, lifting_dim, grid_size)

        # GLU
        V = self.glu_val(S)
        G = torch.sigmoid(self.glu_gate(S))
        out = V * G
        

        # Residual
        return x + out  # (batch_size, lifting_dim, grid_size)


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

        # Lifting
        lift_cfg = lift_cfg or {}
        self.lift = LocalLift1D(
            in_channel=self.in_channel + 1,  # +1 for grid
            lifting_dim=self.lifting_dim,
            kernel_size=lift_cfg.get("kernel_size", 1),
            dilation=lift_cfg.get("dilation", 1),
            norm=lift_cfg.get("norm", None),
        )

        # STU layers over spatial dimension
        Phi = build_stu_basis_rfft(
            grid_size=self.grid_size,
            k=self.k_space,
            basis="hilbert",
            cache_dir=basis_cache,
        )
        self.layers = nn.ModuleList([
            STULayer(
                lifting_dim=self.lifting_dim, 
                grid_size=self.grid_size, 
                Phi_f=Phi) for _ in range(self.depth_space)
        ])

        # Projection to output channels
        self.head = nn.Conv1d(self.lifting_dim, self.out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Accept either:
          - x: (batch_size, in_channel, grid_size)
        """
        assert x.dim() == 3, f"Expected 3D input, got {x.shape}"
        batch_size, in_channel, grid_size = x.shape
        assert in_channel == self.in_channel and grid_size == self.grid_size, f"Expected (batch_size,{self.in_channel},{self.grid_size}), got {x.shape}"

        h = self.concat_grid(x) # (batch_size, in_channel + 1, grid_size)
        h = self.lift(h) # (batch_size, lifting_dim, grid_size)
        for layer in self.layers:   
            h = layer(h) # (batch_size, lifting_dim, grid_size)
        y = self.head(h) # (batch_size, out_dim, grid_size)
        return y  # (batch_size, out_dim, grid_size)


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
        assert grid_size == self.grid_size, f"grid_size mismatch: {grid_size} vs {self.grid_size}"
        assert in_channel == self.in_channel, f"in_channel mismatch: {in_channel} vs {self.in_channel}"

        x_perm = x.permute(0, 2, 1).contiguous()  # (batch_size, in_channel = 1, n = grid_size)
        y = self.core(x_perm)                     # (batch_size, out_dim, n)
        y = y.permute(0, 2, 1).contiguous()       # (batch_size, grid_size, out_dim)
        return y
