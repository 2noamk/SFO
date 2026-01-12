from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import hilbert


# =========================================================
# 1) STU basis in frequency domain
#    (reuse the 1D Hilbert basis and build a separable 2D basis)
# =========================================================

def _is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def _orthonormalize_rows(V: torch.Tensor) -> torch.Tensor:
    """
    Given V shape (L, L), returns row-orthonormal matrix (L, L).
    We do QR on V^T so columns become orthonormal, then transpose back.
    """
    Q, _ = torch.linalg.qr(V.T)   # Q: (L, L), columns orthonormal
    return Q.T


def _build_fourier_real_basis(L: int, dtype=torch.float64) -> torch.Tensor:
    """
    Real orthonormal Fourier basis on a length-L grid:
      v0 = constant
      then cos(2π m n/L), sin(2π m n/L) pairs
      (and Nyquist cosine when L even)
    Returns V shape (L, L) with orthonormal rows (up to numerical precision).
    """
    n = torch.arange(L, dtype=dtype)
    V = torch.zeros((L, L), dtype=dtype)

    # Row 0: constant
    V[0, :] = 1.0 / math.sqrt(L)

    r = 1  # next row index
    max_m = L // 2

    for m in range(1, max_m + 1):
        # Nyquist frequency (only cosine) when L even and m == L/2
        if (L % 2 == 0) and (m == L // 2):
            V[r, :] = math.sqrt(2.0 / L) * torch.cos(2 * math.pi * m * n / L)
            r += 1
        else:
            V[r, :] = math.sqrt(2.0 / L) * torch.cos(2 * math.pi * m * n / L)
            r += 1
            if r < L:
                V[r, :] = math.sqrt(2.0 / L) * torch.sin(2 * math.pi * m * n / L)
                r += 1
        if r >= L:
            break

    # In rare cases due to loop structure, ensure we filled all rows
    # (should already be exact for all L).
    return V


def _build_chebyshev_basis(L: int, dtype=torch.float64) -> torch.Tensor:
    """
    Chebyshev T_m basis sampled on x in [-1, 1].
    We then QR-orthonormalize rows for stability.
    """
    # Chebyshev nodes often use cos(pi*(i+0.5)/L), but you likely want uniform grid points.
    # We'll use uniform grid; then orthonormalize to avoid ill-conditioning.
    x = torch.linspace(-1.0, 1.0, L, dtype=dtype)
    theta = torch.acos(torch.clamp(x, -1.0, 1.0))  # acos needs clamp

    V = torch.empty((L, L), dtype=dtype)
    for m in range(L):
        V[m, :] = torch.cos(m * theta)  # T_m(x)

    V = _orthonormalize_rows(V)
    return V


def _build_haar_wavelet_basis(L: int, dtype=torch.float64) -> torch.Tensor:
    """
    Orthonormal Haar wavelet basis (rows) for length L.
    Requires L to be a power of 2.

    Constructs:
      - one scaling function (constant)
      - wavelets at scales 1,2,4,...,L/2 with appropriate normalization
    """
    if not _is_power_of_two(L):
        raise ValueError(f"Haar wavelet basis requires grid_size power-of-2, got L={L}.")

    V = torch.zeros((L, L), dtype=dtype)

    # Scaling (constant) row
    V[0, :] = 1.0 / math.sqrt(L)

    r = 1
    # scale s is the half-support length of the wavelet block
    s = 1
    while s < L and r < L:
        block = 2 * s
        # number of wavelets at this scale
        num = L // block
        # amplitude so each wavelet has L2 norm 1
        amp = 1.0 / math.sqrt(block)

        for j in range(num):
            start = j * block
            mid = start + s
            end = start + block
            V[r, start:mid] = amp
            V[r, mid:end] = -amp
            r += 1
            if r >= L:
                break
        s *= 2

    return V


@torch.no_grad()
def build_stu_basis_rfft_2d(
    grid_size_x: int,
    grid_size_y: int,
    num_eigenfunc: int,
    basis: str = "chebyshev",   # hilbert | random | fourier | chebyshev | haar
    cache_dir: str | Path = "./basis_cache",
) -> torch.Tensor:
    """
    Build separable 2D STU basis in the frequency domain using top-k modes.

    Returns
    -------
    Phi_f : (k, grid_size_x, grid_size_y//2 + 1) complex64
    """
    grid_size_x = int(grid_size_x)
    grid_size_y = int(grid_size_y)
    k = int(num_eigenfunc)
    assert 1 <= k <= min(grid_size_x, grid_size_y)

    basis = basis.lower()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    cache_path = cache_dir / (
        f"{basis}_rfft2d_x{grid_size_x}_y{grid_size_y}_k{k}.pt"
    )

    # ---------------------------
    # 1) Load cache
    # ---------------------------
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        return payload["Phi_f"].to(torch.complex64)

    # ---------------------------
    # 2) Build 1D bases (rows orthonormal)
    # ---------------------------
    def build_1d_basis(L: int) -> torch.Tensor:
        if basis == "hilbert":
            from scipy.linalg import hilbert
            Z = torch.tensor(hilbert(L), dtype=torch.float64)
            w, V = torch.linalg.eigh(Z)
            idx = torch.argsort(w, descending=True)
            return V.T[idx]  # rows

        elif basis == "random":
            R = torch.randn(L, L, dtype=torch.float64)
            Q, _ = torch.linalg.qr(R)
            return Q.T

        elif basis == "fourier":
            return _build_fourier_real_basis(L, dtype=torch.float64)

        elif basis == "chebyshev":
            return _build_chebyshev_basis(L, dtype=torch.float64)

        elif basis in ("wavelet", "haar"):
            return _build_haar_wavelet_basis(L, dtype=torch.float64)

        else:
            raise ValueError(f"Unknown basis: {basis}")

    Vx = build_1d_basis(grid_size_x)[:k]   # (k, grid_size_x)
    Vy = build_1d_basis(grid_size_y)[:k]   # (k, grid_size_y)

    # ---------------------------
    # 3) Build separable 2D modes + rFFT
    # ---------------------------
    grid_size_y_r = grid_size_y // 2 + 1
    Phi_f = torch.empty(
        (k, grid_size_x, grid_size_y_r), dtype=torch.complex64
    )

    for i in range(k):
        phi_2d = torch.outer(Vx[i], Vy[i])  # (Nx, Ny)
        Phi_f[i] = torch.fft.rfft2(
            phi_2d, s=(grid_size_x, grid_size_y)
        ).to(torch.complex64)

    # ---------------------------
    # 4) Cache
    # ---------------------------
    payload = {
        "Phi_f": Phi_f.cpu(),
        "grid_size_x": grid_size_x,
        "grid_size_y": grid_size_y,
        "k": k,
        "basis": basis,
    }
    torch.save(payload, cache_path)

    return Phi_f

# =========================================================
# 2) Core STU spatial operator (2D)
# =========================================================

class STULayer2D(nn.Module):
    """
    Single STU layer on a 2D torus (circular conv via 2D FFT).

    Input / Output:
      x: (batch_size, lifting_dim, grid_size_x, grid_size_y)
    """
    def __init__(self, lifting_dim: int, grid_size_x: int, grid_size_y: int, Phi_f: torch.Tensor):
        super().__init__()
        assert Phi_f.ndim == 3, "Phi_f must be (num_hilbert_functions, grid_size_x, grid_size_y_r)"
        num_hilbert_functions, grid_size_x_r, grid_size_y_r = Phi_f.shape
        assert grid_size_x_r == grid_size_x, f"grid_size_x mismatch: {grid_size_x_r} vs {grid_size_x}"

        self.lifting_dim = int(lifting_dim)
        self.grid_size_x = int(grid_size_x)
        self.grid_size_y = int(grid_size_y)
        self.num_hilbert_functions = int(num_hilbert_functions)
        self.grid_size_y_r = int(grid_size_y_r)

        # complex basis in frequency domain
        self.register_buffer("Phi_f", Phi_f.clone())

        # learnable mixing per mode
        self.Theta = nn.Parameter(torch.empty(self.num_hilbert_functions, self.lifting_dim, self.lifting_dim))
        nn.init.kaiming_uniform_(self.Theta, a=math.sqrt(5))

        # simple per-position normalization over channels
        self.norm = nn.LayerNorm(self.lifting_dim)
        
        # ---- Flash STU MLP block ----
        hidden_dim = 2 * lifting_dim  # expansion ratio like Flash-STU
        self.mlp_fc1 = nn.Conv2d(lifting_dim, hidden_dim, kernel_size=1)
        self.mlp_fc2 = nn.Conv2d(hidden_dim, lifting_dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, lifting_dim, grid_size_x, grid_size_y)
        """
        batch_size, lifting_dim, grid_size_x, grid_size_y = x.shape
        assert lifting_dim == self.lifting_dim and grid_size_x == self.grid_size_x and grid_size_y == self.grid_size_y

        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # LayerNorm over channels, keep shape (batch_size, lifting_dim, grid_size_x, grid_size_y)
        z = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)  # (batch_size, lifting_dim, grid_size_x, grid_size_y)

        # 2D RFFT over spatial dimensions
        Xf = torch.fft.rfft2(z, dim=(-2, -1))  # (batch_size, lifting_dim, grid_size_x, grid_size_y_r) What is grid_size_y_r, why not grid_size_x_r also?

        # Project onto num_hilbert_functions modes with Phi_f:
        #   Xf (batch_size, lifting_dim, 1, grid_size_x, grid_size_y_r) * Phi_f^* (1, 1, num_hilbert_functions, grid_size_x, grid_size_y_r)
        Uf = Xf.unsqueeze(2) * self.Phi_f.conj().unsqueeze(0).unsqueeze(0)  # (batch_size, lifting_dim, num_hilbert_functions, grid_size_x, grid_size_y_r) what exactly does this do?

        # Back to spatial domain (circular) via inverse rfft2
        U = torch.fft.irfft2(Uf, s=(self.grid_size_x, self.grid_size_y), dim=(-2, -1))  # (batch_size, lifting_dim, num_hilbert_functions, grid_size_x, grid_size_y)

        # b d l x y , l h d -> b h l x y
        U_mix = torch.einsum("bdlxy,lhd->bhlxy", U, self.Theta) # (batch_size, lifting_dim, num_hilbert_functions, grid_size_x, grid_size_y)
    
        # Aggregate over modes
        S = U_mix.sum(dim=2)  # (batch_size, lifting_dim, grid_size_x, grid_size_y)
        
        # ---- Flash-STU MLP ----
        H = self.mlp_fc1(S)
        H = self.act(H)
        out = self.mlp_fc2(H)

        # Residual
        return x + out


# =========================================================
# 3) Local lifting + positional embedding (2D)
# =========================================================

def _make_activation(name: Optional[str]) -> nn.Module:
    if name is None or name.lower() == "gelu":
        return nn.GELU()
    name = name.lower()
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "identity":
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


class LocalLift2D(nn.Module):
    """
    Local lifting from (batch_size, in_channels, grid_size_x, grid_size_y) → (batch_size, out_channels, grid_size_x, grid_size_y).

    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        dilation: int = 1,
        hidden: Optional[List[int]] = None,
        act: str = "gelu",
        dropout: float = 0.0,
        norm: Optional[str] = None,  # "layer" | "group" | None
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = int(kernel_size)
        self.dilation = int(dilation)

        padding = dilation * (kernel_size - 1) // 2
        self.net = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
            padding_mode="circular",
            bias=True,
        )
            
        if norm is None:
            self.norm = nn.Identity()
        elif norm.lower() == "layer":
            # LayerNorm over channels -> GroupNorm with 1 group
            self.norm = nn.GroupNorm(num_groups=1, num_channels=out_channels, affine=True)
        elif norm.lower() == "group":
            # choose a divisor of out_channels
            num_groups = 8
            while out_channels % num_groups != 0 and num_groups > 1:
                num_groups //= 2
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=out_channels, affine=True)
        else:
            raise ValueError(f"Unknown norm: {norm}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.net(x)
        return self.norm(y)



class add_2d_grid(nn.Module):
    """
    Add 2D grid
    """
    def __init__(self, grid_range: float = 1.0):
        super().__init__()
        self.grid_range = grid_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, in_channelsannel, grid_size_x, grid_size_y)
        batch_size, in_channelsannel, grid_size_x, grid_size_y = x.shape
        device, dtype = x.device, x.dtype
        ys = torch.linspace(0, self.grid_range, grid_size_x, device=device, dtype=dtype)
        xs = torch.linspace(0, self.grid_range, grid_size_y, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")  # (grid_size_x, grid_size_y)
        yy = yy.expand(batch_size, 1, grid_size_x, grid_size_y)
        xx = xx.expand(batch_size, 1, grid_size_x, grid_size_y)
        return torch.cat([x, yy, xx], dim=1)

# =========================================================
# 4) Stacked spatial-only STU for 2D
# =========================================================


class StackedSTU2D(nn.Module):
    """
    Spatial-only STU stack for 2D inputs.

    I/O:
      forward(a): a (batch_size, in_channels, grid_size_x, grid_size_y) -> (batch_size, out_channels, grid_size_x, grid_size_y)
    """
    def __init__(
        self,
        grid_size_x: int,
        grid_size_y: int,
        grid_range: float,
        in_channels: int,
        out_dim: int,
        *,
        lifting_dim: int = 64,
        depth_space: int = 4,
        num_eigenfunc: int = 8,
        basis_cache: str | Path = "./basis_cache",
        lift_cfg: Optional[dict] = None,
    ):
        super().__init__()
        self.grid_size_x = int(grid_size_x)
        self.grid_size_y = int(grid_size_y)
        self.grid_range = float(grid_range)
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        self.lifting_dim = int(lifting_dim)
        self.depth_space = int(depth_space)
        self.num_eigenfunc = int(num_eigenfunc)

        # Positional embedding
        self.concat_grid = add_2d_grid(grid_range=self.grid_range)

        # Lifting
        lift_cfg = lift_cfg or {}
        self.lift = LocalLift2D(
            in_channels=self.in_channels + 1 + 1,  # +2 for 2D grid
            out_channels=self.lifting_dim,
            kernel_size=lift_cfg.get("kernel_size", 1),
            dilation=lift_cfg.get("dilation", 1),
            hidden=lift_cfg.get("hidden"),
            act=lift_cfg.get("act", "gelu"),
            dropout=lift_cfg.get("dropout", 0.0),
            norm=lift_cfg.get("norm", None),
        )

        # STU basis in 2D frequency domain.
        Phi = build_stu_basis_rfft_2d(
            grid_size_x=self.grid_size_x,
            grid_size_y=self.grid_size_y,
            num_eigenfunc=num_eigenfunc,
            basis="random",
            cache_dir=basis_cache,
        )
        
        self.layers = nn.ModuleList([
            STULayer2D(lifting_dim=self.lifting_dim, grid_size_x=self.grid_size_x, grid_size_y=self.grid_size_y, Phi_f=Phi)
            for _ in range(self.depth_space)
        ])

        # Projection to output channels
        self.head = nn.Conv2d(self.lifting_dim, self.out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
          - x: (batch_size, in_channels, grid_size_x, grid_size_y)
        """
        assert x.dim() == 4, f"Expected 4D input, got {x.shape}"
        batch_size, lifting_dim, grid_size_x, grid_size_y = x.shape
        assert lifting_dim == self.in_channels and grid_size_x == self.grid_size_x and grid_size_y == self.grid_size_y, \
            f"Expected (batch_size,{self.in_channels},{self.grid_size_x},{self.grid_size_y}), got {x.shape}"

        h = self.concat_grid(x) # (batch_size, in_channels + 2, grid_size_x, grid_size_y)
        h = self.lift(h)    # (batch_size, lifting_dim, grid_size_x, grid_size_y)
        for layer in self.layers:
            h = layer(h)
        y = self.head(h)
        return y  # (batch_size, out_channels, grid_size_x, grid_size_y)


# =========================================================
# 5) Wrapper with FNO2d-like API
# =========================================================


class STU2d(nn.Module):
    """
    Wrapper that mimics the FNO2d constructor so it can be used
    as a drop-in replacement in PDEBench-style code.

    Parameters
    ----------
    in_channels : int
        Input channel dimension (e.g., history steps + coordinates).
    out_dim: int
        Output channel dimension.
    modes1, modes2 : int
        Not used directly as Fourier modes here; we only use
        modes1 * modes2 as a proxy for k_space (number of STU modes).
    width : int
        Latent channel dimension lifting_dim.
    """
    def __init__(
        self,
        in_channel: int,
        num_eigenfunc: int,
        H: int,
        N_H: int,
        lifting_dim: int,
        grid_range: float,
        grid_size: int,
        kernel_layers: int = 4,
        out_dim: int = 0,
        mlp_type=None,
        device: str = "cuda",
    ):
        super().__init__()
        self.in_channels = int(in_channel)
        self.out_dim = int(out_dim)
        self.num_eigenfunc = int(num_eigenfunc)
        self.lifting_dim = int(lifting_dim)

        # grid_size_x, grid_size_y may not be known at construction time, so we lazily
        # initialize the core on the first forward call.
        self.grid_size_x, self.grid_size_y = grid_size
        
        # k_space = max(1, self.num_eigenfunc * self.num_eigenfunc)
        
        basis_cache = os.path.join(
            os.path.dirname(__file__),
            "basis_cache"
        )
        
        self.core = StackedSTU2D(
            grid_size_x=self.grid_size_x,
            grid_size_y=self.grid_size_y,
            grid_range=grid_range,
            in_channels=self.in_channels,
            out_dim=self.out_dim,
            lifting_dim=self.lifting_dim,
            depth_space=kernel_layers,
            num_eigenfunc=num_eigenfunc, 
            basis_cache=basis_cache,
            lift_cfg={
                "kernel_size": 5,
                "dilation": 1,
                "hidden": [self.lifting_dim, self.lifting_dim],
                "act": "gelu",
                "dropout": 0.0,
                "norm": "group",
            },
        )

    def forward(self, x: torch.Tensor, grid: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch_size, grid_size_x, grid_size_y, in_channels)

        Returns
        -------
        y: (batch_size, grid_size_x, grid_size_y, out_channels)
        """
        assert x.dim() == 4, f"Expected (batch_size,grid_size_x,grid_size_y,in_channels), got {x.shape}"
        batch_size, grid_size_x, grid_size_y, in_channels = x.shape
        assert in_channels == self.in_channels, f"in_channels mismatch: {in_channels} vs {self.in_channels}"
        
        x_ch_first = x.permute(0, 3, 1, 2).contiguous() # (batch_size, in_channels, grid_size_x, grid_size_y)
        y = self.core(x_ch_first)  # (batch_size, out_channels, grid_size_x, grid_size_y)
        return y.permute(0, 2, 3, 1).contiguous() # (batch_size, grid_size_x, grid_size_y, out_channels)

