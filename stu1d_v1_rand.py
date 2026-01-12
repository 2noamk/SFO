from __future__ import annotations

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

# @torch.no_grad()
# def build_stu_basis_rfft(
#     grid_size: int,
#     k: int,
#     basis: str = "random",
#     cache_dir: str | Path = "./basis_cache",
# ) -> torch.Tensor:
#     """
#     Build top-l basis functions in frequency domain.
#     Options:
#         - 'hilbert': Hilbert eigenvectors (original)
#         - 'random' : random orthonormal basis via QR
#     """
#     L = int(grid_size)
#     assert 1 <= k <= L
#     basis = basis.lower()

#     cache_dir = Path(cache_dir)
#     cache_dir.mkdir(parents=True, exist_ok=True)
#     cache_path = cache_dir / f"{basis}_rfft_L{L}.pt"

#     # ---------------------------
#     # 1. Load from cache if exists
#     # ---------------------------
#     if cache_path.exists():
#         payload = torch.load(cache_path, map_location="cpu")
#         V_f = payload["eigvecs_rfft"]
#         return V_f[:k].to(torch.complex64)

#     # ---------------------------
#     # 2. Build basis
#     # ---------------------------
#     if basis == "hilbert":
#         # Original behavior
#         Z = torch.tensor(hilbert(L), dtype=torch.float64)
#         w, V = torch.linalg.eigh(Z)
#         V_t = V.T

#     elif basis == "random":
#         # Random Gaussian matrix → QR → orthonormal rows
#         R = torch.randn(L, L, dtype=torch.float64)  # random matrix
#         Q, _ = torch.linalg.qr(R)                    # Q is orthonormal
#         V_t = Q                                      # same shape as before

#     else:
#         raise ValueError(f"Unknown basis type: {basis}")

#     # ---------------------------
#     # 3. Convert to frequency domain
#     # ---------------------------
#     V_f = torch.fft.rfft(V_t, n=L, dim=-1)  # (L, L//2+1)

#     payload = {"eigvecs_rfft": V_f.cpu()}
#     torch.save(payload, cache_path)

#     # Return only top-k rows
#     return V_f[:k].to(torch.complex64)


# If you use SciPy's hilbert() today, keep that import as-is.
# from scipy.linalg import hilbert


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
def build_stu_basis_rfft(
    grid_size: int,
    k: int,
    basis: str = "wavelet",
    cache_dir: str | Path = "./basis_cache",
) -> torch.Tensor:
    """
    Build top-k basis functions in frequency domain (rFFT).

    Options:
        - 'hilbert'   : Hilbert eigenvectors (original)
        - 'random'    : random orthonormal basis via QR
        - 'fourier'   : real orthonormal Fourier basis (cos/sin)
        - 'chebyshev' : Chebyshev T_m sampled on [-1,1], then QR-orthonormalized
        - 'haar'      : Haar wavelet basis (requires grid_size power-of-2)

    Returns:
        V_f[:k] as complex64, shape (k, L//2+1)
    """
    L = int(grid_size)
    assert 1 <= k <= L
    basis = basis.lower()

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{basis}_rfft_L{L}.pt"

    # 1) Load cache
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        V_f = payload["eigvecs_rfft"]
        return V_f[:k].to(torch.complex64)

    # 2) Build basis in time domain, rows are basis vectors length L
    if basis == "hilbert":
        print("Using Hilbert basis for STU1D")
        # Keep your current method; just note: you may want to sort eigenvectors by |eigenvalue| descending.
        from scipy.linalg import hilbert
        Z = torch.tensor(hilbert(L), dtype=torch.float64)
        w, V = torch.linalg.eigh(Z)   # ascending eigenvalues
        V_t = V.T                     # rows are eigenvectors

    elif basis == "random":
        print("Using Random basis for STU1D")
        R = torch.randn(L, L, dtype=torch.float64)
        Q, _ = torch.linalg.qr(R)     # columns orthonormal
        V_t = Q.T                     # make rows orthonormal

    elif basis == "fourier":
        print("Using Fourier basis for STU1D")
        V_t = _build_fourier_real_basis(L, dtype=torch.float64)

    elif basis == "chebyshev":
        print("Using Chebyshev basis for STU1D")
        V_t = _build_chebyshev_basis(L, dtype=torch.float64)

    elif basis in ("wavelet", "haar"):
        print("Using Haar wavelet basis for STU1D")
        V_t = _build_haar_wavelet_basis(L, dtype=torch.float64)

    else:
        raise ValueError(f"Unknown basis type: {basis}")

    # 3) Convert to frequency domain
    V_f = torch.fft.rfft(V_t, n=L, dim=-1)  # (L, L//2+1), complex128

    # Save cache
    payload = {"eigvecs_rfft": V_f.cpu()}
    torch.save(payload, cache_path)

    return V_f[:k].to(torch.complex64)

# =========================================================
# 2) Core STU spatial operator
# =========================================================
class STULayer(nn.Module):
    def __init__(self, lifting_dim: int, grid_size: int, Phi_f: torch.Tensor):
        super().__init__()
        num_hilbert_functions, grid_size_r = Phi_f.shape

        self.lifting_dim = lifting_dim
        self.grid_size = grid_size
        self.num_hilbert_functions = num_hilbert_functions
        self.grid_size_r = grid_size_r

        self.register_buffer("Phi_f", Phi_f.clone())

        self.Theta = nn.Parameter(
            torch.empty(num_hilbert_functions, lifting_dim, lifting_dim)
        )
        nn.init.kaiming_uniform_(self.Theta, a=math.sqrt(5))

        self.norm = nn.LayerNorm(lifting_dim)

        # ---- Flash STU MLP block ----
        hidden_dim = 2 * lifting_dim  # expansion ratio like Flash-STU
        self.mlp_fc1 = nn.Conv1d(lifting_dim, hidden_dim, kernel_size=1)
        self.mlp_fc2 = nn.Conv1d(hidden_dim, lifting_dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, lifting_dim, grid_size = x.shape
        assert lifting_dim == self.lifting_dim
        assert grid_size == self.grid_size

        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        z = self.norm(x.transpose(1, 2)).transpose(1, 2)

        Xf = torch.fft.rfft(z, dim=-1)
        Uf = Xf.unsqueeze(2) * self.Phi_f.conj().unsqueeze(0).unsqueeze(0)
        U = torch.fft.irfft(Uf, n=self.grid_size, dim=-1)

        U_mix = torch.einsum("bdln,lhd->bhln", U, self.Theta)
        S = U_mix.sum(dim=2)  # (batch, lifting_dim, grid_size)

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
            # basis="random",
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
