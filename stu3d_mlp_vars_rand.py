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
def build_stu_basis_rfft_3d(
    grid_size_x: int,
    grid_size_y: int,
    grid_size_z: int,
    num_eigenfunc: int,
    basis: str = "fourier",   # hilbert | random | fourier | chebyshev | haar
    cache_dir: str | Path = "./basis_cache",
) -> torch.Tensor:
    """
    Build a separable (tied-index) 3D STU basis on an Nx×Ny×Nz grid in the frequency domain.

    Returns
    -------
    Phi_f : (k, Nx, Ny, Nz//2 + 1) complex64
    """
    Nx, Ny, Nz = int(grid_size_x), int(grid_size_y), int(grid_size_z)
    k = int(num_eigenfunc)
    assert Nx > 0 and Ny > 0 and Nz > 0
    assert 1 <= k <= min(Nx, Ny, Nz)

    basis = basis.lower()
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{basis}_rfft3d_x{Nx}_y{Ny}_z{Nz}_k{k}.pt"

    # ---------------------------
    # 1) Cache
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

    Vx = build_1d_basis(Nx)[:k]  # (k, Nx)
    Vy = build_1d_basis(Ny)[:k]  # (k, Ny)
    Vz = build_1d_basis(Nz)[:k]  # (k, Nz)

    # ---------------------------
    # 3) Build separable 3D modes + rFFT
    # ---------------------------
    Nz_r = Nz // 2 + 1
    Phi_f = torch.empty((k, Nx, Ny, Nz_r), dtype=torch.complex64)

    for i in range(k):
        # (Nx, Ny, Nz)
        phi_3d = torch.einsum("i,j,k->ijk", Vx[i], Vy[i], Vz[i])
        Phi_f[i] = torch.fft.rfftn(phi_3d, s=(Nx, Ny, Nz)).to(torch.complex64)

    # ---------------------------
    # 4) Cache
    # ---------------------------
    payload = {
        "Phi_f": Phi_f.cpu(),
        "grid_size_x": Nx,
        "grid_size_y": Ny,
        "grid_size_z": Nz,
        "k": k,
        "basis": basis,
    }
    torch.save(payload, cache_path)

    return Phi_f

# =========================================================
# 2) Core STU spatial operator (2D)
# =========================================================

class STULayer3D(nn.Module):
    """
    Single STU layer on a 3D torus (circular conv via 3D FFT).
    """
    def __init__(self, lifting_dim: int, grid_size_x: int, grid_size_y: int, grid_size_z: int, Phi_f: torch.Tensor):
        super().__init__()
        assert Phi_f.ndim == 4, "Phi_f must be (num_hilbert_functions, grid_size_x, grid_size_y, grid_size_z_r)"
        num_hilbert_functions, grid_size_x_r, grid_size_y_r, grid_size_z_r = Phi_f.shape
        assert grid_size_x_r == grid_size_x, f"grid_size_x mismatch: {grid_size_x_r} vs {grid_size_x}"

        self.lifting_dim = int(lifting_dim)
        self.grid_size_x = int(grid_size_x)
        self.grid_size_y = int(grid_size_y)
        self.grid_size_z = int(grid_size_z)
        self.num_hilbert_functions = int(num_hilbert_functions)
        self.grid_size_z_r = int(grid_size_z_r)

        # complex basis in frequency domain
        self.register_buffer("Phi_f", Phi_f.clone())

        # learnable mixing per mode
        self.Theta = nn.Parameter(torch.empty(self.num_hilbert_functions, self.lifting_dim, self.lifting_dim))
        nn.init.kaiming_uniform_(self.Theta, a=math.sqrt(5))

        # simple per-position normalization over channels
        self.norm = nn.LayerNorm(self.lifting_dim)

        # ---- Flash STU MLP block ----
        hidden_dim = 2 * lifting_dim  # expansion ratio like Flash-STU
        self.mlp_fc1 = nn.Conv3d(lifting_dim, hidden_dim, kernel_size=1)
        self.mlp_fc2 = nn.Conv3d(hidden_dim, lifting_dim, kernel_size=1)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch_size, num_vars, lifting_dim, grid_size_x, grid_size_y, grid_size_z)
        """
        batch_size, num_vars, lifting_dim, grid_size_x, grid_size_y, grid_size_z = x.shape
        assert lifting_dim == self.lifting_dim and grid_size_x == self.grid_size_x and grid_size_y == self.grid_size_y and grid_size_z == self.grid_size_z

        if x.dtype != torch.float32:
            x = x.to(torch.float32)

        # LayerNorm over channels, keep shape (batch_size, num_vars, lifting_dim, grid_size_x, grid_size_y, grid_size_z)
        z = self.norm(x.permute(0, 1, 3, 4, 5, 2)).permute(0, 1, 5, 2, 3, 4)  # (batch_size, num_vars, lifting_dim, grid_size_x, grid_size_y, grid_size_z)

        # 3D RFFT over spatial dimensions
        Xf = torch.fft.rfftn(z, dim=(-3, -2, -1))  # (batch_size, num_vars, lifting_dim, grid_size_x, grid_size_y, grid_size_z_r)

        # Project onto num_hilbert_functions modes with Phi_f:
        Uf = Xf.unsqueeze(3) * self.Phi_f.conj().unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (batch_size, num_vars, lifting_dim, num_hilbert_functions, grid_size_x, grid_size_y, grid_size_z_r)

        # Back to spatial domain (circular) via inverse rfft3
        U = torch.fft.irfftn(Uf, s=(self.grid_size_x, self.grid_size_y, self.grid_size_z), dim=(-3, -2, -1))  # (batch_size, num_vars, lifting_dim,num_hilbert_functions, grid_size_x, grid_size_y, grid_size_z)

        # b v d l x y z, l h d -> b v h l x y z
        U_mix = torch.einsum("bvdlxyz,lhd->bvhlxyz", U, self.Theta)  # (batch_size, num_vars, lifting_dim, num_hilbert_functions, grid_size_x, grid_size_y, grid_size_z)

        # Aggregate over modes
        S = U_mix.sum(dim=3)  # (batch_size, num_vars, lifting_dim, grid_size_x, grid_size_y, grid_size_z)
        
        # ---- Flash-STU MLP ----
        H = self.mlp_fc1(S.reshape(-1, self.lifting_dim, self.grid_size_x, self.grid_size_y, self.grid_size_z))
        H = self.act(H)
        out = self.mlp_fc2(H).reshape(batch_size, num_vars, self.lifting_dim, self.grid_size_x, self.grid_size_y, self.grid_size_z)

        # Residual
        return x + out


# =========================================================
# 3) Local lifting + positional embedding (2D)
# =========================================================
class LocalLift3D(nn.Module):
    """
    Local lifting from (batch_size, num_vars, in_channels, grid_size_x, grid_size_y, grid_size_z) → (batch_size, num_vars, out_channels, grid_size_x, grid_size_y, grid_size_z).
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
        self.net = nn.Conv3d(
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
        batch_size, num_vars, in_channels, grid_size_x, grid_size_y, grid_size_z = x.shape  # (batch_size, num_vars, in_channels, grid_size_x, grid_size_y, grid_size_z)
        y = self.net(x.reshape(-1, in_channels, grid_size_x, grid_size_y, grid_size_z))  # Keep the variable dimension intact
        return self.norm(y).reshape(batch_size, num_vars, self.out_channels, grid_size_x, grid_size_y, grid_size_z)
class add_3d_grid(nn.Module):
    """
    Add 3D grid
    """
    def __init__(self, grid_range: float = 1.0):
        super().__init__()
        self.grid_range = grid_range

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch_size, num_vars, in_channel, grid_size_x, grid_size_y, grid_size_z)
        batch_size, num_vars, in_channels, grid_size_x, grid_size_y, grid_size_z = x.shape
        device, dtype = x.device, x.dtype

        ys = torch.linspace(0, self.grid_range, grid_size_x, device=device, dtype=dtype)
        xs = torch.linspace(0, self.grid_range, grid_size_y, device=device, dtype=dtype)
        zs = torch.linspace(0, self.grid_range, grid_size_z, device=device, dtype=dtype)

        zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")  # (grid_size_x, grid_size_y, grid_size_z)

        # Add batch and num_vars dimensions for the grid, no need for expand
        zz = zz.unsqueeze(0).unsqueeze(0).expand(batch_size, num_vars, -1, -1, -1, -1)
        yy = yy.unsqueeze(0).unsqueeze(0).expand(batch_size, num_vars, -1, -1, -1, -1)
        xx = xx.unsqueeze(0).unsqueeze(0).expand(batch_size, num_vars, -1, -1, -1, -1)

        # Concatenate the grid with the input tensor
        return torch.cat([x, zz, yy, xx], dim=2)  # (batch_size, num_vars, in_channels + 3, grid_size_x, grid_size_y, grid_size_z)

# class add_3d_grid(nn.Module):
#     """
#     Add 3D grid
#     """
#     def __init__(self, grid_range: float = 1.0):
#         super().__init__()
#         self.grid_range = grid_range

#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         # x: (batch_size, num_vars, in_channel, grid_size_x, grid_size_y, grid_size_z)
#         batch_size, num_vars, in_channels, grid_size_x, grid_size_y, grid_size_z = x.shape
#         device, dtype = x.device, x.dtype
#         ys = torch.linspace(0, self.grid_range, grid_size_x, device=device, dtype=dtype)
#         xs = torch.linspace(0, self.grid_range, grid_size_y, device=device, dtype=dtype)
#         zs = torch.linspace(0, self.grid_range, grid_size_z, device=device, dtype=dtype)
#         zz, yy, xx = torch.meshgrid(zs, ys, xs, indexing="ij")  # (grid_size_x, grid_size_y, grid_size_z)
#         yy = yy.expand(batch_size, num_vars, 1, grid_size_x, grid_size_y, grid_size_z)
#         xx = xx.expand(batch_size, num_vars, 1, grid_size_x, grid_size_y, grid_size_z)
#         zz = zz.expand(batch_size, num_vars, 1, grid_size_x, grid_size_y, grid_size_z)
#         return torch.cat([x, zz, yy, xx], dim=1)

# =========================================================
# 4) Stacked spatial-only STU for 3D
# =========================================================


class StackedSTU3D(nn.Module):
    """
    Spatial-only STU stack for 3D inputs.

    I/O:
      forward(a): a (batch_size, num_vars, in_channels, grid_size_x, grid_size_y, grid_size_z) -> (batch_size, num_vars, out_channels, grid_size_x, grid_size_y, grid_size_z)
    """
    def __init__(
        self,
        grid_size_x: int,
        grid_size_y: int,
        grid_size_z: int,
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
        self.grid_size_z = int(grid_size_z)
        self.grid_range = float(grid_range)
        self.in_channels = int(in_channels)
        self.out_dim = int(out_dim)
        self.lifting_dim = int(lifting_dim)
        self.depth_space = int(depth_space)
        self.num_eigenfunc = int(num_eigenfunc)

        # Positional embedding
        self.concat_grid = add_3d_grid(grid_range=self.grid_range)

        # Lifting
        lift_cfg = lift_cfg or {}
        self.lift = LocalLift3D(
            in_channels=self.in_channels + 1 + 1 + 1,  # +3 for 3D grid
            out_channels=self.lifting_dim,
            kernel_size=lift_cfg.get("kernel_size", 1),
            dilation=lift_cfg.get("dilation", 1),
            hidden=lift_cfg.get("hidden"),
            act=lift_cfg.get("act", "gelu"),
            dropout=lift_cfg.get("dropout", 0.0),
            norm=lift_cfg.get("norm", None),
        )

        # STU basis in 3D frequency domain.
        Phi = build_stu_basis_rfft_3d(
            grid_size_x=self.grid_size_x,
            grid_size_y=self.grid_size_y,
            grid_size_z=self.grid_size_z,
            num_eigenfunc=num_eigenfunc,
            basis="random",
            cache_dir=basis_cache,
        )
        
        self.layers = nn.ModuleList([
            STULayer3D(lifting_dim=self.lifting_dim, grid_size_x=self.grid_size_x, grid_size_y=self.grid_size_y, grid_size_z=self.grid_size_z, Phi_f=Phi)
            for _ in range(self.depth_space)
        ])

        # Projection to output channels
        self.head = nn.Conv3d(self.lifting_dim, self.out_dim, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
          - x: (batch_size, num_vars, in_channels, grid_size_x, grid_size_y, grid_size_z)
        """
        assert x.dim() == 6, f"Expected 6D input, got {x.shape}"
        batch_size, num_vars, lifting_dim, grid_size_x, grid_size_y, grid_size_z = x.shape
        assert lifting_dim == self.in_channels and grid_size_x == self.grid_size_x and grid_size_y == self.grid_size_y and grid_size_z == self.grid_size_z, \
            f"Expected (batch_size,{vars},{self.in_channels},{self.grid_size_x},{self.grid_size_y},{self.grid_size_z}), got {x.shape}"

        h = self.concat_grid(x) # (batch_size, num_vars, in_channels + 3, grid_size_x, grid_size_y, grid_size_z)
        h = self.lift(h)    # (batch_size, num_vars, lifting_dim, grid_size_x, grid_size_y, grid_size_z)
        for layer in self.layers:
            h = layer(h)
        y = self.head(h.reshape(-1, self.lifting_dim, grid_size_x, grid_size_y, grid_size_z)) # (batch_size * num_vars, out_dim, grid_size_x, grid_size_y, grid_size_z)
        return y.reshape(batch_size, num_vars, self.out_dim, grid_size_x, grid_size_y, grid_size_z)  # (batch_size, num_vars, out_channels, grid_size_x, grid_size_y, grid_size_z)


# =========================================================
# 5) Wrapper with FNO3d-like API
# =========================================================


class STU3d(nn.Module):
    """
    Wrapper that mimics the FNO3d constructor so it can be used
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
        self.grid_size_x, self.grid_size_y, self.grid_size_z = grid_size
        
        # k_space = max(1, self.num_eigenfunc * self.num_eigenfunc)
        
        basis_cache = os.path.join(
            os.path.dirname(__file__),
            "basis_cache"
        )
        
        self.core = StackedSTU3D(
            grid_size_x=self.grid_size_x,
            grid_size_y=self.grid_size_y,
            grid_size_z=self.grid_size_z,
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
        x: (batch_size, grid_size_x, grid_size_y, grid_size_z, in_channels, num_vars)

        Returns
        -------
        y: (batch_size, grid_size_x, grid_size_y, grid_size_z, out_channels, num_vars)
        """
        assert x.dim() == 6, f"Expected (batch_size,grid_size_x,grid_size_y,grid_size_z,in_channels, num_vars), got {x.shape}"
        batch_size, grid_size_x, grid_size_y,  grid_size_z, in_channels, num_vars = x.shape
        assert in_channels == self.in_channels, f"in_channels mismatch: {in_channels} vs {self.in_channels}"
        
        x_ch_first = x.permute(0, 5, 4, 1, 2, 3).contiguous() # (batch_size, num_vars, in_channels, grid_size_x, grid_size_y, grid_size_z)
        y = self.core(x_ch_first)  # (batch_size, num_vars, out_channels, grid_size_x, grid_size_y, grid_size_z)
        return y.permute(0, 3, 4, 5, 2, 1).contiguous() # (batch_size, grid_size_x, grid_size_y, grid_size_z, out_channels, num_vars)

