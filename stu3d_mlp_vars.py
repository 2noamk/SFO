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


@torch.no_grad()
def build_stu_basis_rfft_3d(
    grid_size_x: int,
    grid_size_y: int,
    grid_size_z: int,
    num_eigenfunc: int,
    basis: str = "hilbert",
    cache_dir: str | Path = "./basis_cache",
) -> torch.Tensor:
    """
    Build a separable 3D STU basis on an grid_size_x×grid_size_y×grid_size_z grid in the frequency domain.

    We first build 1D Hilbert bases along each axis (spatial domain),
    then form outer products to get 3D eigenfunctions phi_{i,j}(x,y,z),
    and finally take a 3D real FFT.

    Returns
    -------
    Phi_f : (num_hilbert_functions, grid_size_x, grid_size_y, grid_size_z_r) complex64
        and grid_size_z_r = grid_size_z//2 + 1.
    """
    grid_size_x = int(grid_size_x)
    grid_size_y = int(grid_size_y)
    grid_size_z = int(grid_size_z)
    num_eigenfunc = int(num_eigenfunc)
    sqrt_num_eigenfunc = int(math.ceil(math.sqrt(num_eigenfunc)))
    assert grid_size_x > 0 and grid_size_y > 0 and grid_size_z > 0
    assert num_eigenfunc >= 1 and num_eigenfunc <= grid_size_x and num_eigenfunc <= grid_size_y and num_eigenfunc <= grid_size_z

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{basis}_rfft3d_grid_size_x{grid_size_x}_grid_size_y{grid_size_y}_grid_size_z{grid_size_z}_l{num_eigenfunc}.pt"

    # if cache_path.exists():
    #     payload = torch.load(cache_path, map_location="cpu")
    #     Phi_f = payload["eigfuncs_rfft3d"]
    #     return Phi_f.to(torch.complex64)
         
    # Build 1D Hilbert eigenvectors in spatial domain for each axis
    Zx = torch.tensor(hilbert(grid_size_x), dtype=torch.float64)
    wx, Vx = torch.linalg.eigh(Zx)   # (grid_size_x,), (grid_size_x, grid_size_x)
    idx_x = torch.argsort(wx, descending=True)[:num_eigenfunc]
    Vx_top = Vx.T[idx_x]            # (num_eigenfunc, grid_size_x)

    Zy = torch.tensor(hilbert(grid_size_y), dtype=torch.float64)
    wy, Vy = torch.linalg.eigh(Zy)  # (grid_size_y,), (grid_size_y, grid_size_y)
    idx_y = torch.argsort(wy, descending=True)[:num_eigenfunc]
    Vy_top = Vy.T[idx_y]            # (num_eigenfunc, grid_size_y)

    Zz = torch.tensor(hilbert(grid_size_z), dtype=torch.float64)
    wz, Vz = torch.linalg.eigh(Zz)  # (grid_size_z,), (grid_size_z, grid_size_z)
    idx_z = torch.argsort(wz, descending=True)[:num_eigenfunc]
    Vz_top = Vz.T[idx_z]            # (num_eigenfunc, grid_size_z)


    # Build separable 3D basis functions and take rfft2
    grid_size_z_r = grid_size_z // 2 + 1
    Phi_f = torch.empty((num_eigenfunc, grid_size_x, grid_size_y, grid_size_z_r), dtype=torch.complex64)
   
    l_idx = 0
    for i in range(num_eigenfunc):
        phi_x = Vx_top[i]  # (grid_size_x,)
        phi_y = Vy_top[i]  # (grid_size_y,)
        phi_z = Vz_top[i]  # (grid_size_z,)
        phi_3d = torch.einsum('i,j,k->ijk', (phi_x, phi_y, phi_z))  # (grid_size_x, grid_size_y, grid_size_z)
        phi_f_3d = torch.fft.rfftn(phi_3d, s=(grid_size_x, grid_size_y, grid_size_z))  # (grid_size_x, grid_size_y, grid_size_z_r)
        Phi_f[l_idx] = phi_f_3d.to(torch.complex64)
        l_idx += 1


    payload = {
        "eigfuncs_rfft3d": Phi_f.cpu(),
        "grid_size_x": grid_size_x,
        "grid_size_y": grid_size_y,
        "grid_size_z": grid_size_z,
        "num_eigenfunc": num_eigenfunc,
    }
    torch.save(payload, cache_path)
    return Phi_f.to(torch.complex64)


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
        Xf = torch.fft.rfftn(z, dim=(-3, -2, -1))  # (B, V, d, Nx, Ny, Nz_r) complex

        # Apply basis filters in frequency
        Phi = self.Phi_f.conj().unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1,1,1,L,Nx,Ny,Nz_r)
        Uf  = Xf.unsqueeze(3) * Phi                                      # (B,V,d,L,Nx,Ny,Nz_r) complex

        # Keep Theta real: mix real/imag separately
        A = Uf.real  # (B,V,d,L,Nx,Ny,Nz_r)
        B = Uf.imag  # (B,V,d,L,Nx,Ny,Nz_r)

        # Mix d -> h using Theta and sum over l directly in frequency
        # Output: (B, V, h, Nx, Ny, Nz_r)
        Sf_re = torch.einsum("bvdlxyz,lhd->bvhxyz", A, self.Theta)
        Sf_im = torch.einsum("bvdlxyz,lhd->bvhxyz", B, self.Theta)

        Sf = torch.complex(Sf_re, Sf_im)  # (B, V, h, Nx, Ny, Nz_r) complex

        # One inverse FFT total
        S = torch.fft.irfftn(
            Sf,
            s=(self.grid_size_x, self.grid_size_y, self.grid_size_z),
            dim=(-3, -2, -1),
        )  # (B, V, h, Nx, Ny, Nz) float


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
            basis="hilbert",
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

