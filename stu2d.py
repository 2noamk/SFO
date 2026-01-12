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
def build_stu_basis_rfft_2d(
    grid_size_x: int,
    grid_size_y: int,
    num_eigenfunc: int,
    basis: str = "hilbert",
    cache_dir: str | Path = "./basis_cache",
) -> torch.Tensor:
    """
    Build a separable 2D STU basis on an grid_size_x×grid_size_y grid in the frequency domain.

    We first build 1D Hilbert bases along each axis (spatial domain),
    then form outer products to get 2D eigenfunctions phi_{i,j}(x,y),
    and finally take a 2D real FFT.

    Returns
    -------
    Phi_f : (num_hilbert_functions, grid_size_x, grid_size_y_r) complex64
        2D RFFT of the selected eigenfunctions, where num_hilbert_functions = num_hilbert_functions_x*num_hilbert_functions_y,
        and grid_size_y_r = grid_size_y//2 + 1.
    """
    grid_size_x = int(grid_size_x)
    grid_size_y = int(grid_size_y)
    num_eigenfunc = int(num_eigenfunc)
    sqrt_num_eigenfunc = int(math.ceil(math.sqrt(num_eigenfunc)))
    assert grid_size_x > 0 and grid_size_y > 0
    assert num_eigenfunc >= 1 and num_eigenfunc <= grid_size_x and num_eigenfunc <= grid_size_y

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{basis}_rfft2d_grid_size_x{grid_size_x}_grid_size_y{grid_size_y}_l{num_eigenfunc}.pt"

    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu")
        Phi_f = payload["eigfuncs_rfft2d"]
        return Phi_f.to(torch.complex64)
         
    # Build 1D Hilbert eigenvectors in spatial domain for each axis
    Zx = torch.tensor(hilbert(grid_size_x), dtype=torch.float64)
    wx, Vx = torch.linalg.eigh(Zx)   # (grid_size_x,), (grid_size_x, grid_size_x)
    idx_x = torch.argsort(wx, descending=True)[:num_eigenfunc]
    Vx_top = Vx.T[idx_x]            # (num_eigenfunc, grid_size_x)

    Zy = torch.tensor(hilbert(grid_size_y), dtype=torch.float64)
    wy, Vy = torch.linalg.eigh(Zy)  # (grid_size_y,), (grid_size_y, grid_size_y)
    idx_y = torch.argsort(wy, descending=True)[:num_eigenfunc]
    Vy_top = Vy.T[idx_y]            # (num_eigenfunc, grid_size_y)

    # Build separable 2D basis functions and take rfft2
    grid_size_y_r = grid_size_y // 2 + 1
    Phi_f = torch.empty((num_eigenfunc, grid_size_x, grid_size_y_r), dtype=torch.complex64)
   
    l_idx = 0
    for i in range(num_eigenfunc):
        phi_x = Vx_top[i]  # (grid_size_x,)
        phi_y = Vy_top[i]  # (grid_size_y,)
        phi_2d = torch.outer(phi_x, phi_y)  # (grid_size_x, grid_size_y)
        phi_f_2d = torch.fft.rfft2(phi_2d, s=(grid_size_x, grid_size_y))  # (grid_size_x, grid_size_y_r)
        Phi_f[l_idx] = phi_f_2d.to(torch.complex64)
        l_idx += 1


    payload = {
        "eigfuncs_rfft2d": Phi_f.cpu(),
        "grid_size_x": grid_size_x,
        "grid_size_y": grid_size_y,
        "num_eigenfunc": num_eigenfunc,
    }
    torch.save(payload, cache_path)
    return Phi_f.to(torch.complex64)


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

        # gated linear unit on the aggregated signal
        self.glu_val = nn.Conv2d(self.lifting_dim, self.lifting_dim, kernel_size=1)
        self.glu_gate = nn.Conv2d(self.lifting_dim, self.lifting_dim, kernel_size=1)

        # simple per-position normalization over channels
        self.norm = nn.LayerNorm(self.lifting_dim)

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

        # GLU
        V = self.glu_val(S)
        G = torch.sigmoid(self.glu_gate(S))
        out = V * G

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
            basis="hilbert",
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

