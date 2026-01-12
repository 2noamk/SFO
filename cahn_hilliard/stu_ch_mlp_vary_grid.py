#!/usr/bin/env python3
"""
STU1d on DS data

This is a drop-in sibling of svd1d_ca.py:

- Uses the SAME data loader: `read_mult_data` (deepONet-style DS data)
- Uses the SAME training helper: `build_and_train_model`
- chaps the model from `SVD1d` to `STU1d` (spatial STU operator)

Invoke similarly to svd1d_ca.py, e.g.:

    python stu_ca.py \
        --data_path /data/ch/ch.hdf5 \
        --num_eigenfunc 20 --lifting_dim 32
"""

import argparse
import psutil
import os
import sys
import numpy as np
import torch

# Make sure your repo root is on the path
sys.path.append("/")

from read_data import read_single_data
from train_model import build_and_train_model
from utils import log_mem
from stu1d_mlp_vary_grid import STU1d


# =========================================================
# CLI
# =========================================================

def get_args():
    parser = argparse.ArgumentParser(description="STU1d on DS data (DeepONet-style).")

    # Data / paths
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/ch/ch.hdf5",
        help="Path to the DS DeepONet HDF5 file.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="cahnhilliard",
        help="Short name for logging / saving (same as in svd1d_ca).",
    )

    # Resolution & splitting
    parser.add_argument(
        "--reduced_resolution",
        type=int,
        default=4,
        help="Spatial downsampling factor r (x -> x[::r]).",
    )
    parser.add_argument(
        "--reduced_resolution_t",
        type=int,
        default=1,
        help="Temporal downsampling factor r_t (t -> t[::r_t]).",
    )
    parser.add_argument(
        "--initial_step",
        type=int,
        default=1,
        help="Number of initial time steps used as input (as in svd1d_ch).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of seeds used for test set (as in read_mult_data).",
    )
    parser.add_argument(
        "--subsample_factor",
        type=int,
        default=1,
        help="downsampling factor.",
    )
    # SVD/STU "spectral" hyper-params (we reuse them in STU)
    parser.add_argument(
        "--num_eigenfunc",
        type=int,
        default=20,
        help="Number of eigenfunctions / STU modes (k_space).",
    )
    parser.add_argument(
        "--lifting_dim",
        type=int,
        default=32,
        help="Latent width C in STU; matches SVD1d lifting_dim.",
    )
    parser.add_argument(
        "--kernel_layers",
        type=int,
        default=4,
        help="Number of stacked STU spatial layers (depth_space).",
    )

    # Optim / training
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of training epochs.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Mini-batch size.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help="Initial learning rate (Adam).",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=100,
        help="Step size for LR scheduler (StepLR).",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="LR decay factor for StepLR.",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="l2",
        choices=["l2", "l1"],
        help="Loss to use inside train_and_test (same as SVD code).",
    )

    # Misc
    parser.add_argument(
        "--random_seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Disable plotting in build_and_train_model.",
    )

    return parser.parse_args()


# =========================================================
# Main
# =========================================================

def main():
    args = get_args()

    # Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # Seeding
    torch.manual_seed(args.random_seed)
    np.random.seed(args.random_seed)

    # -----------------------------------------------------
    # 1) Load DS DeepONet-style data
    # -----------------------------------------------------
    print(f"[INFO] Loading DS data from: {args.data_path}")

    
    x_train, _, _, train_loader, _ = read_single_data(
        file_path= args.data_path, 
        batch_size=args.batch_size,
        initial_step=args.initial_step,
        reduced_resolution=args.reduced_resolution,
        reduced_resolution_t=args.reduced_resolution_t,
        reduced_batch=1,
        test_ratio=args.test_ratio,
        )
    
    _, x_test, y_test, _, test_loader = read_single_data(
        file_path= args.data_path, 
        batch_size=args.batch_size,
        initial_step=args.initial_step,
        reduced_resolution=args.reduced_resolution * args.subsample_factor,
        reduced_resolution_t=args.reduced_resolution_t,
        reduced_batch=1,
        test_ratio=args.test_ratio,
        )
    
    

    print(f"[INFO] x_train.shape: {x_train.shape}")
    print(f"[INFO] x_test.shape:  {x_test.shape}")
    print(f"[INFO] y_test.shape:  {y_test.shape}")

    # DeepONet 1D case: x is (N, X, in_dim), y is (N, X, out_dim)
    if x_train.ndim != 3 or y_test.ndim != 3:
        raise RuntimeError(
            f"Expected 1D spatial DeepONet data: x,y ∈ ℝ^{'{N×X×C}'}, "
            f"got x_train {x_train.shape}, y_test {y_test.shape}"
        )

    N_train = x_train.shape[0]
    N_test = x_test.shape[0]
    grid_size = x_train.shape[1]
    in_dim = x_train.shape[-1]
    out_dim = y_test.shape[-1]

    print(f"[INFO] seed={args.random_seed}")
    print(f"[INFO] N_train={N_train}, N_test={N_test}")
    print(f"[INFO] grid_size={grid_size}, in_dim={in_dim}, out_dim={out_dim}")
    print(f"[INFO] num_eigenfunc={args.num_eigenfunc}, lifting_dim={args.lifting_dim}")

    # For consistency with svd1d_ch:
    #   grid_range is usually 1.0 for x ∈ [0,1].
    grid_range = 1.0
    in_channel = in_dim

    # -----------------------------------------------------
    # 2) Define STU model class for the training helper
    # -----------------------------------------------------
    model_class = STU1d

    # If you want to log memory as in the SVD script
    proc = psutil.Process(os.getpid())
    log_mem("[STU1d-DS] before training", proc)

    # -----------------------------------------------------
    # 3) Train via the generic SVD training pipeline
    # -----------------------------------------------------
    print("[INFO] Starting training (STU1d)...")
    build_and_train_model(
        data_name=args.data_name,
        model_class=model_class,
        x_test=x_test,
        y_test=y_test,
        train_loader=train_loader,
        test_loader=test_loader,
        ntrain=N_train,
        ntest=N_test,
        num_eigenfunc=args.num_eigenfunc,
        H=1,
        N_H=1,
        lifting_dim=args.lifting_dim,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        step_size=args.step_size,
        gamma=args.gamma,
        grid_size=grid_size,
        in_channel=in_channel,
        grid_range=grid_range,
        mlp_type="linear",
        kernel_layers=args.kernel_layers,
        in_dim=in_dim,
        out_dim=out_dim,
        device=device,
        loss_type=args.loss_type,
        y_normalizer=None,
        random_seed=args.random_seed,
        plot=not args.no_plot,
        mno=False,
        no_orth=True,                   # orthogonality regularization is SVD-specific
        model_name='stu1d_ch',
    )

    log_mem("[STU1d-DS] after training", proc)
    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()


"""

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /ch_20_64_mlp_vary_2_r.txt \
  python -u /cahn_hilliard/stu_ch_mlp_vary_grid.py\
  --num_eigenfunc 20 \
  --lifting_dim 64 \
  --subsample_factor 2
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /ch_20_64_mlp_vary_4_r.txt \
  python -u /cahn_hilliard/stu_ch_mlp_vary_grid.py\
  --num_eigenfunc 20 \
  --lifting_dim 64 \
  --subsample_factor 4
  
"""
