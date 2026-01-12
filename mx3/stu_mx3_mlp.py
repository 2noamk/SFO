#!/usr/bin/env python3
"""
STU3d on MX data

- Uses the SAME data loader: `read_mult_data` (deepONet-style MX data)
- Uses the SAME training helper: `build_and_train_model`
- Swaps the model from `SVD1d` to `STU1d` (spatial STU operator)

    python stu_ds.py \
        --data_path /data/ac2/ac2.hdf5 \
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

from read_data import read_mult_data
from train_model import build_and_train_model
from utils import log_mem
from stu3d_mlp_vars import STU3d


# =========================================================
# CLI
# =========================================================

def get_args():
    parser = argparse.ArgumentParser(description="STU3d on MX data (DeepONet-style).")

    # Data / paths
    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/mx3/mx3.hdf5",
        help="Path to the mx3 DeepONet HDF5 file.",
    )
    parser.add_argument(
        "--data_name",
        type=str,
        default="mx3",
        help="Short name for logging / saving.",
    )

    # Resolution & splitting
    parser.add_argument(
        "--reduced_resolution",
        type=int,
        default=1,
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
        default=2,
        help="Number of initial time steps used as input (as in svd1d_ds).",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.1,
        help="Fraction of seeds used for test set (as in read_mult_data).",
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
        default=1,
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
    # 1) Load MX DeepONet-style data
    # -----------------------------------------------------
    print(f"[INFO] Loading MX data from: {args.data_path}")

    
    x_train, x_test, y_test, train_loader, test_loader = read_mult_data(
        file_path= args.data_path, 
        batch_size=args.batch_size,
        initial_step=args.initial_step,
        reduced_resolution=args.reduced_resolution,
        reduced_resolution_t=args.reduced_resolution_t,
        reduced_batch=1,
        test_ratio=args.test_ratio,
        )
        
    # x_train=x_train[:1]
    # x_test=x_test[:1]
    # y_test=y_test[:1]
    

    print(f"[INFO] x_train.shape: {x_train.shape}")
    print(f"[INFO] x_test.shape:  {x_test.shape}")
    print(f"[INFO] y_test.shape:  {y_test.shape}")

    # 3D case: (a(x, y, z)=N, x, y, z, t=out_dim, num_vars) for this case
    if x_train.ndim != 6 or y_test.ndim != 6:
        raise RuntimeError(
            f"Expected 6D spatial DeepONet data: x,y,z ∈ ℝ^{'{N×X×Y×Z×T×VAR}'}, "
            f"got {x_train.ndim} x_train {x_train.shape}, y_test {y_test.shape}"
        )

    N_train = x_train.shape[0]
    N_test = x_test.shape[0]
    grid_size=(x_test.shape[1], x_test.shape[2], x_test.shape[3])  # Assuming cube grid for 3D data
    in_dim = x_train.shape[-2]
    out_dim = y_test.shape[-2]
    num_vars = y_test.shape[-1]

    print(f"[INFO] N_train={N_train}, N_test={N_test}")
    print(f"[INFO] grid_size={grid_size}, in_dim={in_dim}, out_dim={out_dim}, num_vars={num_vars}")

    # For consistency with svd1d_ds:
    #   grid_range is usually 1.0 for x ∈ [0,1].
    grid_range = 1.0
    in_channel = in_dim

    # -----------------------------------------------------
    # 2) Define STU model class for the training helper
    # -----------------------------------------------------
    model_class = STU3d

    # If you want to log memory as in the SVD script
    proc = psutil.Process(os.getpid())
    log_mem("[STU3d-MX] before training", proc)

    # -----------------------------------------------------
    # 3) Train via the generic SVD training pipeline
    # -----------------------------------------------------
    print("[INFO] Starting training (STU3d)...")
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
        mlp_type=None,                  # SVD-specific; STU ignores it
        kernel_layers=args.kernel_layers,
        in_dim=in_dim,
        out_dim=out_dim,
        num_vars=num_vars,
        device=device,
        loss_type=args.loss_type,
        y_normalizer=None,
        random_seed=args.random_seed,
        plot=not args.no_plot,
        mno=False,
        no_orth=True,                   # orthogonality regularization is SVD-specific
        model_name='stu2d_mx3d',        # for saving purposes,
        boundary_loss=True,
    )

    log_mem("[STU3d-MX] after training", proc)
    print("[INFO] Training finished.")


if __name__ == "__main__":
    main()


"""
 bsub -q normal -gpu num=1:mode=exclusive_process -M 100G -n 32  -o /mx3_boundary.txt python -u /mx3/stu_mx3_mlp.py
  

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx_20_32_stu_mlp_4_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 4 \
  --lifting_dim 32 \
  --random_seed 0 
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx_20_32_stu_mlp_8_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 8 \
  --lifting_dim 32 \
  --random_seed 0 
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx_20_32_stu_mlp_12_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 12 \
  --lifting_dim 32 \
  --random_seed 0 
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx_20_32_stu_mlp_16_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 16 \
  --lifting_dim 32 \
  --random_seed 0 
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx_20_32_stu_mlp_20_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 20 \
  --lifting_dim 32 \
  --random_seed 0 
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx_20_32_stu_mlp_24_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 24 \
  --lifting_dim 32 \
  --random_seed 0 
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx_20_32_stu_mlp_32_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 32 \
  --lifting_dim 32 \
  --random_seed 0 
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_16_32_mlp_rt8.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 16 \
  --lifting_dim 32 \
  --random_seed 0 \
  --reduced_resolution_t 8
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_16_32_mlp_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 20 \
  --lifting_dim 32 \
  --random_seed 0 \
  --epochs 500
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_16_32_mlp_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 24 \
  --lifting_dim 32 \
  --random_seed 0 \
  --epochs 500
  
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_16_32_mlp_reduced_2.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 16 \
  --lifting_dim 32 \
  --reduced_resolution 2
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_8_32_mlp_reduced_4.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 8 \
  --lifting_dim 32 \
  --reduced_resolution 4
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_5_32_mlp_reduced_6.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 5 \
  --lifting_dim 32 \
  --reduced_resolution 6
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_4_32_mlp_reduced_8.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 4 \
  --lifting_dim 32 \
  --reduced_resolution 8
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_3_32_mlp_reduced_10.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 3 \
  --lifting_dim 32 \
  --reduced_resolution 10

    
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_20_32_mlp_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 20 \
  --lifting_dim 32 \
  --random_seed 0
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_20_32_mlp_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 20 \
  --lifting_dim 32 \
  --random_seed 1
  
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_20_32_mlp_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 20 \
  --lifting_dim 32 \
  --random_seed 2
  
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_20_32_mlp_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 20 \
  --lifting_dim 32 \
  --random_seed 3
  
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_20_32_mlp_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 20 \
  --lifting_dim 32 \
  --random_seed 4
  
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_20_32_mlp_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 20 \
  --lifting_dim 32 \
  --random_seed 5
  
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_20_32_mlp_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 20 \
  --lifting_dim 32 \
  --random_seed 6
  
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_20_32_mlp_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 20 \
  --lifting_dim 32 \
  --random_seed 7
  
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_20_32_mlp_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 20 \
  --lifting_dim 32 \
  --random_seed 8
  
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_20_32_mlp_r.txt \
  python -u /mx3/stu_mx3_mlp.py\
  --num_eigenfunc 20 \
  --lifting_dim 32 \
  --random_seed 9
  
  
  """