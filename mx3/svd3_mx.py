import numpy as np
import argparse
import torch
import sys
sys.path.append('/')
from utils import *
from train_model import build_and_train_model
from read_data import read_mult_data
from svd3d import SVD3d, MLP3D


parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=0)
parser.add_argument("--num_eigenfunc", type=int, default=4)
parser.add_argument("--N_H", type=int, default=3)
parser.add_argument("--H", type=int, default=64)
parser.add_argument("--lifting_dim", type=int, default=64)
parser.add_argument("--kernel_layers", type=int, default=4)
args = parser.parse_args()

random_seed = args.random_seed


torch.manual_seed(random_seed)
np.random.seed(random_seed)
import random
random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
device = ('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


""" Model configurations """
data_path  = "/data/mx3/mx3.hdf5"
data_name = 'mx3'

batch_size = 1
learning_rate = 0.001
epochs = 500
step_size = 100
gamma = 0.5

grid_range = 1
in_channel = 5  # (a(x, y, z), x, y, z, t1, t2) for this case
loss_type = 'l2'
initial_step=2
reduced_resolution=1
reduced_resolution_t=1

model_class = SVD3d
mlp_type = MLP3D
loss_type = 'l2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


num_eigenfunc = args.num_eigenfunc
N_H = args.N_H
H = args.H
lifting_dim = args.lifting_dim
kernel_layers = args.kernel_layers


x_train, x_test, y_test, train_loader, test_loader = read_mult_data(
    file_path=data_path, 
    batch_size=batch_size,
    initial_step=initial_step,
    reduced_resolution=reduced_resolution,
    reduced_resolution_t=reduced_resolution_t,
    reduced_batch=1,
    test_ratio=0.1)



ntrain=x_train.shape[0]
ntest=x_test.shape[0]
print(x_train.shape, x_test.shape, y_test.shape)

    

grid_size=(x_test.shape[1], x_test.shape[2], x_test.shape[3])  # Assuming cube grid for 3D data
in_dim = x_train.shape[-2]
out_dim = y_test.shape[-2]
num_vars = y_test.shape[-1]




proc = psutil.Process(os.getpid())
log_mem("startup", proc)

build_and_train_model(
    data_name=data_name,
    model_class=model_class,
    x_test=x_test,
    y_test=y_test,
    train_loader=train_loader,
    test_loader=test_loader,
    ntrain=ntrain,
    ntest=ntest,
    num_eigenfunc=num_eigenfunc,
    H=H,
    N_H=N_H,
    lifting_dim=lifting_dim,
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    step_size=step_size,
    gamma=gamma,
    grid_size=grid_size,
    in_channel=in_channel,
    grid_range=grid_range,
    device=device,
    loss_type=loss_type,
    kernel_layers=kernel_layers,
    y_normalizer=None,
    mlp_type=mlp_type,
    in_dim=in_dim, 
    out_dim=out_dim,
    random_seed=random_seed,
    num_vars=num_vars
)
log_mem("end", proc)

"""
 bsub -q normal -gpu num=1:mode=exclusive_process -M 600G -n 32  -o /mx3_svd.txt python -u /mx3/svd3_mx.py



bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_svd_r.txt \
  python -u /mx3/svd3_mx.py \
  --random_seed 0

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_svd_r.txt \
  python -u /mx3/svd3_mx.py \
  --random_seed 1

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_svd_r.txt \
  python -u /mx3/svd3_mx.py \
  --random_seed 2

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_svd_r.txt \
  python -u /mx3/svd3_mx.py \
  --random_seed 3

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_svd_r.txt \
  python -u /mx3/svd3_mx.py \
  --random_seed 4

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_svd_r.txt \
  python -u /mx3/svd3_mx.py \
  --random_seed 5

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_svd_r.txt \
  python -u /mx3/svd3_mx.py \
  --random_seed 6

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_svd_r.txt \
  python -u /mx3/svd3_mx.py \
  --random_seed 7

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_svd_r.txt \
  python -u /mx3/svd3_mx.py \
  --random_seed 8

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /mx3_svd_r.txt \
  python -u /mx3/svd3_mx.py \
  --random_seed 9
"""