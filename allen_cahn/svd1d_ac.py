import numpy as np
import argparse
import torch
import sys
sys.path.append('/')
from utils import *
from train_model import build_and_train_model
from read_data import read_single_data
from svd1d import SVD1d, LSTM1D

random_seed = 5
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
data_path = "/data/ac1d/1D_Allen-Cahn_0.0001_5.hdf5"
data_name = 'ac'

batch_size = 50
learning_rate = 0.001
epochs = 500
step_size = 100
gamma = 0.5

grid_range = 1
in_channel = 2   # (a(x, y), x, t) for this case
grid_size = 1024
in_dim = 1
out_dim = 101

model_class = SVD1d
mlp_type = LSTM1D
loss_type = 'l2'
random_seed = 1

parser = argparse.ArgumentParser()
parser.add_argument("--num_eigenfunc", type=int, default=8)
parser.add_argument("--N_H", type=int, default=4)
parser.add_argument("--H", type=int, default=32)
parser.add_argument("--lifting_dim", type=int, default=128)
parser.add_argument("--kernel_layers", type=int, default=4)
args = parser.parse_args()

num_eigenfunc = args.num_eigenfunc
N_H = args.N_H
H = args.H
lifting_dim = args.lifting_dim
kernel_layers = args.kernel_layers


x_train, x_test, y_test, train_loader, test_loader = read_single_data(
    file_path=data_path, 
    batch_size=batch_size,
    initial_step=1,
    reduced_resolution=4,
    reduced_resolution_t=1,
    reduced_batch=1,
    test_ratio=0.1)

ntrain=x_train.shape[0]
ntest=x_test.shape[0]
grid_size=x_test.shape[1]
print(x_train.shape, x_test.shape, y_test.shape)

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
    random_seed=random_seed
)

log_mem("end", proc)
"""
 bsub -q normal -gpu num=1:mode=exclusive_process -M 80G -n 32  -o /ac_svd.txt python -u /allen_cahn/svd1d_ac.py

"""