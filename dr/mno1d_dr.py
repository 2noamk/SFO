# coding=utf-8
import torch
import numpy as np
import os
import sys

sys.path.append("/baselines/mno/")
sys.path.append("/baselines/")
from mno1d import *
from read_data import read_single_data
from train_model import build_and_train_model
from svd1d import SVD1d, MLP1D, CONV1D

random_seed = 0
torch.manual_seed(random_seed)
np.random.seed(random_seed)
import random
random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
device = ('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


data_path  = "/data/dr/ReacDiff_Nu0.5_Rho1.0.hdf5"
data_name = 'ReacDiff'

batch_size = 50
learning_rate = 0.001
epochs = 500
step_size = 100
gamma = 0.5

grid_range = 1
in_channel = 2   # (a(x, y), x, t) for this case
grid_size = 256
in_dim = 1
out_dim = 101
loss_type = 'l2'
initial_step=1
reduced_resolution=4
reduced_resolution_t=1


x_train, x_test, y_test, train_loader, test_loader = read_single_data(
    file_path=data_path, 
    batch_size=50,
    initial_step=1,
    reduced_resolution=4,
    reduced_resolution_t=1,
    reduced_batch=1,
    test_ratio=0.1)

ntrain=x_train.shape[0]
ntest=x_test.shape[0]
grid_size=x_test.shape[1]
print(x_train.shape, x_test.shape, y_test.shape)


ntest=x_test.shape[0]
grid_size=x_test.shape[1]

num_eigenfunc=8
H=128
N_H=3
lifting_dim=512

model_class = MNO1d

mno = True


build_and_train_model(
    data_name=data_name, model_class=model_class, mlp_type=MLP1D,
    x_test=x_test, y_test=y_test, train_loader=train_loader, test_loader=test_loader,
    ntrain=ntrain, ntest=ntest, num_eigenfunc=num_eigenfunc, H=H, N_H=N_H, lifting_dim=lifting_dim, out_dim=out_dim,
    epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, step_size=step_size,  
    gamma=gamma, grid_size=grid_size, in_channel=in_channel, grid_range=grid_range, device=device,
    loss_type=loss_type, random_seed=random_seed, mno=mno, no_orth=False
    )
torch.cuda.empty_cache()



"""
 bsub -q normal -gpu num=1:mode=exclusive_process -M 40G -n 32  -o /baselines/mno/dr_mno.txt python -u /dr/mno1d_dr.py
"""