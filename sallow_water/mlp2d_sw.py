# coding=utf-8
import torch
import numpy as np
import os
import sys

sys.path.append("/baselines/mlp/")
sys.path.append("/baselines/")
from mlp2d import *
from train_func import *

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


""" Model configurations """
data_path  = "/data/sw2/sw2.hdf5"
data_name = 'sw'

batch_size = 10
learning_rate = 0.001
epochs = 200
step_size = 100
gamma = 0.5

grid_range = 1
in_channel = 3   # (a(x, y), x, y, t) for this case
grid_size_x = 128
grid_size_y = 128
grid_size = (grid_size_x, grid_size_y)
in_dim = 1
out_dim = 101
loss_type = 'l2'
initial_step=1
reduced_resolution=1
reduced_resolution_t=1


x_train, x_test, y_test, train_loader, test_loader = read_mult_data_sw(
    file_path=data_path, 
    batch_size=batch_size,
    initial_step=initial_step,
    reduced_resolution=reduced_resolution,
    reduced_resolution_t=reduced_resolution_t,
    reduced_batch=1,
    test_ratio=0.1)

  


ntrain=x_train.shape[0]
ntest=x_test.shape[0]
grid_size=x_test.shape[1]
print(x_train.shape, x_test.shape, y_test.shape)

lifting_dim=64


ntest=x_test.shape[0]
grid_size=x_test.shape[1]

model = MLP2d(in_channel, lifting_dim, grid_range, grid_size,
                 kernel_layers=4, out_dim=out_dim, device=device).to(device)


    
last_test_loss = train_and_test(
    model=model, 
    train_loader=train_loader,
    test_loader=test_loader, 
    ntrain=ntrain, 
    ntest=ntest, 
    epochs=epochs,
    batch_size=batch_size,
    learning_rate=learning_rate,
    step_size=step_size, 
    gamma=gamma, 
    in_dim=in_dim,
    out_dim=out_dim,
    loss_type=loss_type,
    grid_size=grid_size,
    device=device, 
    reduced_resolution=reduced_resolution, 
    reduced_resolution_t=reduced_resolution_t
    )

torch.cuda.empty_cache()



"""
 bsub -q normal -gpu num=1:mode=exclusive_process -M 40G -n 32  -o /baselines/mlp/dr_mlp.txt python -u /dr/mlp1d_dr.py
"""