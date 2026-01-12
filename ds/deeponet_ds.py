# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import os
import sys
import h5py

from timeit import default_timer



sys.path.append("/")
from read_data import read_mult_data

sys.path.append("/baselines")
sys.path.append("/baselines/deeponet")
from deeponet import *
from train_baseline import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--random_seed", type=int, default=0)
args = parser.parse_args()
random_seed = args.random_seed
print(random_seed)
torch.manual_seed(random_seed)
np.random.seed(random_seed)
import random
random.seed(random_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)
device = ('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

data_path  = "/data/ds/ds.hdf5"
data_name = 'ReacSorp'

batch_size = 50
learning_rate = 0.001
epochs = 500
step_size = 100
gamma = 0.5

grid_range = 1
in_channel = 1  # (a(x)) for this case
grid_size = 256
in_dim = 1
out_dim = 101
loss_type = 'l2'
initial_step=1
reduced_resolution=4
reduced_resolution_t=1



x_train, x_test, y_test, train_loader, test_loader = read_mult_data(
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

model = DeepONetCartesianProd1D(
    size=grid_size,
    in_channel_branch=in_channel,
    query_dim=1,
    out_channel = out_dim).to(device)


train_and_test_grid(
    model=model, 
    train_loader=train_loader,
    test_loader=test_loader, 
    ntrain=ntrain, 
    ntest=ntest, 
    epochs=epochs,
    batch_size=batch_size,
    in_channels=in_channel,
    learning_rate=learning_rate,
    step_size=step_size, 
    gamma=gamma, 
    out_dim=out_dim,
    loss_type=loss_type,
    grid_size=grid_size,
    device=device, 
    add_grid=True
    )
                   


torch.cuda.empty_cache()



"""
 bsub -q normal -gpu num=1:mode=exclusive_process -M 80G -n 32  -o /baselines/deeponet/ds1_deeponet.txt python -u /ds/deeponet_ds.py
"""