import numpy as np
import argparse
import torch
import sys
sys.path.append('/')
from utils import *
from train_model import build_and_train_model
from read_data import read_mult_data
from svd2d import SVD2d, LSTM2D

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

model_class = SVD2d
loss_type = 'l2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
random_seed = 0

parser = argparse.ArgumentParser()
parser.add_argument("--num_eigenfunc", type=int, default=4)
parser.add_argument("--N_H", type=int, default=3)
parser.add_argument("--H", type=int, default=64)
parser.add_argument("--lifting_dim", type=int, default=512)
args = parser.parse_args()

num_eigenfunc = args.num_eigenfunc
N_H = args.N_H
H = args.H
lifting_dim = args.lifting_dim


x_train, x_test, y_test, train_loader, test_loader = read_mult_data(
    file_path=data_path, 
    batch_size=batch_size,
    initial_step=1,
    reduced_resolution=1,
    reduced_resolution_t=1,
    reduced_batch=1,
    test_ratio=0.1)
    
    
ntrain=x_train.shape[0]
ntest=x_test.shape[0]

grid_size=(x_test.shape[1], x_test.shape[2])  # Assuming square grid for 2D data

in_dim=x_test.shape[-1]
out_dim=y_test.shape[-1]

print(x_train.shape, x_test.shape, y_test.shape)

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
    y_normalizer=None,
    mlp_type=LSTM2D,
    in_dim=in_dim, 
    out_dim=out_dim,
    random_seed=random_seed
)

"""
 bsub -q normal -gpu num=1:mode=exclusive_process -M 80G -n 32  -o /baselines/lstm/sw_lstm.txt python -u /sallow_water/svd2dlstm_sw.py

"""