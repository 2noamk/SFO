# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import TensorDataset, DataLoader, Dataset
import numpy as np
import argparse
import os
import sys
import h5py

from timeit import default_timer


sys.path.append("/")
from read_data import read_single_data

sys.path.append("/baselines")
sys.path.append("/baselines/kno/")
from kno import *
from train_baseline import *


parser = argparse.ArgumentParser(description="FNO.")


parser.add_argument(
    "--random_seed",
    type=int,
    default=0,
    help="Random seed.",
)
args = parser.parse_args()
torch.manual_seed(args.random_seed)
np.random.seed(args.random_seed)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.random_seed)
device = ('cuda' if torch.cuda.is_available() else 'cpu')
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

""" Model configurations """
data_path  = "/data/ch/ch.hdf5"
data_name = 'ch'

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

operator_size = 8
encoder = encoder_mlp(in_dim, operator_size)
decoder = decoder_mlp(out_dim, operator_size)
model = KNO1d(encoder, decoder, operator_size).to(device)

    
last_test_loss = train_and_test_grid(
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
    add_grid=False
    )

torch.cuda.empty_cache()



"""
 bsub -q normal -gpu num=1:mode=exclusive_process -M 80G -n 32  -o /baselines/kno/ch1_kno.txt python -u /cahn_hilliard/kno1d_ch.py
 

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /baselines/kno/ch1_kno.txt \
  python -u /cahn_hilliard/kno1d_ch.py \
  --random_seed 0

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /baselines/kno/ch1_kno.txt \
  python -u /cahn_hilliard/kno1d_ch.py \
  --random_seed 1
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /baselines/kno/ch1_kno.txt \
  python -u /cahn_hilliard/kno1d_ch.py \
  --random_seed 2

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /baselines/kno/ch1_kno.txt \
  python -u /cahn_hilliard/kno1d_ch.py \
  --random_seed 3
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /baselines/kno/ch1_kno.txt \
  python -u /cahn_hilliard/kno1d_ch.py \
  --random_seed 4

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /baselines/kno/ch1_kno.txt \
  python -u /cahn_hilliard/kno1d_ch.py \
  --random_seed 5
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /baselines/kno/ch1_kno.txt \
  python -u /cahn_hilliard/kno1d_ch.py \
  --random_seed 6

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /baselines/kno/ch1_kno.txt \
  python -u /cahn_hilliard/kno1d_ch.py \
  --random_seed 7
  
bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /baselines/kno/ch1_kno.txt \
  python -u /cahn_hilliard/kno1d_ch.py \
  --random_seed 8

bsub -q normal -gpu num=1:mode=exclusive_process -M 200G -n 32 \
  -o /baselines/kno/ch1_kno.txt \
  python -u /cahn_hilliard/kno1d_ch.py \
  --random_seed 9
"""