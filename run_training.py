import os
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
import argparse
from humanfriendly import format_timespan

from train.trainer_inv import TrainerInv
from utils.base_utils import load_cfg
import os
import torch
import numpy as np
import time

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str, default='configs/shape/custom/goldenqilin.yaml')
# parser.add_argument('--cfg', type=str, default='configs/mat/syn/compressor.yaml')
flags = parser.parse_args()

train_time_st = time.time()

TrainerInv(load_cfg(flags.cfg), config_path=flags.cfg).run()

print(f"Training done, costs {format_timespan(time.time() - train_time_st)}.")
