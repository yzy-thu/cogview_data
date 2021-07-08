import os
import sys
import math
import random
from tqdm import tqdm

import numpy as np
import torch
import argparse

parser = argparse.ArgumentParser(description="preprocess args")
parser.add_argument("--img_tokenizer_path", type=str, default=None)
parser.add_argument('--img_tokenizer_num_tokens', type=int, default=8192)
args = parser.parse_args()

from data_utils import get_tokenizer
tokenizer = get_tokenizer(args)

json_dir = '/root/mnt/dingming/wudao100g'
name = 'wudao100g'
seq_len = 1089
    
datasets = [os.path.join(json_dir, x) for x in os.listdir(json_dir) if x.endswith('.json')]
from preprocess.preprocess_text_jsonformat_data import extract_code
extract_code(datasets, name, seq_len)