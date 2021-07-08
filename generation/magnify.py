# -*- encoding: utf-8 -*-
'''
@File    :   magnify.py
@Time    :   2021/01/14 00:41:40
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F

# TODO finish this after training super-resolution task

# def magnify(model, tokenizer, tokens_list, args):
                
#         # 32 * 32 to 4 16 * 16
#         s = int(math.sqrt(tokens_list.numel() + 1e-6))
#         code = tokens_list.view(s, s)
#         sep_token = torch.tensor([tokenizer.command_name_map['eos']], dtype=torch.long, device=code.device)

#         magnified_code = code.new_zeros((s * 2, s * 2), dtype=torch.long) - 1
#         for i in tqdm(range(s // 8 - 1)):
#             for j in range(s // 8 - 1):
#                 code_part = code[8 * i: 8 * (i+2), 8 * j: 8 * (j+2)].reshape(-1)
#                 context_length = code_part.shape[0] + 1

#                 magnified_code_part = magnified_code[16 * i: 16 * (i+2), 16 * j: 16 * (j+2)].reshape(-1)
#                 context_tokens_tensor = torch.cat([code_part, sep_token, magnified_code_part], dim=0)
#                 while context_tokens_tensor[context_length] >= 0:
#                     context_length += 1

#                 magnified_code_part_completed, _ = sample_sequence_seg(model, tokenizer, context_tokens_tensor, context_length, args, code.device, valid_sep=tokenizer.image_tokens)
#                 magnified_code[16 * i: 16 * (i+2), 16 * j: 16 * (j+2)] = magnified_code_part_completed[257:].view(32, 32)
#         return magnified_code