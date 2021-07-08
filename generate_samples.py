# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sample Generate GPT2"""

import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import argparse
import time
from datetime import datetime
from arguments import get_args
from utils import Timers
from pretrain_gpt2 import initialize_distributed
from pretrain_gpt2 import set_random_seed
from utils import load_checkpoint, get_checkpoint_iteration
from data_utils import get_tokenizer
import mpu
import deepspeed
import json

from fp16 import FP16_Module
from model import GPT2Model
from model import DistributedDataParallel as DDP
from utils import print_rank_0
from pretrain_gpt2 import get_model
import math
from copy import deepcopy
from tqdm import tqdm
from generation import get_batch, filling_sequence, add_interlacing_beam_marks


def setup_model(args):
    """Setup model and optimizer."""

    model = get_model(args)

    if args.load is not None:
        if args.deepspeed:
            iteration, release, success = get_checkpoint_iteration(args)
            path = os.path.join(args.load, str(iteration), "mp_rank_00_model_states.pt")
            checkpoint = torch.load(path)
            model.load_state_dict(checkpoint["module"])
            print(f"Load model file {path}")
        else:
            _ = load_checkpoint(
                model, None, None, args, load_optimizer_states=False)

    return model

from PIL import Image
from preprocess.utils import get_image_transforms
def read_image(args, output=None):
    tokenizer = get_tokenizer()
    img_transform = get_image_transforms()
    terminate_runs, skip_run = 0, 0
    if mpu.get_model_parallel_rank() == 0:
        while True:
            img_path = input("\nPlease input an image path (stop to exit) >>> ")
            if not img_path:
                print('Query should not be empty!')
                continue
            if img_path == "stop":
                terminate_runs = 1
                break
            try:
                # breakpoint()
                # print(img_path)
                img_path = img_path.split(' ')
                if len(img_path) == 2:
                    pre_text = img_path[1]
                else:
                    pre_text = None
                img_path = img_path[0]
                img = Image.open(img_path)
                img = img_transform(img)
                img = torch.stack([img])
                img = img.to('cuda:0')
                img_code = tokenizer.img_tokenizer.EncodeAsIds(img)
                # breakpoint()
                img_code = tokenizer.wrap_code(img_code[0].cpu().numpy()).tolist()
                img_code = img_code + [tokenizer['[ROI1]']]
                mask_len = 0
                if pre_text is not None:
                    text_list = tokenizer(pre_text)
                    mask_len += len(text_list)
                    img_code.extend(text_list)
                img_code.extend([-1]*(60-mask_len)) #1089 - 1024 - 1 -1 -1 -1
                add_interlacing_beam_marks(img_code)
                context_length = len(img_code)
                # img_code = img_code + [tokenizer['[ROI1]']]
            except ValueError as e:
                print(e)
                continue
            if context_length >= args.max_position_embeddings:
                print("\nContext length", context_length,
                      f"\nPlease give smaller context than {args.max_position_embeddings}!")
                continue
            break
    else:
        context_length = 0

    terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
    torch.distributed.broadcast(terminate_runs_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    terminate_runs = terminate_runs_tensor[0].item()
    if terminate_runs == 1:
        return terminate_runs, img_path, None, None

    context_length_tensor = torch.cuda.LongTensor([context_length])

    torch.distributed.broadcast(context_length_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    context_length = context_length_tensor[0].item()
    if mpu.get_model_parallel_rank() == 0:
        context_tokens_tensor = torch.cuda.LongTensor(img_code)
    else:
        context_tokens_tensor = torch.cuda.LongTensor([0] * context_length)
    torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())

    img_path = img_path.split('/')[-1].split('.')[-2]
    return terminate_runs, img_path, context_tokens_tensor, context_length


def read_context(args, output=None):
    tokenizer = get_tokenizer()
    terminate_runs, skip_run = 0, 0
    if mpu.get_model_parallel_rank() == 0:
        while True:
            raw_text = input("\nPlease Input Query (stop to exit) >>> ") 
            if not raw_text:
                print('Query should not be empty!')
                continue
            if raw_text == "stop":
                terminate_runs = 1
                break
            if output is not None:
                output.write(raw_text)
            try:
                seq = tokenizer.parse_query(raw_text)
                # TODO addargs
                add_interlacing_beam_marks(seq)
                context_length = len(seq)
            except ValueError as e:
                print(e)
                continue
            if context_length >= args.max_position_embeddings:
                print("\nContext length", context_length,
                      f"\nPlease give smaller context than {args.max_position_embeddings}!")
                continue
            break
    else:
        context_length = 0

    terminate_runs_tensor = torch.cuda.LongTensor([terminate_runs])
    torch.distributed.broadcast(terminate_runs_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    terminate_runs = terminate_runs_tensor[0].item()

    if terminate_runs == 1:
        return terminate_runs, raw_text, None, None

    context_length_tensor = torch.cuda.LongTensor([context_length])

    torch.distributed.broadcast(context_length_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    context_length = context_length_tensor[0].item()
    if mpu.get_model_parallel_rank() == 0:
        context_tokens_tensor = torch.cuda.LongTensor(seq)
    else:
        context_tokens_tensor = torch.cuda.LongTensor([0] * context_length)
    torch.distributed.broadcast(context_tokens_tensor, mpu.get_model_parallel_src_rank(),
                                group=mpu.get_model_parallel_group())
    return terminate_runs, raw_text, context_tokens_tensor, context_length

def generate_text(model, args, device):
    tokenizer = get_tokenizer()
    model.eval()
    output_path = "./text_samples"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cnt = 0
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs, img_path, img_code, context_length = read_image(args)
            if terminate_runs == 1:
                return
            output_file = os.path.join(output_path, f"{img_path[:20]}-{cnt}.txt")
            start_time = time.time()
            output_tokens_list = filling_sequence(model, img_code, args,
                                                     invalid_slices=[slice(0, tokenizer.img_tokenizer.num_tokens)]
                                                     )
            if mpu.get_model_parallel_rank() == 0:
                # os.system('clear')
                print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
                print("\nContext:", img_path, flush=True)
                txts = []
                for seq in output_tokens_list:
                    decoded_txts, decoded_imgs = tokenizer.DecodeIds(seq.tolist())
                    txts.append(str(decoded_txts).split("。")[0].split('\'')[-1] + '。\n')
                print("\nSave to: ", output_file, flush=True)
                print(txts)
                with open(output_file, "w", encoding='utf-8') as f:
                    f.writelines(txts)
                    # json.dump(txts, f, ensure_ascii=False)
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            cnt += 1
    pass


def generate_images(model, args, device):
    from torchvision.utils import save_image
    tokenizer = get_tokenizer()
    model.eval()
    output_path = "./samples"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    cnt = 0
    with torch.no_grad():
        while True:
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            terminate_runs, raw_text, seq, context_length = read_context(args)
            if terminate_runs == 1:
                return
            # output_file = os.path.join(output_path, f"{raw_text[:20]}-{cnt}-{datetime.now().strftime('%m-%d-%H-%M')}.jpg")
            output_file = os.path.join(output_path, f"{raw_text[:20]}-{cnt}.jpg")
            start_time = time.time()
            output_tokens_list = filling_sequence(model, seq, args, 
                invalid_slices=[slice(tokenizer.img_tokenizer.num_tokens, tokenizer.num_tokens)]
                )
            if mpu.get_model_parallel_rank() == 0:
                # os.system('clear')
                print("\nTaken time {:.2f}\n".format(time.time() - start_time), flush=True)
                print("\nContext:", raw_text, flush=True)
                imgs = []
                for seq in output_tokens_list:
                    decoded_txts, decoded_imgs = tokenizer.DecodeIds(seq.tolist())
                    imgs.append(decoded_imgs[0])
                imgs = torch.cat(imgs, dim=0)
                print("\nSave to: ", output_file, flush=True)
                save_image(imgs, output_file, normalize=True)
                
            torch.distributed.barrier(group=mpu.get_model_parallel_group())
            cnt += 1

def prepare_tokenizer(args):
    tokenizer = get_tokenizer(args)

    num_tokens = tokenizer.num_tokens
    before = num_tokens
    after = before
    multiple = args.make_vocab_size_divisible_by * \
               mpu.get_model_parallel_world_size()
    while (after % multiple) != 0:
        after += 1
    print_rank_0('> padded vocab (size: {}) with {} dummy '
                 'tokens (new size: {})'.format(
        before, after - before, after))

    args.vocab_size = after
    # args.vocab_size = 58240 #TODO ????????
    print("prepare tokenizer done", flush=True)

    return tokenizer


def main():
    """Main training program."""

    print('Generate Samples')

    # Disable CuDNN.
    torch.backends.cudnn.enabled = False

    # Arguments.
    args = get_args()

    # Pytorch distributed.
    initialize_distributed(args)

    # Random seeds for reproducability.
    set_random_seed(args.seed)

    # get the tokenizer
    tokenizer = prepare_tokenizer(args)

    # Model, optimizer, and learning rate.
    model = setup_model(args)

    # setting default batch size to 1
    args.batch_size = 1

    if args.generate_type == "text2img":
        generate_images(model, args, torch.cuda.current_device())
    else:
        generate_text(model, args, torch.cuda.current_device())


if __name__ == "__main__":
    main()
