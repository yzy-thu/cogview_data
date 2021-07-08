# -*- encoding: utf-8 -*-
'''
@File    :   sampling.py
@Time    :   2021/01/13 19:52:12
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

from pretrain_gpt2 import get_masks_and_position_ids
from data_utils import get_tokenizer


def top_k_logits(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    # This function has been mostly taken from huggingface conversational ai code at
    # https://medium.com/huggingface/how-to-build-a-state-of-the-art-conversational-ai-with-transfer-learning-2d818ac26313


    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        # convert to 1D
        logits = logits.view(logits.size()[1]).contiguous()
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        # going back to 2D
        logits = logits.view(1, -1).contiguous()

    return logits

def get_batch(context_tokens, device, args):
    tokens = context_tokens
    if len(tokens.shape) == 1:
        tokens = tokens.view(args.batch_size, -1).contiguous()
    else:
        tokens = tokens.view(tokens.shape[0], -1).contiguous()
    tokens = tokens.to(device)

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_masks_and_position_ids(
        tokens)
    return tokens, attention_mask, position_ids

def filling_sequence(
        model, 
        seq, 
        args, 
        mems=None, 
        invalid_slices=[], 
        **kwargs):
    '''
        seq: [2, 3, 5, ..., -1(to be generated), -N (N beams), -1]
        context_length: first non(-1)s
    '''
    device = seq.device
    assert len(seq.shape) == 1
    out_seq_length = len(seq)
    # building the initial tokens, attention_mask, and position_ids
    context_length = 0
    while seq[context_length] >= 0:
        context_length += 1
    tokens, attention_mask, position_ids = get_batch(seq[:context_length], device, args)

    counter = context_length - 1 # == len(tokens) - 1
    index = 0 # len(mems)
    if mems is None:
        mems = []
    score = [0] # sum log likelihood for beams

    if args.is_sparse == 2:
        tokenizer = get_tokenizer()
        img_txt_sep = tokenizer.img_tokenizer.num_tokens
        img_indices_bool = (tokens < img_txt_sep)
        txt_indices_bool = (~img_indices_bool)
    elif args.is_sparse == 0:
        txt_indices_bool = img_indices_bool = None
    else:
        raise ValueError('set is_sparse==2 for inference.')

    while counter < (out_seq_length - 1):
        # Now, we want to generate seq[counter + 1]
        # token[:, index: counter+1] are just added.
        
        if index == 0: # first 
            logits, *mems = model(tokens, position_ids, attention_mask, txt_indices_bool, img_indices_bool, is_sparse=args.is_sparse, *mems)
            index = counter
        elif seq[counter + 1] >= 0: # provided
            tokens, mems, score = shrink_beams(tokens, mems, 1, score)
            counter += 1
            tokens = torch.cat((tokens, seq[counter: counter+1].view(tokens.shape[0], 1)), dim=1)
            if args.is_sparse == 2:
                img_indices_bool = (tokens < img_txt_sep)
                txt_indices_bool = (~img_indices_bool)
            continue
        else:
            assert tokens.shape[1] == counter + 1 
            # TODO each time, the feed input cannot be too long (window size), or it will have a discrepcy from sparse training, but this is not very important. 
            tokens, mems, score = shrink_beams(tokens, mems, -seq[counter + 1], score)
            logits, *mems = model(tokens[:, index: ], 
                torch.arange(index, counter + 1, dtype=torch.long, device=tokens.device).unsqueeze(0),
                0, # rebuild in transformers (sep version)
                txt_indices_bool, img_indices_bool, args.is_sparse,
                *mems)
            index = counter
        nb = -seq[counter + 1]
        counter += 1
        index += 1

        logits = logits[:, -1]
        temp = args.temperature# * 0.7 if counter < 1024 * 0.15 else args.temperature
        # TODO since the temperature is crucial, how can we find a good setting?
        logits /= temp
        for invalid_slice in invalid_slices: # forbide to generate other tokens
            logits[..., invalid_slice] = -float('Inf')
        logits = top_k_logits(logits, top_k=args.top_k, top_p=args.top_p)
        log_probs = F.softmax(logits, dim=-1)
        # expand beams
        if nb > 1 and tokens.shape[0] == 1:
            # import pdb; pdb.set_trace()
            tokens = tokens.expand(nb, -1).contiguous()
            mems = [mem.expand(nb, -1, -1) for mem in mems]
            prev = torch.multinomial(log_probs, num_samples=nb, replacement=True)
            score = torch.log(torch.gather(log_probs, dim=1, index=prev)[0]).tolist()
        else:
            assert tokens.shape[0] == nb

            prev = torch.multinomial(log_probs, num_samples=1)
            score_plus = torch.log(torch.gather(log_probs, dim=1, index=prev)[:, 0])
            for idx in range(nb):
                score[idx] += score_plus[idx]
        
        tokens = torch.cat((tokens, prev.view(tokens.shape[0], 1)), dim=1)
        if args.is_sparse == 2: # update indices
            img_indices_bool = (tokens < img_txt_sep)
            txt_indices_bool = (~img_indices_bool)

    output_tokens_list = tokens.view(tokens.shape[0], -1).contiguous()
    return output_tokens_list

def shrink_beams(tokens, mems, nb, score):
    if tokens.shape[0] == nb:
        return tokens, mems, score
    # shrink
    maximum = max(score)
    max_idx = score.index(maximum)
    tokens = tokens[max_idx].unsqueeze(0)
    score = [0]
    new_mems = [mem[max_idx: max_idx + 1] for mem in mems]
    return tokens, new_mems, score

def add_interlacing_beam_marks(seq, nb=12, period=1024):
    assert isinstance(seq, list) or len(seq.shape) == 1
    blk_cnt = 0
    for i in range(len(seq)):
        if seq[i] == -1:
            blk_cnt += 1
            seq[i] = -nb
            if blk_cnt == period:
                nb += (nb % 2) * 2 - 1
                blk_cnt = 0
        else:
            blk_cnt = 0
    
    