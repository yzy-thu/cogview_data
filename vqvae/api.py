# This is an API file to export an VQVAE/... for tokenization
# Can rewrite the APIs for VQGAN.
# Don't forget to freeze the relavant .py files.

import torch
import math

# production APIs

from .vqvae_zc import VQVAE

def new_model():
    '''Return a New Instance of VQVAE, the same parameters with the pretrained model.
        This is for torch.load().
    '''
    return VQVAE(
        channel=512, n_res_block=0,
        n_res_channel=32, embed_dim=256,
        n_embed=8192, stride=6
    )

# from .vqvae import VQVAE

# def new_model():
#     '''Return a New Instance of VQVAE, the same parameters with the pretrained model.
#         This is for torch.load().
#     '''
#     return VQVAE()

def pretrained_url():
    return 'http://zh-pretrain.oss-cn-zhangjiakou.aliyuncs.com/vqvae_ckpt_set/vqvae_hard_018.pt'

def img2code(model, img):
    '''Convert a batch of img to code
    Args:
        model: The tokenizer model.
        img: [b, c, h, w]
    '''
    with torch.no_grad():
        quant_t1, _, id_t1 = model.encode(img)
    return id_t1.view(img.shape[0], -1) 

def code2img(model, code):
    '''Convert a batch of code to imgs
    Args:
        model: ...
        code: [b, h, w] or [b, h*w] LongTensor
    '''
    if len(code.shape) == 2:
        s = int(math.sqrt(len(code.view(-1))) + 1e-5)
        code = code.view(code.shape[0], s, s)
    with torch.no_grad():
        out = model.decode_code(code)
    return out

