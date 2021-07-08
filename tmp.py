import os
import sys
import math
import random
from tqdm import tqdm

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="preprocess args")
    parser.add_argument("--dataset", type=str, default="ali")
    parser.add_argument("--img_tokenizer_path", type=str, default='vqvae_hard_biggerset_011.pt')
    parser.add_argument("--encode_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    print(args)
    img_size = args.encode_size * 8

    # args = argparse.Namespace()
    # args.img_tokenizer_path = 'pretrained/vqvae/vqvae_hard_018.pt'#old path
    # args.img_tokenizer_path = 'pretrained/vqvae/vqvae_hard_biggerset_011.pt'
    # args.img_tokenizer_path = '/root/mnt/vqvae_1epoch_64x64.pt'
    args.img_tokenizer_num_tokens = None

    device = f'cuda:{args.device}'
    torch.cuda.set_device(device)
    name = args.dataset + "_" + args.img_tokenizer_path.split(".")[0] + ".lmdb"
    args.img_tokenizer_path = f"/dataset/fd5061f6/cogview/{args.img_tokenizer_path}"

    datasets = {}
    datasets["bigpic"] = [
        ['/workspace/yzy/bigpic_image_data.json'],
        ['/workspace/yzy/bigpic_imagesFolder.rar'],
        "json_ks"
    ]
    datasets["ali"] = [
        ['/workspace/hwy/cogview2.0/ali/sq_gouhou_white_pict_title_word_256_fulltitle.tsv'],
        ['/workspace/hwy/cogview2.0/ali/ali_white_picts_256.zip'],
        "tsv"
    ]
    txt_files, img_folders, txt_type = datasets[args.dataset]

    os.environ['UNRAR_LIB_PATH'] = '/workspace/yzy/cogview20/libunrar.so'


    from data_utils import get_tokenizer
    tokenizer = get_tokenizer(args)
    model = tokenizer.img_tokenizer.model

    print("finish init vqvae_model")

    from preprocess.preprocess_text_image_data import extract_code,extract_code_super_resolution_patches, extract_code_double_code

    # =====================   Define Imgs   ======================== #
    from preprocess.raw_datasets import H5Dataset, StreamingRarDataset, ZipDoubleCodeDataset, StreamingRarDoubleCodeDataset

    datasets = []
    transform1 = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize([0.79093, 0.76271, 0.75340], [0.30379, 0.32279, 0.32800]),]
    )
    transform2 = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        transforms.Normalize([0.79093, 0.76271, 0.75340], [0.30379, 0.32279, 0.32800]),]
    )
    for img_folder in img_folders:
        if img_folder[-3:] == "rar":
            dataset = StreamingRarDoubleCodeDataset
        elif img_folder[-3:] == "zip":
            dataset = ZipDoubleCodeDataset
        else:
            dataset = None
        dataset = dataset(path=img_folder, transform1=transform1, transform2=transform2,
                                                default_size=img_size)
        datasets.append(dataset)
    print('Finish reading meta-data of dataset.')
    # ===================== END OF BLOCK ======================= #

    # from preprocess import show_recover_results


    # loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=8)
    # loader = iter(loader)
    # samples = []
    # for k in range(8):
    #     x = next(loader)
    #     print(x[1])
    #     x = x[0].to(device)
    #     samples.append(x)
    # samples = torch.cat(samples, dim=0)
    # show_recover_results(model, samples)

    # =====================   Load Text   ======================== #
    if txt_type == "json":
        import json
        txt_list = []
        for txt in txt_files:
            with open(txt, 'r') as fin:
                t = json.load(fin)
                txt_list.extend(list(t.items()))
        tmp = []
        for k, v in tqdm(txt_list):
            tmp.append((v['uniqueKey'], v['cnShortText']))
        text_dict = dict(tmp)
    elif txt_type == "json_ks":
        import json
        txt_list = []
        for txt in txt_files:
            with open(txt, 'r') as fin:
                t = json.load(fin)
            txt_list.extend(t["RECORDS"])
        tmp = []
        for v in tqdm(txt_list):
            tmp.append((v['uniqueKey'], v['cnShortText']))
        text_dict = dict(tmp)
    elif txt_type == "tsv":
        import pandas as pd
        txt_list = []
        for txt in txt_files:
            t = pd.read_csv(txt, sep='\t')
            txt_list.extend(list(t.values))
        tmp = []
        for k, v in tqdm(txt_list):
            tmp.append((str(k), v))
        text_dict = dict(tmp)
    elif txt_type == "json_ks":
        import json

        txt_list = []
        for txt in txt_files:
            with open(txt, 'r') as fin:
                t = json.load(fin)
            txt_list.extend(t["RECORDS"])
        tmp = []
        for v in tqdm(txt_list):
            if 'cnShortText' not in v or len(v['cnShortText']) <= 1 or not is_contain_chinese(v['cnShortText']):
                print("warning: some item do not have cnShortText")
                continue
            tmp.append((v['uniqueKey'], v['cnShortText']))
        text_dict = dict(tmp)
    else:
        des = dataset.h5["input_concat_description"]
        txt_name = dataset.h5["input_name"]
        tmp = []
        for i in tqdm(range(len(des))):
            tmp.append((i, des[i][0].decode("latin-1")+txt_name[i][0].decode("latin-1")))
        text_dict = dict(tmp)
    print('Finish reading texts of dataset.')
    # ===================== END OF BLOCK ======================= #

    # extract_code(model, datasets, text_dict, name, device, txt_type)
    # extract_code_super_resolution_patches(model, datasets, text_dict, name, device, txt_type)
    extract_code_double_code(model, datasets, text_dict, name, device, txt_type)