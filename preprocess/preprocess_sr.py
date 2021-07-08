
# here put the import lib
import os
import sys
import math
import random
from tqdm import tqdm
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import lmdb
from .pretokenized_data import make_text_image_batch, make_image_text_batch, make_edge_image_batch
import PIL
import timeit

def chose_batch_method(data_type):
    if data_type == "text_img":
        return make_text_image_batch
    if data_type == "img_text":
        return make_image_text_batch
    if data_type == "edge_img":
        return make_edge_image_batch

@torch.no_grad()
def extract_code(model, datasets, text_dict, name, device, data_type, ratio = 1):
    index = 0
    map_size = 1024 * 1024 * 1024 * 1024
    pre_path = '/root/mnt/lmdb/'
    img_cnt = 1
    if data_type == "img_text":
        pre_path += "img2text/"
    if data_type == "edge_img":
        pre_path += "edge_img/"
        img_cnt = 2
    lmdb_env = lmdb.open(pre_path + name + "_sr.lmdb", map_size=map_size, writemap=True)
    batch_method = chose_batch_method(data_type)
    print(pre_path + name+ ".lmdb")
    print(device)
    with lmdb_env.begin(write=True) as txn:
        for dataset_index, dataset in enumerate(datasets):
            loader = DataLoader(dataset, batch_size=96, shuffle=False, num_workers=1)
            print(str(dataset) + " index: " + str(dataset_index))
            pbar = tqdm(loader)
            cnt = 0
            try:
                total_cnt = len(pbar)
            except TypeError:
                total_cnt = -1
            for raw_imgs, raw_filenames in pbar:
                cnt += 1
                if total_cnt != -1 and cnt > total_cnt * ratio:
                    break
                imgs0 = []
                imgs1 = []
                filenames = []
                for i, filename in enumerate(raw_filenames):
                    if filename != "not_a_image" and text_dict.__contains__(filename):
                        if img_cnt == 1:
                            imgs0.append(raw_imgs[i])
                        else:
                            imgs0.append(raw_imgs[0][i])
                            imgs1.append(raw_imgs[1][i])

                        filenames.append(filename)
                    else:
                        print("warning: deleted damaged image")
                if len(imgs0) == 0:
                    break
                imgs0 = torch.stack(imgs0)
                imgs1 = torch.stack(imgs1)
                imgs0 = imgs0.to(device)
                imgs1 = imgs1.to(device)
                try:
                    txts = [text_dict[filename] for filename in filenames]
                    if data_type == "edge_img":
                        codes = batch_method(model, txts, imgs0, imgs1)
                    else:
                        codes = batch_method(model, txts, imgs0)
                    for code in codes:
                        txn.put(str(index).encode('utf-8'), pickle.dumps(code))
                        index += 1
                except KeyError:
                    print("warning: KeyError. The text cannot be find")
                    pass
        txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))
    print("finish")
    # print("start mdb copy")
    # old_path = pre_path + name + ".lmdb/"
    # new_path = pre_path + name + "_after.lmdb/"
    # os.mkdir(new_path)
    # os.system(f"mdb_copy {old_path} {new_path}")
