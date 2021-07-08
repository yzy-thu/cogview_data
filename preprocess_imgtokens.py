
import lmdb
import mpu
import torch
import argparse
import deepspeed
import json
import os
from arguments import get_args
from data_utils import get_tokenizer
from torch.utils.data import DataLoader
from preprocess.pretokenized_data import *
from torch.utils.data import Dataset
from io import BytesIO
from PIL import Image
import PIL
import base64
import sys

class SougouDataset(Dataset):
    def __init__(self, bin_path, id_path):
        self.bin = open(bin_path, "rb")
        self.ids = open(id_path, "r")
        self.contents = self.ids.readlines()
        self.image_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize([0.79093, 0.76271, 0.75340], [0.30379, 0.32279, 0.32800])
        ])
        self.file = self.bin.read()
        self.bin.close()
    def __len__(self):
        return len(self.contents)

    def __getitem__(self, idx):
        try:
            id, begin, end = self.contents[idx].split('\t')
            if end[-1] == '\n':
                end = end[:-1]
            begin = int(begin)
            end = int(end)
            # self.bin.seek(begin)
            # image = self.bin.read(end-begin+1)
            image = self.file[begin:end+1]
            img1 = Image.open(BytesIO(image)).convert('RGB')
            img1 = self.image_transform(img1)
        except Exception as e:
            print(e)
            return "not id", torch.zeros((3, 256, 256))
        return id, img1
    def __del__(self):
        self.ids.close()
        self.bin.close()


def main():
    id_path = "/dataset/fd5061f6/sougou/ids/"
    args = get_args()
    local_rank = args.local_rank
    world_rank = args.world_rank
    # world_rank = int(os.environ['RANK'])
    # world_size = int(os.environ['WORLD_SIZE'])
    world_size = 6
    device = local_rank % torch.cuda.device_count()
    device = f'cuda:{device}'
    torch.cuda.set_device(device)
    img_tokenizer_path = "/dataset/fd5061f6/cogview/vqvae_hard_biggerset_011.pt"
    bin_pre_path = "/dataset/58e8b681/ordinary/billion/"
    # tokens_path = f"/home/mingding/cogview_data/"
    tokens_path = f"/dataset/fd5061f6/sougou/tokens{world_rank}/"
    args.img_tokenizer_path = img_tokenizer_path
    tokenizer = get_tokenizer(args)
    model = tokenizer.img_tokenizer.model

    total_cnt = world_size * 8
    rank = world_rank * 8 + local_rank
    name_list = os.listdir(id_path)
    name_list = sorted(name_list)

    bin_cnt = len(name_list)
    print("cnt", bin_cnt, total_cnt)
    one_work = (bin_cnt - 1) // total_cnt + 1
    begin_pos = rank * one_work
    end_pos = min((rank + 1) * one_work, bin_cnt)
    print("rank begin end", rank, begin_pos, end_pos)
    for i in range(begin_pos, end_pos):
        bin_name = name_list[i][:-3]
        name = tokens_path + bin_name + ".jsonl"
        if os.path.exists(name):
            continue
        f = open(name, "w")
        dataset = SougouDataset(bin_pre_path + bin_name, id_path + name_list[i])
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=1)
        if local_rank == 0:
            print("now_pos is:", i - begin_pos)
            pbar = tqdm(loader)
        else:
            pbar = loader
        for ids, raw_imgs in pbar:
            imgs0 = []
            new_ids = []
            for j, id in enumerate(ids):
                if id != "not id":
                    imgs0.append(raw_imgs[j].to(device))
                    new_ids.append(id)
            if len(imgs0) == 0:
                continue
            imgs0 = torch.stack(imgs0)
            codes = make_image_batch(model, imgs0)
            for num, code in enumerate(codes):
                f.write(new_ids[num] + "\t" + json.dumps(code.tolist()))
                f.write("\n")
        f.close()
    pass

if __name__ == "__main__":
    main()