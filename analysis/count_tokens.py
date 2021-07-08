
import torch
import argparse
from data_utils import get_tokenizer
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from preprocess.pretokenized_data import make_image_batch
import os
import pandas as pd
import numpy as np

def TSNE_code(model_path):
    args = argparse.Namespace()
    args.img_tokenizer_path = model_path
    args.img_tokenizer_num_tokens = None
    tokenizer = get_tokenizer(args)
    image_tokens = tokenizer.img_tokenizer.model.quantize_t.embed.cpu().numpy()
    X_tsne = TSNE(n_components=2).fit_transform(image_tokens)
    plt.scatter(X_tsne[:,0], X_tsne[:,1])
    plt.savefig(f".{model_path.split('/')[-1]}.png")


def count_code_distribution(model, datasets, name, device, ratio):
    num = [0] * get_tokenizer().img_tokenizer.num_tokens
    for dataset_index, dataset in enumerate(datasets):
        loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
        print(str(dataset) + " index: " + str(dataset_index))
        pbar = tqdm(loader)
        total_num = len(pbar)
        now_num = 0
        for raw_imgs, raw_filenames in pbar:
            now_num += 1
            imgs = []
            for i, filename in enumerate(raw_filenames):
                if filename != "not_a_img":
                    imgs.append(raw_imgs[i])
            imgs = torch.stack(imgs)
            imgs = imgs.to(device)
            codes = make_image_batch(model, imgs)
            for code in codes:
                for token in code:
                    num[token] += 1
            if total_num * ratio <= now_num:
                break
    bin = []
    zero_count = 0
    for i in num:
        if i == 0:
            zero_count += 1
        else:
            bin.append(i)
    # num = sorted(num, reverse=True)
    bin = pd.Series(bin)
    plt.subplot(211)
    hist, bins, _ = plt.hist(bin, bins=10)
    logbins = np.logspace(np.log10(bins[0]), np.log10(bins[-1]), len(bins))
    plt.subplot(212)
    plt.hist(bin, bins=logbins)
    plt.xscale('log')
    print("0 count", zero_count)
    plt.savefig(f"hist_{name}.png")
    # if not os.path.exists(f"hist_{name}"):
    #     os.mkdir(f"hist_{name}")
    # minn = 0
    # plt_num = 0
    # tmp = []
    # for i in num:
    #     if minn == 0:
    #         minn = i
    #     tmp.append(i)
    #     if i/minn > 20:
    #         plt.subplot(plt_num)
    #         plt_num += 1
    #         plt.hist(tmp)
    #         tmp = []
    #         minn = 0
    # if minn != 0:
    #     plt.subplot(plt_num)
    #     plt_num += 1
    # plt.savefig(f"hist_{name}/hist_{name}.png")
    pass


if __name__ == "__main__":
    model_path = "/root/mnt/vqvae_hard_biggerset_011.pt"
    # model_path = "/root/cogview2/pretrained/vqvae/vqvae_hard_018.pt"
    device = "cuda:0"
    torch.cuda.set_device(device)
    count_code_distribution(model_path)
    sys.exit()