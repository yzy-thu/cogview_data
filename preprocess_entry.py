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
from analysis import count_code_distribution
from torchvision.datasets import CocoCaptions
from preprocess.raw_datasets import *
from preprocess.utils import get_image_transforms

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="preprocess args")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--img_tokenizer_path", type=str, default='vqvae_hard_biggerset_011.pt')
    parser.add_argument("--encode_size", type=int, default=32)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--data_type", type=str, default="text_img")# edge_img
    parser.add_argument("--ratio", type=float, default=1)
    args = parser.parse_args()
    print(args)

    # args = argparse.Namespace()
    # args.img_tokenizer_path = 'pretrained/vqvae/vqvae_hard_018.pt'#old path
    # args.img_tokenizer_path = 'pretrained/vqvae/vqvae_hard_biggerset_011.pt'
    # args.img_tokenizer_path = '/root/mnt/vqvae_1epoch_64x64.pt'
    args.img_tokenizer_num_tokens = None

    device = f'cuda:{args.device}'
    torch.cuda.set_device(device)
    name = args.dataset + "_" + args.img_tokenizer_path.split(".")[0]
    args.img_tokenizer_path = f"/dataset/0a40c810/{args.img_tokenizer_path}"

    data_type = args.data_type
    datasets = {}
    datasets["ali"] = [
        ['/root/mnt/sq_gouhou_white_pict_title_word_256_fulltitle.tsv'],
        ['/root/mnt/dingming/ali_white_picts_256.zip'],
        "tsv"
    ]
    datasets["ks3"] = [
        ['/root/mnt/KS3/a_baidu_image_msg_data.json'],
        ['/root/mnt/KS3/downloadImages.rar'],
        "json_ks"
    ]
    datasets["zijian"] = [
        ['/root/mnt/zijian/zj_duomotai_clean_done_data_new.json',
         '/root/mnt/zijian/zj_duomotai_local_server_last_surplus_120w.json'],
        ['/root/mnt/imageFolder_part01.rar',
         '/root/mnt/zijian/imagesFolder_last_surplus_120w.rar'],
        "json"
    ]
    datasets["google"] = [
        ['/root/mnt/google/google_image_message_data.json'],
        ['/root/mnt/google/downloadImage_2020_12_16.rar'],
        "json_ks"
    ]
    datasets["google_part2"] = [
        ['/root/mnt/google/google_image_message_data.json'],
        ['/root/mnt/new_dataset/downloadImage.rar'],
        "json_ks"
    ]
    datasets["shutterstock"] = [
        ['/root/mnt/new_dataset/notTranslate/shutterstock_image_data_translated.json'],
        ['/root/mnt/new_dataset/notTranslate/shutterstock_imagesFolder.rar'],
        "json_ks"
    ]

    # image_net
    image_list = os.listdir("/root/mnt/image_net/train/")
    image_list = ["/root/mnt/image_net/train/"+image_path for image_path in image_list]
    datasets["image_net"] = [
        ['/root/mnt/image_net/infolist.json'],
        image_list,
        "dict"
    ]

    #chinaWebsite
    pre = '/root/mnt/new_dataset/chinaWebsiteImg/'
    datasets["bigpic"] = [
        [pre + 'bigpic_image_data.json'],
        [pre + 'bigpic_imagesFolder.rar'],
        "json_ks"
    ]
    datasets["nipic"] = [
        [pre + 'nipic_image_data.json'],
        [pre + 'nipic_imagesFolder.rar'],
        "json_ks"
    ]
    datasets["sina2"] = [
        [pre + 'sina_other_img_image_data.json'],
        [pre + 'sina_other_img_imagesFolder.rar'],
        "json_ks"
    ]
    datasets["sina1"] = [
        [pre + 'sina_img_image_data.json'],
        [pre + 'sina_img_imagesFolder.rar'],
        "json_ks"
    ]
    datasets["enterdesk"] = [
        [pre + 'enterdesk_image_data.json'],
        [pre + 'enterdesk_imagesFolder.rar'],
        "json_ks"
    ]
    datasets["wy163"] = [
        [pre + 'wy163_img_image_data.json'],
        [pre + 'wy163_img_imagesFolder.rar'],
        "json_ks"
    ]
    datasets["lanrentuku"] = [
        [pre + 'lanrentuku_image_data.json'],
        [pre + 'lanrentuku_imagesFolder.rar'],
        "json_ks"
    ]
    pre = '/root/mnt/new_dataset/chinaWebsiteImg/' + "20210315_data/"
    datasets["51tietu_image_data"] = [
        [pre + "51tietu_image_data.json"],
        [pre + "51tietu_imagesFolder_new.rar"],
        "json_ks"
    ]
    datasets["51yuansu_image_data"] = [
        [pre + "51yuansu_image_data.json"],
        [pre + "51yuansu_imagesFolder.rar"],
        "json_ks"
    ]
    datasets["birdnet_image_data"] = [
        [pre + "birdnet_image_data.json"],
        [pre + "birdnet_imagesFolder.rar"],
        "json_ks"
    ]
    datasets["ppbc_other_image_data"] = [
        [pre + "ppbc_other_image_data.json"],
        [pre + "ppbc_other_imagesFolder.rar"],
        "json_ks"
    ]
    datasets["20210315_data"] = [
        [pre + "51tietu_image_data.json", pre + "51yuansu_image_data.json", pre + "birdnet_image_data.json", pre + "ppbc_other_image_data.json"],
        [pre + "51tietu_imagesFolder.rar", pre + "51yuansu_imagesFolder.rar", pre + "birdnet_imagesFolder.rar", pre + "ppbc_other_imagesFolder.rar"],
        "json_ks"
    ]

    pre = '/root/mnt/new_dataset/chinaWebsiteImg/' + "20210320_data/"
    datasets["adv_bigpic_image_data"] = [
        [pre + "adv_bigpic_image_data.json"],
        [pre +"adv_bigpic_imagesFolder.rar"],
        "json_ks"
    ]
    datasets["ivsky_image_data"] = [
        [pre + "ivsky_image_data.json"],
        [pre + "ivsky_imagesFolder.rar"],
        "json_ks"
    ]

    # datasets["20210320_data"] = [
    #     [pre + "adv_bigpic_image_data.json", pre +"ivsky_image_data.json", pre + "pic_netbian_image_data.json"],
    #     [pre + "adv_bigpic_imagesFolder.rar", pre + "ivsky_imagesFolder.rar",  pre + "pic_netbian_imagesFolder.zip"],
    #     "json_ks"
    # ]

    datasets["new_zijian"]= [
        ["/root/mnt/dingming/data_0323/translated_a_zj_shutterstock_image_data20210319.json"],
        ["/root/mnt/dingming/data_0323/a_zj_shutterstock_imagesFolder_20210319.rar"],
        "json_ks"
    ]

    pre = '/root/mnt/new_dataset/chinaWebsiteImg/' + "20210327_data/"
    datasets["adv2_bigpic_image_data"] = [
        [pre + "adv2_bigpic_image_data.json"],
        [pre +"adv2_bigpic_imagesFolder.rar"],
        "json_ks"
    ]
    datasets["hubeidaily_image_data"] = [
        [pre + "hubeidaily_image_data.json"],
        [pre + "hubeidaily_imagesFolder.rar"],
        "json_ks"
    ]
    datasets["jituwang_image_data"] = [
        [pre + "jituwang_image_data.json"],
        [pre + "jituwang_imagesFolder.rar"],
        "json_ks"
    ]
    datasets["tooopen_image_data"] = [
        [pre + "tooopen_image_data.json"],
        [pre + "tooopen_imagesFolder.rar"],
        "json_ks"
    ]

    #coco

    pre = "/root/mnt/new_dataset/coco_flicker/mscoco_new/mscoco/"
    root = pre + "train2014.zip"
    json_file = pre + "captions_train2014.json"
    coco_dataset = CocoCaptions(root = root, annFile=json_file, transform=get_image_transforms())
    #615361it
    datasets["coco"] = [
        ["/dataset/0a40c810/sq_ccsbu_img_text_t2i_data_coco.tsv"],
        ["/dataset/0a40c810/sq_ccsbu_img_text_t2i_data_coco.tsv"],
        "coco_tsv"
    ]
    # import os
    # total_size = 0
    # for key,value in datasets.items():
    #     dataset_images = value[0]
    #     for image_data in dataset_images:
    #         total_size += os.path.getsize(image_data)
    # total_size = total_size//1024//1024//1024
    #
    # #测试大小
    #
    # exit(0)

    txt_files, img_folders, txt_type = datasets[args.dataset]

    img_size = 256
    # os.environ['UNRAR_LIB_PATH'] = '/usr/local/lib/libunrar.so'


    from data_utils import get_tokenizer
    tokenizer = get_tokenizer(args)
    model = tokenizer.img_tokenizer.model

    print("finish init vqvae_model")

    from preprocess.preprocess_text_image_data import extract_code

    # =====================   Define Imgs   ======================== #

    datasets = []
    for img_folder in img_folders:
        if data_type == "img_text" or data_type == "text_img":
            # if img_folder[-3:] == "rar":
            #     dataset = StreamingRarDataset
            if img_folder[-3:] == "zip":
                dataset = ZipDataset
            elif img_folder[-3:] == "tar":
                dataset = TarDataset
            elif img_folder[-2:] == "h5":
                dataset = H5Dataset
            elif img_folder[-3:] == "tsv":
                dataset = TsvDataset
            else:
                dataset = ImageFileDataset
            dataset = dataset(img_folder, get_image_transforms())
        elif data_type == "edge_img":
            edge_path = "/root/cogview2_data/RCF-pytorch/"
            dataset = EdgeTsvDataset(img_folder, edge_path + args.dataset, [get_image_transforms(128), get_image_transforms(256)])
        print(img_folder)
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
    def is_contain_chinese(check_str):
        for ch in check_str:
            if u'\u4e00' <= ch <= u'\u9fff':
                return True
        return False

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
    elif txt_type == "txt":
        txt_list = []
        for txt in txt_files:
            with open(txt, 'r') as fin:
                lines = fin.readlines()
            for line in lines:
                key, value = line[:-1].split('\t')
                key = key[:-2]
                txt_list.append((key, value))
        text_dict = dict(txt_list)
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
    elif txt_type == "coco_tsv":
        import csv
        csv.field_size_limit(500 * 1024 * 1024)
        txt_list = []
        tmp = []
        max_cnt = 0
        for txt in txt_files:
            with open(txt, "r") as f:
                tsvreader = csv.reader(f, delimiter='\t')
                for line in tqdm(tsvreader):
                    max_cnt += 1
                    if args.data_type == "edge_img" and max_cnt == 100000:
                        break
                    tmp.append((line[0], line[2]))
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
    elif txt_type == "dict":
        import json
        text_dict = {}
        for txt in txt_files:
            with open(txt, "r") as fin:
                t = json.load(fin)
                text_dict.update(t)
    else:
        des = dataset.h5["input_concat_description"]
        txt_name = dataset.h5["input_name"]
        tmp = []
        for i in tqdm(range(len(des))):
            tmp.append((i, des[i][0].decode("latin-1")+txt_name[i][0].decode("latin-1")))
        text_dict = dict(tmp)

    print('Finish reading texts of dataset.')
    # ===================== END OF BLOCK ======================= #



    ratio = args.ratio
    extract_code(model, datasets, text_dict, name, device, data_type, ratio)
    # extract_code_sr(model, datasets, text_dict, name, device, data_type, ratio)
    # from analysis import count_code_distribution
    # ratio = 0.1
    # count_code_distribution(model, datasets, name, device, ratio)




