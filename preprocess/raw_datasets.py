# -*- encoding: utf-8 -*-
'''
@File    :   raw_datasets.py
@Time    :   2021/01/24 15:31:34
@Author  :   Ming Ding 
@Contact :   dm18@mails.tsinghua.edu.cn
'''

# here put the import lib
import os
import sys
import math
import random
from tqdm import tqdm
import ctypes
import io

import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset, IterableDataset
from torchvision import datasets
# import unrar
from PIL import Image
import timeit
from collections import Iterable
# from unrar import rarfile
# from unrar import unrarlib
# from unrar import constants
# from unrar.rarfile import _ReadIntoMemory, BadRarFile
import zipfile
import PIL
from io import BytesIO
from PIL import Image
import base64


class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        dirs, filename = os.path.split(path)
        filename = filename.split('.')[0]
        return sample, filename

# class RarDataset(Dataset):
#     def __init__(self, path, transform=None):
#         from unrar import rarfile
#         self.rar = rarfile.RarFile(path)
#         self.infos = self.rar.infolist()
#         self.transform = transform
#     def __len__(self):
#         return len(self.infos)
#     def __getitem__(self, idx):
#         target_info = self.infos[idx]
#         img = Image.open(self.rar.open(target_info))
#         dirs, filename = os.path.split(self.infos[idx].filename)
#         filename = filename.split('.')[0]
#         if self.transform is not None:
#             img = self.transform(img)
#         return img, filename


class ZipDataset(Dataset):
    def __init__(self, path, transform=None):
        self.zip = zipfile.ZipFile(path)
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.members = [info for info in self.zip.infolist() if info.filename[-1] != os.sep]
        else:
            all_members = [info for info in self.zip.infolist() if info.filename[-1] != os.sep]
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            self.members = [x for i, x in enumerate(all_members) if i % num_workers == worker_id]
        self.transform = transform
    def __len__(self):
        return len(self.members)
    def __getitem__(self, idx):
        target_info = self.members[idx]
        try:
            img = Image.open(self.zip.open(target_info))
            dirs, filename = os.path.split(self.members[idx].filename)
            filename = filename.split('.')[0]
            if self.transform is not None:
                img = self.transform(img)
            return img, filename
        except (PIL.UnidentifiedImageError, PIL.Image.DecompressionBombError):
            print("UnidentifiedImageError")
            return torch.zeros((3, 256, 256)), "not_a_image"
        except zipfile.BadZipFile as e:
            print(e)
            return torch.zeros((3, 256, 256)), "not_a_image"


import h5py

class H5Dataset(Dataset):
    def __init__(self, path, transform=None):
        self.h5 = h5py.File(path, "r")
        self.images = self.h5["input_image"]
        self.members = None
        self.transform = transform

    def create_members(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.members = self.h5['index'][:]
        else:
            all_members = self.h5['index'][:]
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            self.members = [x for i, x in enumerate(all_members) if i % num_workers == worker_id]

    def __len__(self):
        if self.members is None:
            self.create_members()
        return len(self.members)

    def __getitem__(self, idx):
        if self.members is None:
            self.create_members()
        target_info = self.members[idx]
        try:
            img = Image.fromarray(self.images[target_info][0])
            if self.transform is not None:
                img = self.transform(img)
            return img, int(target_info)
        except(OSError, IndexError):
            print("warning: OSError or IndexError")
            return Image.new('RGB', (256, 256), (255, 255, 255)), -1


import tarfile
class TarDataset(Dataset):
    def __init__(self, path, transform=None):
        self.tar = tarfile.TarFile(path)
        self.members = self.tar.getmembers()
        self.transform = transform

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx):
        target_info = self.members[idx]
        img = Image.open(self.tar.extractfile(target_info)).convert('RGB')
        dirs, filename = os.path.split(self.members[idx].name)
        filename = filename.split('.')[0]
        if self.transform is not None:
            img = self.transform(img)
        return img, filename


class JsonlDataset(IterableDataset):
    def __init__(self, path):
        import jsonlines
        print(path)
        self.jsonl = jsonlines.open(path, "r")
        self.iter = self.jsonl.__iter__()
    def __iter__(self):
        return self
    def __next__(self):
        def xstr(s):
            return '' if s is None else str(s)
        next_file = self.iter.__next__()
        key_list = ["title", "content", "abstract", "q_title", "q-content", "ans-content"]
        txt = ""
        for key in key_list:
            if key in next_file:
                txt +=xstr(next_file[key])
        if "best_answer" in next_file and "content" in next_file["best_answer"]:
            txt += xstr(next_file["best_answer"]["content"])
        if "other_answers" in next_file:
            answer_list = next_file["other_answers"]
            for answer in answer_list:
                if "content" in answer:
                    txt += xstr(answer["content"])
        return txt

import csv
class TsvDataset(IterableDataset):

    def __init__(self, path, transform=None):
        self.f = open(path, "r")
        self.tsvreader = csv.reader(self.f, delimiter='\t')
        self.transform = transform
        def callback_fn(image_base64, id):
            try:
                img = Image.open(BytesIO(base64.urlsafe_b64decode(image_base64))).convert('RGB')
                if self.transform is not None:
                    img = self.transform(img)
                return img, id
            except (PIL.UnidentifiedImageError, PIL.Image.DecompressionBombError):
                print("UnidentifiedImageError")
                return torch.zeros((3, 256, 256)), "not_a_image"
        self.callback_fn = callback_fn
    def __iter__(self):
        def get_next():
            for line in self.tsvreader:
                yield self.callback_fn(line[3], line[0])
        return iter(get_next())
    def __del__(self):
        self.f.close()


class EdgeTsvDataset(IterableDataset):
    def __init__(self, path1, path2, transform=None):
        print(path1)
        print(path2)
        self.f = open(path1, "r")
        self.tsvreader = csv.reader(self.f, delimiter='\t')
        self.transform = transform
        self.path2 = path2
        def callback_fn(image_base64, id):
            try:
                img1 = Image.open(BytesIO(base64.urlsafe_b64decode(image_base64))).convert('RGB')
                img2 = self.path2 + f"/{id}.jpg"
                if not os.path.exists(img2):
                    raise PIL.UnidentifiedImageError("no file exist")
                img2 = Image.open(img2).convert('RGB')
                if self.transform is not None:
                    img1 = self.transform[1](img1)
                    img2 = self.transform[0](img2)
                return (img2, img1), id
            except (PIL.UnidentifiedImageError, PIL.Image.DecompressionBombError) as e:
                print("UnidentifiedImageError", e)
                return (torch.zeros((3, 128, 128)), torch.zeros((3, 256, 256))), "not_a_image"
        self.callback_fn = callback_fn
    def __iter__(self):
        def get_next():
            for line in self.tsvreader:
                yield self.callback_fn(line[3], line[0])
        return iter(get_next())
    def __del__(self):
        self.f.close()

import lmdb
class SougouDataset(Dataset):
    def __init__(self, bin_path, id_path):
        self.bin = open(bin_path, "rb")
        self.ids = open(id_path, "r")
        self.contents = self.ids.readlines()
    def __len__(self):
        return len(self.contents)
    def __getitem__(self, idx):
        begin, end = self.contents[idx]
        self.bin.seek(begin)
        image = self.bin.read(end)
        img1 = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert('RGB')
        return img1
    def __del__(self):
        self.ids.close()
        self.bin.close()


    # def __


# class StreamingRarDataset(IterableDataset):
#     def __init__(self, path, transform=None):
#         from PIL import ImageFile
#         ImageFile.LOAD_TRUNCATED_IMAGES = True
#         print("begin open rar")
#         self.rar = rarfile.RarFile(path)
#         print("finish open rar")
#         self.transform = transform
#         def callback_fn(file_buffer, filename):
#             try:
#                 img = Image.open(file_buffer.get_bytes()).convert('RGB')
#                 dirs, filename = os.path.split(filename)
#                 filename = filename.split('.')[0]
#                 if self.transform is not None:
#                     img = self.transform(img)
#                 return img, filename
#             except (PIL.UnidentifiedImageError, PIL.Image.DecompressionBombError, OSError):
#                 print("UnidentifiedImageError")
#                 return torch.zeros((3, 256, 256)), "not_a_image"
#         self.callback_fn = callback_fn
#         # new handle
#         self.handle = None
#         self.callback_fn = callback_fn
#
#     def __len__(self):
#         return len(self.rar.filelist)
#     def __next__(self):
#         if self.pointer >= len(self.members):
#             raise StopIteration()
#         if self.handle == None:
#             archive = unrarlib.RAROpenArchiveDataEx(
#             self.rar.filename, mode=constants.RAR_OM_EXTRACT)
#             self.handle = self.rar._open(archive)
#         # callback to memory
#         self.data_storage = _ReadIntoMemory()
#         c_callback = unrarlib.UNRARCALLBACK(self.data_storage._callback)
#         unrarlib.RARSetCallback(self.handle, c_callback, 0)
#         handle = self.handle
#         try:
#             rarinfo = self.rar._read_header(handle)
#             while rarinfo is not None:
#                 if rarinfo.filename == self.members[self.pointer]:
#                     self.rar._process_current(handle, constants.RAR_TEST)
#                     break
#                 else:
#                     self.rar._process_current(handle, constants.RAR_SKIP)
#                 rarinfo = self.rar._read_header(handle)
#
#             if rarinfo is None:
#                 self.data_storage = None
#
#         except unrarlib.UnrarException:
#             raise BadRarFile("Bad RAR archive data.")
#
#         if self.data_storage is None:
#             raise KeyError('There is no item named %r in the archive' % self.members[self.pointer])
#
#         # return file-like object
#         ret = self.data_storage
#         if self.callback_fn is not None:
#             ret = self.callback_fn(ret, self.members[self.pointer])
#         self.pointer += 1
#         return ret
#
#     def __iter__(self):
#         worker_info = torch.utils.data.get_worker_info()
#         if worker_info is None:
#             self.members = self.rar.namelist()
#         else:
#             all_members = self.rar.namelist()
#             num_workers = worker_info.num_workers
#             worker_id = worker_info.id
#             self.members = [x for i, x in enumerate(all_members) if i % num_workers == worker_id]
#         self.pointer = 0
#         return self
#
#     def __del__(self):
#         self.rar._close(self.handle)


