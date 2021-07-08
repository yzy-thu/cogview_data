import os
import pickle
from collections import namedtuple

import torch
from torch.utils.data import Dataset, IterableDataset
from torchvision import datasets
import lmdb


CodeRow = namedtuple('CodeRow', ['top', 'bottom'])


class ImageFileDataset(datasets.ImageFolder):
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        path, _ = self.samples[index]
        dirs, filename = os.path.split(path)
        _, class_name = os.path.split(dirs)
        filename = os.path.join(class_name, filename)

        return sample, target, filename

from unrar import rarfile, unrarlib
from unrar import constants
from unrar.rarfile import _ReadIntoMemory, BadRarFile
from PIL import Image
import PIL
class StreamingRarDataset(IterableDataset):
    def __init__(self, path, transform=None):
        from PIL import ImageFile
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        self.rar = rarfile.RarFile(path)
        self.transform = transform
        def callback_fn(file_buffer, filename):
            try:
                img = Image.open(file_buffer.get_bytes()).convert('RGB')
                dirs, filename = os.path.split(filename)
                filename = filename.split('.')[0]
                if self.transform is not None:
                    img = self.transform(img)
                return img, filename
            except PIL.UnidentifiedImageError:
                print("UnidentifiedImageError")
                return None, None
        self.callback_fn = callback_fn
        # new handle
        self.handle = None
        self.callback_fn = callback_fn

    def __len__(self):
        return len(self.rar.filelist)
    def __next__(self):
        if self.pointer >= len(self.members):
            raise StopIteration()
        if self.handle == None:
            archive = unrarlib.RAROpenArchiveDataEx(
            self.rar.filename, mode=constants.RAR_OM_EXTRACT)
            self.handle = self.rar._open(archive)
        # callback to memory
        self.data_storage = _ReadIntoMemory()
        c_callback = unrarlib.UNRARCALLBACK(self.data_storage._callback)
        unrarlib.RARSetCallback(self.handle, c_callback, 0)
        handle = self.handle
        try:
            rarinfo = self.rar._read_header(handle)
            while rarinfo is not None:
                if rarinfo.filename == self.members[self.pointer]:
                    self.rar._process_current(handle, constants.RAR_TEST)
                    break
                else:
                    self.rar._process_current(handle, constants.RAR_SKIP)
                rarinfo = self.rar._read_header(handle)

            if rarinfo is None:
                self.data_storage = None

        except unrarlib.UnrarException:
            raise BadRarFile("Bad RAR archive data.")

        if self.data_storage is None:
            raise KeyError('There is no item named %r in the archive' % self.members[self.pointer])

        # return file-like object
        ret = self.data_storage
        if self.callback_fn is not None:
            ret = self.callback_fn(ret, self.members[self.pointer])
        self.pointer += 1
        if ret[0] is None:
            return self.__next__()
        return ret

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            self.members = self.rar.namelist()
        else:
            all_members = self.rar.namelist()
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            self.members = [x for i, x in enumerate(all_members) if i % num_workers == worker_id]
        self.pointer = 0
        return self

    def __del__(self):
        self.rar._close(self.handle)


class LMDBDataset(Dataset):
    def __init__(self, path):
        self.env = lmdb.open(
            path,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        if not self.env:
            raise IOError('Cannot open lmdb dataset', path)

        with self.env.begin(write=False) as txn:
            self.length = int(txn.get('length'.encode('utf-8')).decode('utf-8'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = str(index).encode('utf-8')

            row = pickle.loads(txn.get(key))

        return torch.from_numpy(row.top), torch.from_numpy(row.bottom)
