import argparse
import pickle

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import lmdb
from tqdm import tqdm

from dataset import ImageFileDataset
from vqvae import VQVAE
from train_vqvae import get_image_dataset_by_name
from torchvision import transforms
from torchvision.transforms.functional import resize

crops = [
    transforms.RandomCrop(256),
    transforms.RandomCrop(128),
    transforms.RandomCrop(64)
]
def sample_patch(img, num):
    ret = []
    for i, n in enumerate(num):
        patches = [crops[i](img) for j in range(n)]
        ret.append(patches)
    return ret

def extract(lmdb_env, lmdb_env_layout, loader, model, device):
    index = 0
    index_layout = 0
    with lmdb_env.begin(write=True) as txn:
        with lmdb_env_layout.begin(write=True) as txn_layout:
            pbar = tqdm(loader)

            for img, _ in pbar:
                img = img.to(device)
                patch_groups = sample_patch(img, [1, 5, 10])

                # =====================   Insert top level tokens   ======================== #
                for patch in patch_groups[0]:
                    _, _, id_t = model.encode(resize(patch, size=128))
                    id_t = id_t.detach().cpu().numpy()
                    for tokens in id_t:
                        txn_layout.put(str(index_layout).encode('utf-8'), pickle.dumps(tokens))
                        index_layout += 1
                # ===================== END OF BLOCK ======================= #
                
                for group in patch_groups[1:]:
                    for patch in group:
                        _, _, id_high = model.encode(resize(patch, size=128))
                        token1 = id_high.detach().cpu().numpy()

                        patch2 = resize(patch, size=patch.shape[-1] // 2)
                        _, _, id_low = model.encode(resize(patch2, size=64))
                        token2 = id_low.detach().cpu().numpy()

                        for top, bottom in zip(token1, token2):
                            row = (top, bottom)
                            txn.put(str(index).encode('utf-8'), pickle.dumps(row))
                            index += 1
                            # pbar.set_description(f'inserted: {index}')

            txn.put('length'.encode('utf-8'), str(index).encode('utf-8'))
            txn_layout.put('length'.encode('utf-8'), str(index_layout).encode('utf-8'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--ckpt', type=str)
    parser.add_argument('--name', type=str)
    parser.add_argument('path', type=str)

    args = parser.parse_args()

    device = 'cuda'

    # transform = transforms.Compose(
    #     [
    #         transforms.Resize(args.size),
    #         transforms.CenterCrop(args.size),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    #     ]
    # )

    # dataset = ImageFileDataset(args.path, transform=transform)
    dataset = get_image_dataset_by_name(args.path)

    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)

    model = VQVAE()
    states = torch.load(args.ckpt)
    if list(states.keys())[0].startswith('module.'):
        states = {k[7:]: v for k, v in states.items()}
    model.load_state_dict(states)
    model = model.to(device)
    model.eval()

    map_size = 100 * 1024 * 1024 * 1024
    env = lmdb.open(args.name, map_size=map_size)
    env2 = lmdb.open(args.name + '_top', map_size=map_size)
    with torch.no_grad():
        extract(env, env2, loader, model, device)
