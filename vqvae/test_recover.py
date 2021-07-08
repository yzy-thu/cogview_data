import argparse
import os

import torch
from torchvision.utils import save_image
from torchvision.io import read_image
from tqdm import tqdm

from vqvae import VQVAE
import torchvision
from torchvision.transforms import Normalize
from train_vqvae import get_image_dataset_by_name



def load_model(model, checkpoint, device):
    ckpt = torch.load(os.path.join('checkpoint', checkpoint))

    
    if 'args' in ckpt:
        args = ckpt['args']

    if model == 'vqvae':
        model = VQVAE()

        
    if 'model' in ckpt:
        ckpt = ckpt['model']
    
    if list(ckpt.keys())[0].startswith('module.'):
        ckpt = {k[7:]: v for k, v in ckpt.items()}

    model.load_state_dict(ckpt)
    model = model.to(device)
    model.eval()

    return model


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--vqvae', type=str)
    parser.add_argument('--input', type=str, default='test.jpg')
    parser.add_argument('--output', type=str, default='test_recover.jpg')

    args = parser.parse_args()

    model_vqvae = load_model('vqvae', args.vqvae, device)

    # img = read_image(args.input).unsqueeze(0).float().to(device) / 255.
    img, _lb = get_image_dataset_by_name('lsun,church_outdoor_train')[3]
    img = img.unsqueeze(0).to(device)

    resize_module = torchvision.transforms.Compose([
        torchvision.transforms.Resize(64),
        torchvision.transforms.Resize(256),
    ])

    img2 = resize_module(img)

    with torch.no_grad():
        quant_t1, _, id_t1 = model_vqvae.encode(img)

    with torch.no_grad():
        quant_t2, _, id_t2 = model_vqvae.encode(img2)

    decoded_sample1 = model_vqvae.decode(quant_t1)
    decoded_sample2 = model_vqvae.decode(quant_t2)

    out = decoded_sample1.clamp(-1, 1)

    criterion = torch.nn.MSELoss()
    recon_loss = criterion(out, img)
    print('recon_loss: ', recon_loss.item())
    recon_loss = criterion(decoded_sample2, img2)
    print('recon_loss2: ', recon_loss.item())

    out = torch.cat([img], dim=0)

    save_image(out, args.output, normalize=True, range=(0, 1))
