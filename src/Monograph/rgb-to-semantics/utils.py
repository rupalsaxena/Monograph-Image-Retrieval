import h5py
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

import color_pallete

def output_mask_jpg(mask, path):
    mask = mask.detach().cpu().numpy()
    mask = np.argmax(mask, axis=0)
    mask = mask.squeeze()
    H, W = mask.shape

    out_img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            out_img[i, j] = color_pallete.palette[mask[i, j]]
    out_img = Image.fromarray(out_img)
    out_img.save(path)

def output_label_jpg(mask, path):
    mask = mask.detach().cpu().numpy()
    mask = mask.squeeze()
    H, W = mask.shape

    out_img = np.zeros((H, W, 3), dtype=np.uint8)
    for i in range(H):
        for j in range(W):
            out_img[i, j] = color_pallete.palette[mask[i, j]]
    out_img = Image.fromarray(out_img)
    out_img.save(path)

def output_masks_hdf5(mask, path):
    resize = transforms.Resize((768, 1024), antialias=True)
    mask = torch.argmax(mask, dim=0)
    mask = mask.unsqueeze(0)
    mask = resize(mask)
    mask = mask.detach().cpu().numpy()
    with h5py.File(path, 'w') as f:
        f.create_dataset('dataset', data=mask)

def output_rgb_hdf5(img, path):
    resize = transforms.Resize((768, 1024), antialias=True)
    img = resize(img)
    img = img.detach().cpu().numpy()
    with h5py.File(path, 'w') as f:
        f.create_dataset('dataset', data=img, dtype='f')

def output_label_hdf5(mask, path):
    resize = transforms.Resize((768, 1024), antialias=True)
    mask = resize(mask)
    mask = mask.detach().cpu().numpy()
    with h5py.File(path, 'w') as f:
        f.create_dataset('dataset', data=mask)
    