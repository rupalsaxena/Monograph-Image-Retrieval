import os
import h5py
import numpy as np
from torchvision import transforms
import torch

def get_rgb(path, setting, scene, frame):
    rgb_path = f'ai_{setting}/images/scene_cam_{scene}_final_hdf5/frame.{frame}.color.hdf5'
    rgb_data = h5py.File(path + rgb_path)['dataset'][:].astype("float32")
    return rgb_data

def get_depth(path, setting, scene, frame):
    depth_path = f'ai_{setting}/images/scene_cam_{scene}_geometry_hdf5/frame.{frame}.depth_meters.hdf5'
    depth_data = h5py.File(path + depth_path)['dataset'][:].astype("float32")
    return depth_data

def get_depth_pred(path, setting, scene, frame):
    depth_path = f'ai_{setting}/images/scene_cam_{scene}_final_preview/frame.{frame}.depth_meters.pt'
    depth_data = torch.load(path + depth_path)
    return depth_data

def get_semantic(path, filename):
    finalpath = os.path.join(path, filename)
    sem_data = h5py.File(finalpath)['dataset'][:].astype("float32")
    sem_data = np.squeeze(sem_data, axis=0)
    return sem_data