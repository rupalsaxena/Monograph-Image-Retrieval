import h5py
import torch

def get_rgb(path, setting, scene, frame):
    rgb_path = f'ai_{setting}/images/scene_cam_{scene}_final_hdf5/frame.{frame}.color.hdf5'
    rgb_data = h5py.File(path + rgb_path)['dataset'][:].astype("float32")
    rgb_data = torch.from_numpy(rgb_data).float()
    return rgb_data

def get_semantic(path, setting, scene, frame):
    semantic_path = f'ai_{setting}/images/scene_cam_{scene}_geometry_hdf5/frame.{frame}.semantic.hdf5'
    semantic_data = h5py.File(path + semantic_path)['dataset'][:].astype("float32")
    semantic_data = torch.from_numpy(semantic_data).float()
    return semantic_data

def get_depth(path, setting, scene, frame):
    depth_path = f'ai_{setting}/images/scene_cam_{scene}_geometry_hdf5/frame.{frame}.depth_meters.hdf5'
    depth_data = h5py.File(path + depth_path)['dataset'][:].astype("float32")
    depth_data = torch.from_numpy(depth_data).float()
    return depth_data