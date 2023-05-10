import os
import h5py
import torch
from torchvision import transforms
import torch.nn.functional as F
from PIL import Image

def get_rgb_from_jpg(path, setting, scene, frame):
    rgb_path = f'ai_{setting}/images/scene_cam_{scene}_final_preview/frame.{frame}.color.jpg'
    fullpath = os.path.join(path, rgb_path)
    rgb_img = Image.open(fullpath)
    transform = transforms.Compose([
        transforms.Resize((320, 320)),
        transforms.ToTensor()]) # Convert back to tensor
    rgb_img = transform(rgb_img)
    return rgb_img

def get_semantic_label(path, setting, scene, frame):
    semantic_path = f'ai_{setting}/images/scene_cam_{scene}_geometry_hdf5/frame.{frame}.semantic.hdf5'
    semantic_data = h5py.File(path + semantic_path)['dataset'][:]
    semantic_data = torch.from_numpy(semantic_data).unsqueeze(0)
    semantic_data =  transforms.Resize((320, 320))(semantic_data)

    # # convert semantic label to semantic mask (ignoring -1 values)
    # num_classes = 45
    # mask = torch.zeros((num_classes, 120, 120), dtype=torch.float32)
    # for class_index in range(num_classes):
    #     mask[class_index, :, :] = (semantic_data[0, :, :] == class_index).float()
    
    return semantic_data

def get_semantic(path, setting, scene, frame):
    # deprecated
    semantic_path = f'ai_{setting}/images/scene_cam_{scene}_geometry_hdf5/frame.{frame}.semantic.hdf5'
    semantic_data = h5py.File(path + semantic_path)['dataset'][:].astype("float32")
    semantic_data = torch.from_numpy(semantic_data).float().unsqueeze(0)
    resize = transforms.Resize((200, 200))
    semantic_data = resize(semantic_data)
    return semantic_data

def get_depth(path, setting, scene, frame):
    # untested
    depth_path = f'ai_{setting}/images/scene_cam_{scene}_geometry_hdf5/frame.{frame}.depth_meters.hdf5'
    depth_data = h5py.File(path + depth_path)['dataset'][:].astype("float32")
    depth_data = torch.from_numpy(depth_data).float()
    transform = transforms.Compose([
                    transforms.ToPILImage(), # Convert tensor to PIL image
                    transforms.Resize((200, 200)), # Resize
                    transforms.ToTensor()]) # Convert back to tensor
    depth_data = transform(depth_data)
    return depth_data

def write_img(img, path):
    transform = transforms.Compose([transforms.ToPILImage()])
    input_img = transform(img.permute(2,0,1))
    input_img.save(path)
