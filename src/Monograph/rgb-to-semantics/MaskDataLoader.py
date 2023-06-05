import torch
import torchvision as tv
from torch.utils.data import Dataset

class MaskDataLoader(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        sem_labels = self.data[index][1]
        input_data = self.data[index][0]
        setting_id = self.data[index][2]
        scene_id = self.data[index][3]
        frame_id = self.data[index][4]
        
        return input_data, sem_labels, setting_id, scene_id, frame_id
