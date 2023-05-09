import torch
import torchvision as tv
from torch.utils.data import Dataset

class TorchDataloader(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input_data = self.data[index][0]
        output_data = self.data[index][1]
        setting_id = self.data[index][2]
        scene_id = self.data[index][3]
        frame_id = self.data[index][4]
        
        return input_data, output_data, setting_id, scene_id, frame_id