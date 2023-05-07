import torch
from torch.utils.data import Dataset

class TorchDataloader(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input_data = self.data[index][0]
        output_data = self.data[index][1]

        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output_tensor = torch.tensor(output_data, dtype=torch.float32)
        
        return input_tensor, output_tensor