import torch
import torchvision as tv
from torch.utils.data import Dataset

class TorchDataloader(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform = tv.transforms.Compose([
                      tv.transforms.Resize((256, 320)),
                      tv.transforms.ToTensor(),
                      tv.transforms.Normalize(mean = [0.485, 0.456, 0.406], 
                                              std = [0.229, 0.224, 0.225])])
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input_data = self.data[index][0]
        output_data = self.data[index][1]

        input_tensor = torch.tensor(input_data, dtype=torch.float32)
        output_tensor = torch.tensor(output_data, dtype=torch.float32)

        input_tensor = self.transform(input_tensor)
        output_tensor = self.transform(output_tensor)
        
        return input_tensor, output_tensor