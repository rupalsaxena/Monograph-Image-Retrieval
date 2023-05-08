import torch
import torchvision as tv
from torch.utils.data import Dataset

class TorchDataloader(Dataset):
    def __init__(self, data):
        self.data = data
        self.transform_inputs = tv.transforms.Compose([
                    tv.transforms.ToPILImage(), # Convert tensor to PIL image
                    tv.transforms.Resize((260, 320)), # Resize to 256x256
                    tv.transforms.ToTensor(), # Convert back to tensor
                    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

        self.transform_outputs = tv.transforms.Compose([
                    tv.transforms.ToPILImage(), # Convert tensor to PIL image
                    tv.transforms.Resize((260, 320)), # Resize to 256x256
                    tv.transforms.ToTensor()]) # Convert back to tensor
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        input_data = self.data[index][0].permute(2,0,1)
        output_data = self.data[index][1].unsqueeze(2).permute(2,0,1)

        input_tensor = self.transform_inputs(input_data)
        output_tensor = self.transform_outputs(output_data)
        
        return input_tensor, output_tensor