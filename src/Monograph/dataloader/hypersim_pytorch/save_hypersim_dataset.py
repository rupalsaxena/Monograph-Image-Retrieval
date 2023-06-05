import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

import config
from GenerateHypersimData import GenerateHypersimData
from TorchDataloader import TorchDataloader

# generate hypersim data object
generate = GenerateHypersimData()
data = generate.get_dataset()
print("data ready!")

# # save the dataset object for loading it later
output = config.OUTPUT_PATH 
torch.save(data, output)    
