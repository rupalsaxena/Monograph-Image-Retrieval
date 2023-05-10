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


# # test the dataloader object
dataloader = DataLoader(data, batch_size=32, shuffle=True)
for batch_idx, (images, targets, setting, scene, frame) in enumerate(dataloader):
    print(f"Data {batch_idx} shape: {images.shape}, {targets.shape}")

# # save the dataset object for loading it later
output = config.OUTPUT_PATH
torch.save(data, output)
