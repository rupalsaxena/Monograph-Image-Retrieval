import os
import torch
from torch.utils.data import DataLoader

import config
from GenerateHypersimData import GenerateHypersimData
from TorchDataloader import TorchDataloader

# generate hypersim data object
generate = GenerateHypersimData()
data = generate.get_dataset()
print("data ready!")

# using torch dataloader, create a dataloader object using data
dataset = TorchDataloader(data)
print("dataset ready!")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# # test the dataloader object
batch = next(iter(dataloader))
for i, data in enumerate(batch):
    print(f"Data {i} shape: {data.shape}")

# save the dataset object for loading it later
output = config.OUTPUT_PATH
pathname = os.path.join(output, f"Hypersim{config.PURPOSE}Dataset_Testset.pt")
torch.save(dataset, pathname)
