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

# using torch dataloader, create a dataloader object using data
dataset = TorchDataloader(data)
print("dataset ready!")
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# experimentation: remove this
#transform = transforms.Compose([transforms.ToPILImage()])

# # test the dataloader object
# for batch_idx, (images, targets) in enumerate(dataloader):
#     print(f"Data {batch_idx} shape: {images.shape}")

    # experimentation: remove this
    # for i in range(len(images)):
    #     input_img = transform(images[i])
    #     mask_img = transform(targets[i])
    #     input_img.save(f'images/rgb_{batch_idx * dataloader.batch_size + i}.jpg')
    #     mask_img.save(f'images/semantic_{batch_idx * dataloader.batch_size + i}.jpg')

# save the dataset object for loading it later
output = config.OUTPUT_PATH
torch.save(dataset, output)
