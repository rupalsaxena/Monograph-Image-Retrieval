import sys
import torch 
import numpy as np
import torchvision as tv
from torchvision import models
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from sklearn.metrics import f1_score, roc_auc_score

import config
from MaskDataLoader import MaskDataLoader
sys.path.append("../dataloader/hypersim_pytorch/")
from TorchDataloader import TorchDataloader


def prepare_data(input_path):
    print("preparing dataset")
    # load saved data
    dataset = torch.load(input_path)
    dataset = MaskDataLoader(dataset)

    # split the data into train and test folder
    data_size = len(dataset)
    train_size = int(0.9 * data_size)
    test_size = data_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # # generate dataloader object for train and test set
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, drop_last=True)

    # print
    print("len of train dataset:", len(train_dataset))
    print("len of test dataset:", len(test_dataset))

    return train_loader, test_loader, len(train_dataset), len(test_dataset)

# instantiate pretrained model
def DeepLabV3(out_channels=46):
    torch.cuda.empty_cache()

    print("init network")
    # init network
    model = models.segmentation.deeplabv3_resnet50(pretrained=True, progress=True)

    # updating classing to DeepLabHead for semantic segmentation
    model.classifier = DeepLabHead(2048, out_channels)

    # set model to training mode
    return model

def train_model(input_path, epochs=10):
    trainloader, testloader, train_size, test_size = prepare_data(input_path)
    model = DeepLabV3()

    print("init loss and optimizer")

    # loss and optimizer init
    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
 
    # for epochs
    torch.cuda.empty_cache()
    for epoch in range(epochs):
        print("running for epoch:", epoch)
        model.train()
        train_loss = test_loss = 0.0
        b_f1score = []
        b_auc_score = []

        torch.cuda.empty_cache()
        for data in trainloader:
            inputs = data[0]
            masks = data[1].squeeze().long()

            inputs = inputs.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            # output from network
            outputs = model(inputs)

            # compute loss
            loss = loss_fn(outputs["out"], masks)
        
            # perform update
            loss.backward()
            optimizer.step()

            # accumulate loss
            train_loss += loss.to("cpu").detach()
        
        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for data in testloader:
                inputs = data[0]
                masks = data[1].squeeze().long()

                inputs = inputs.to(device)
                masks = masks.to(device)

                # output from network
                outputs = model(inputs)

                # compute loss
                loss = loss_fn(outputs['out'], masks)

                # accumulate loss
                test_loss += loss.to("cpu").detach()

        print(f"\rEpoch {epoch+1}")
        print(f"train loss: {train_loss/train_size:1.5f}") 
        print(f"val loss: {test_loss/test_size:1.5f}")
    return model

model = train_model(config.INPUT_PATH, epochs=30)
torch.save(model, config.SAVE_MODEL_PATH)
