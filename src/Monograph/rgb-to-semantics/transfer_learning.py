import sys
import torch 
import torchvision as tv
from torchvision import models
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

import config
sys.path.append("../dataloader/hypersim_pytorch/")
from TorchDataloader import TorchDataloader


def prepare_data(input_path):
    print("preparing dataset")
    # load saved data
    dataset = torch.load(input_path)

    # split the data into train and test folder
    data_size = len(dataset)
    train_size = int(0.8 * data_size)
    test_size = data_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # generate dataloader object for train and test set
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    return train_loader, test_loader, len(train_dataset), len(test_dataset)

# instantiate pretrained model
def DeepLabV3(out_channels=1):
    torch.cuda.empty_cache()

    print("init network")
    # init network
    model = models.segmentation.deeplabv3_resnet50(pretrained=True) #(pretrained=True, progress=True)
    
    # updating classing to DeepLabHead for semantic segmentation
    model.classifier = DeepLabHead(2048, out_channels)

    # set model to training mode
    return model

def train_model(input_path, epochs=10):
    trainloader, testloader, train_size, test_size = prepare_data(input_path)
    model = DeepLabV3()

    print("init loss and optimizer")

    # loss and optimizer init
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
 
    # for epochs
    for epoch in range(epochs):
        print("running for epoch:", epoch)

        model.train()
        train_loss = test_loss = 0.0
        torch.cuda.empty_cache()
        for inputs, masks in trainloader:

            inputs = inputs.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()

            # output from network
            outputs = model(inputs)

            # compute loss
            loss = loss_fn(outputs['out'], masks)

            # perform update
            loss.backward()
            optimizer.step()

            # accumulate loss
            train_loss += loss.to("cpu").detach()

        model.eval()
        torch.cuda.empty_cache()
        with torch.no_grad():
            for inputs, masks in testloader:
                inputs = inputs.to(device)
                masks = masks.to(device)

                # output from network
                outputs = model(inputs)

                # compute loss
                loss = loss_fn(outputs['out'], masks)

                # accumulate loss
                test_loss += loss.to("cpu")

        print(f"\rEpoch {epoch+1}; train: {train_loss/train_size:1.5f}, val: {test_loss/test_size:1.5f}")
    return model

model = train_model(config.INPUT_PATH, epochs=50)
torch.save(model, config.SAVE_MODEL_PATH)
