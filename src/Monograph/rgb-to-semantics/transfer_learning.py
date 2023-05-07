import sys
import torch 
import config
import torchvision as tv
from torchvision import models
from torch.utils.data import DataLoader, random_split
from torchvision.models.segmentation.deeplabv3 import DeepLabHead


sys.path.append("../dataloader/hypersim_pytorch/")
from TorchDataloader import TorchDataloader


"""
TODO:
1. Make sure output adjusts as per our outputs
2. freeze layer to use pretrained weights
3. replace number of outputs to get our outputs (not sure if it's applicable here.)
"""

def prepare_data():
    # # data transformation
    # transforms = tv.transforms.Compose([
    #         tv.transforms.ToTensor(),
    #         tv.transforms.Normalize(mean = [0.485, 0.456, 0.406], 
    #                                 std = [0.229, 0.224, 0.225])])

    # load saved data
    dataset = torch.load(config.INPUT_PATH)

    # split the data into train and test folder
    data_size = len(dataset)
    train_size = int(0.8 * data_size)
    test_size = data_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # generate dataloader object for train and test set
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    return train_loader, test_loader, len(train_dataset), len(test_dataset)

# instantiate pretrained model
def DeepLabV3(out_channels=1):
    # init network
    model = models.segmentation.deeplabv3_resnet101(pretrained=True,
                                                    progress=True)
    
    # updating classing to DeepLabHead for semantic segmentation
    model.classifier = DeepLabHead(2048, out_channels)

    # # freeze all the layers of the network
    # for param in model.parameters():
    #     param.requires_grad = False

    # set model to training mode
    return model

def train_model(epochs=10):
    trainloader, testloader, train_size, test_size = prepare_data()
    model = DeepLabV3()

    # loss and optimizer init
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adadelta(model.parameters(), lr=0.001, rho=0.9, eps=1e-06, weight_decay=0)

    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
 
    # for epochs
    for epoch in range(epochs):
        print("running for epoch:", epoch)
        model.train()
        train_loss = test_loss = 0.0
        for inputs, masks in trainloader:
            inputs = inputs.permute(0, 3, 1, 2)
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
            train_loss += loss.to("cpu")

        model.eval()
        with torch.no_grad():
            for inputs, masks in testloader:
                inputs = inputs.permute(0, 3, 1, 2)
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



model = train_model()


