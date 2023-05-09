# import
import sys
import torch
sys.path.append("../dataloader/hypersim_pytorch/")
from TorchDataloader import TorchDataloader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from utils import output_mask
# load model
MODELPATH = "/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/models/deeplabv3.pt"
TESTDATAPATH = "/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/torch_hypersim/HypersimSemanticDataset_Testset.pt"

# create dataloader
print("loading data!")
test_dataset = torch.load(TESTDATAPATH)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
print("data ready!")
# # run model to predict output

# prep model
model = torch.load(MODELPATH)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
torch.cuda.empty_cache()

loss_fn = torch.nn.MSELoss()

# prepare tranform for saving images
transform = transforms.Compose([transforms.ToPILImage()])


print("ready to test")
test_loss = 0
# run on dataloader
with torch.no_grad():
    for batch_idx, data in enumerate(test_loader):
        inputs = data[0]
        masks = data[1]
        setting = data[2]
        scene = data[3]
        frame = data[4]

        inputs = inputs.to(device)
        masks = masks.to(device)

        # output from network
        outputs = model(inputs)
        for i in range(len(inputs)):
            input_img = transform(inputs[i])
            input_img.save(f'predicted_images/rgb_{setting[i]}_{scene[i]}_{frame[i]}.jpg')
            output_mask(outputs["out"][i], f'predicted_images/predicted_{setting[i]}_{scene[i]}_{frame[i]}.jpg')
            output_mask(masks[i], f'predicted_images/semantic_{setting[i]}_{scene[i]}_{frame[i]}.jpg')

        # compute loss and other metrics
        loss = loss_fn(outputs['out'], masks)

        # accumulate loss
        test_loss += loss.to("cpu")
    print("loss:", test_loss/len(test_dataset))