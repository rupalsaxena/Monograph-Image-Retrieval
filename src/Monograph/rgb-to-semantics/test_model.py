import os
import sys
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.append("../dataloader/hypersim_pytorch/")
from TorchDataloader import TorchDataloader

from utils import output_mask
from MaskDataLoader import MaskDataLoader

# TODO: saving hdf5 files part missing

# load model
MODELPATH = "/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/models/deeplabv3.pt"
TESTDATAPATH = "/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/torch_hypersim/test.pt"

# images path
save_format = "jpg" # "hdf5" or "jpg"
main_path = "/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/model_predictions/semantic_seg/"


# create foldername based on format types
if save_format == "jpg":
    rgb_path = os.path.join(main_path, "rgb_imgs_previews")
    semantic_path = os.path.join(main_path, "sem_imgs_previews")
    preds_semantic_path = os.path.join(main_path, "sem_preds_imgs_previews")
elif save_format == "hdf5":
    rgb_path = os.path.join(main_path, "rgb_imgs_hdf5")
    semantic_path = os.path.join(main_path, "sem_imgs_hdf5")
    preds_semantic_path = os.path.join(main_path, "sem_preds_imgs_hdf5")

# create folders if not exist
if not os.path.exists(rgb_path):
   os.makedirs(rgb_path)

if not os.path.exists(semantic_path):
   os.makedirs(semantic_path)

if not os.path.exists(preds_semantic_path):
   os.makedirs(preds_semantic_path)

# create dataloader
print("loading data!")
test_dataset = torch.load(TESTDATAPATH)
test_dataset = MaskDataLoader(test_dataset)
test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)
print("data ready!")

# # run model to predict output

# prep model
model = torch.load(MODELPATH)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
torch.cuda.empty_cache()

# define loss
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

            if save_format == "jpg":
                input_img.save(os.path.join(rgb_path, f"rgb_{setting[i]}_{scene[i]}_{frame[i]}.jpg"))
                output_mask(outputs["out"][i], os.path.join(preds_semantic_path, f"preds_sem_{setting[i]}_{scene[i]}_{frame[i]}.jpg"))
                output_mask(masks[i], os.path.join(semantic_path, f"sem_{setting[i]}_{scene[i]}_{frame[i]}.jpg"))

        # compute loss and other metrics
        loss = loss_fn(outputs['out'], masks)

        # accumulate loss
        test_loss += loss.to("cpu")
    print("loss:", test_loss/len(test_dataset))