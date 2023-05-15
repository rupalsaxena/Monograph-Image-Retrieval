import torch
import torchvision as tv
from torch.utils.data import Dataset

class MaskDataLoader(Dataset):
    def __init__(self, data):
        self.data = data
        # self.num_classes = 46 # total 45 classes but 1 class is -1
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # convert semantic labels to semantic masks
        sem_labels = self.data[index][1]
        # sem_labels[sem_labels==-1]=45
        # sem_masks = torch.zeros((self.num_classes, 320, 420), dtype=torch.float32)
        # for class_index in range(self.num_classes):
        #     sem_masks[class_index, :, :] = (sem_labels[0, :, :] == class_index).float()

        # other vars
        input_data = self.data[index][0]
        setting_id = self.data[index][2]
        scene_id = self.data[index][3]
        frame_id = self.data[index][4]
        
        return input_data, sem_labels, setting_id, scene_id, frame_id
