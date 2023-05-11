import torch
import torchvision as tv
from torch.utils.data import Dataset

class MaskDataLoader(Dataset):
    def __init__(self, data):
        self.data = data
        self.num_classes = 45
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        # convert semantic labels to semantic masks
        sem_labels = self.data[index][1]
        sem_masks = torch.zeros((self.num_classes, 240, 240), dtype=torch.float32)
        for class_index in range(self.num_classes):
            sem_masks[class_index, :, :] = (sem_labels[0, :, :] == class_index).float()

        # failed logic because of -1
        # sem_labels = torch.where(sem_labels == -1, torch.tensor(255), sem_labels)
        # sem_masks = torch.nn.functional.one_hot(sem_labels.squeeze(0), self.num_classes).permute(2, 0, 1).float()

        # other vars
        input_data = self.data[index][0]
        setting_id = self.data[index][2]
        scene_id = self.data[index][3]
        frame_id = self.data[index][4]
        
        return input_data, sem_masks, setting_id, scene_id, frame_id
