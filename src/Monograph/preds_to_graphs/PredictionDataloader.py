import os
import config
from ImgObj import ImgObj
from utils import *

class PredictionDataloader:
    def __init__(self, setting):
        self.sem_pred_path = config.SEM_PATH
        self.hypersim_path = config.HYPERSIM_PATH
        self.depth_path = config.DEPTH_PATH
        self.querry_setting = setting
        self.load_data()
    
    def load_data(self):
        self.dataset = []
        pred_files = os.listdir(self.sem_pred_path)

        for pred_file in pred_files:
            _split = pred_file.split("_")
            setting_id = _split[-4] + "_" + _split[-3]
        
            if setting_id == self.querry_setting and _split[-2] == '00':
                frame_id = _split[-1].split(".")[0]
                scene_id = _split[-2]

                img = ImgObj(setting_id, scene_id, frame_id)
                
                semantic_img = get_semantic(self.sem_pred_path, pred_file)
                # depth_img = get_depth(self.hypersim_path, setting_id, scene_id, frame_id)
                depth_img = get_depth_pred(self.depth_path, setting_id, scene_id, frame_id)
                rgb_img = get_rgb(self.hypersim_path, setting_id, scene_id, frame_id)

                img.set_sematic(semantic_img)
                img.set_depth(depth_img)
                img.set_rgb(rgb_img)
            
                self.dataset.append(img)
    
        if len(self.dataset) == 0: print("setting not found in predicted data: ", self.querry_setting)

    def get_dataset(self):
        return self.dataset

# usage
# pd = PredictionDataloader("014_003")
# data = pd.get_dataset()


