import os
import config
from utils import *

class GenerateHypersimData:
    def __init__(self):
        self._path = config.HYPERSIM_PATH
        self._settings = config.SETTINGS
        self._purpose = config.PURPOSE
        self._data = []
    
    def get_dataset(self):
        # data formate: input_img, targets (semantic labels/depth labels), setting id, scene id, frame id
        for setting_id in self._settings:
            print(f"running for setting id {setting_id}")
            scenes_info = self.get_scenes_frames_ids(setting_id)
            for scene_id in scenes_info:
                frames = scenes_info[scene_id]
                for frame_id in frames:
                    frame_int = int(frame_id)
                    if frame_int > 20:
                        continue
                    input_img = get_rgb_from_jpg(self._path, setting_id, scene_id, frame_id)
                    if self._purpose == "Depth":
                        targets = get_depth(self._path, setting_id, scene_id, frame_id)
                    elif self._purpose=="Semantic":
                        targets = get_semantic_label(self._path, setting_id, scene_id, frame_id)
                    self._data.append((input_img, targets, setting_id, scene_id, frame_id))
        return self._data

    def get_scenes_frames_ids(self, setting):
        """
        getting scene ids and frame ids from setting id
        """
        foldername = os.path.join(self._path, f'ai_{setting}', "images")
        scene_files = os.listdir(foldername)
        scene_info = {}

        for scene in scene_files:
            if scene.startswith("scene_cam") and scene.endswith("geometry_hdf5"):
                split_scene = scene.split("_")
                scene_id = split_scene[2]

                scene_path = os.path.join(foldername, scene)
                frame_files = os.listdir(scene_path)
                frame_ids = []

                for frame in frame_files:
                    if (
                        (self._purpose=="Depth" and frame.startswith("frame.") and frame.endswith("depth_meters.hdf5")) 
                        or 
                        (self._purpose=="Semantic" and frame.startswith("frame.") and frame.endswith("semantic.hdf5"))
                    ):
                        split_frame = frame.split(".")
                        frame_id = split_frame[1]
                        frame_ids.append(frame_id)

                scene_info[scene_id] = frame_ids
        return scene_info

if __name__ == '__main__':
    obj = GenerateHypersimData()
    data = obj.get_dataset()