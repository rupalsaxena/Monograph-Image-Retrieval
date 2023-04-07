import os
import h5py

class ImgObj:
    def __init__(self, setting, scene, frame):
        self.setting = setting
        self.scene = scene
        self.frame = frame
        self.rgb = None
        self.depth = None
        self.semantic = None
    
    def set_rgb(self, rgb):
        self.rgb = rgb
    
    def set_depth(self, depth):
        self.depth = depth

    def set_sematic(self, semantic):
        self.semantic = semantic


class hypersim_dataloader:
    def __init__(self, path):
        # this path will need to be changed for Euler
        self.main_path = path
        
    def get_rgb(self, setting, scene, frame):
        # setting: 000_000
        # scene: 00
        # frame: 0000
        rgb_path = f'ai_{setting}/images/scene_cam_{scene}_final_hdf5/frame.{frame}.color.hdf5'

        rgb_data = h5py.File(self.main_path + rgb_path)['dataset'][:].astype("float32")

        return rgb_data

    def get_semantic(self, setting, scene, frame):
        # setting: 000_000
        # scene: 00
        # frame: 0000
        semantic_path = f'ai_{setting}/images/scene_cam_{scene}_geometry_hdf5/frame.{frame}.semantic.hdf5'

        semantic_data = h5py.File(self.main_path + semantic_path)['dataset'][:].astype("float32")

        return semantic_data

    def get_depth(self, setting, scene, frame):
        # setting: 000_000
        # scene: 00
        # frame: 0000
        depth_path = f'ai_{setting}/images/scene_cam_{scene}_geometry_hdf5/frame.{frame}.depth_meters.hdf5'

        depth_data = h5py.File(self.main_path + depth_path)['dataset'][:].astype("float32")

        return depth_data

    def get_scene_data(self, setting, scene, frame, rgb_flag=True, semantic_flag=True, depth_flag=True):

        imgs = ImgObj(setting, scene, frame)

        if rgb_flag == True:
            rgb = self.get_rgb(setting, scene, frame)
            imgs.set_rgb(rgb)
        
        if semantic_flag == True:
            semantic = self.get_semantic(setting, scene, frame)
            imgs.set_sematic(semantic)

        if depth_flag == True:
            depth = self.get_depth(setting, scene, frame)
            imgs.set_depth(depth)

        return imgs
    
    def get_scenes_frames_ids(self, setting):
        """
        getting scene ids and frame ids from setting id
        """
        foldername = os.path.join(self.main_path, f'ai_{setting}', "images")
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
                    if frame.startswith("frame.") and frame.endswith("depth_meters.hdf5"):
                        split_frame = frame.split(".")
                        frame_id = split_frame[1]
                        frame_ids.append(frame_id)

                scene_info[scene_id] = frame_ids
        return scene_info

    def get_dataset(self, setting):
        """
        getting full dataset from setting id
        """
        dataset = []
        scene_info = self.get_scenes_frames_ids(setting)
        for scene_id in scene_info:
            frames = scene_info[scene_id]
            for frame_id in frames:
                imgs = self.get_scene_data(setting, scene_id, frame_id)
                dataset.append(imgs)
        return dataset

def main():
    # usage 1: get Img object from setting, scene id, frame id
    dataloader = hypersim_dataloader('/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/hypersim/')
    imgs = dataloader.get_scene_data('013_007', '00', '0001', rgb_flag=False)
    print(imgs)

    # usage 2: get full data from setting
    dataloader = hypersim_dataloader('/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/hypersim/')
    dataset = dataloader.get_dataset('013_007')

if __name__ == '__main__':
    main()