import h5py

class hypersim_dataloader:
    def __init__(self, path):
        # this path will need to be changed for Euler
        self.main_path = path
        
    def get_rgb(self, setting, scene, frame):
        # setting: 000_000
        # scene: 00
        # frame: 0000
        rgb_path = f'ai_{setting}/images/scene_cam_{scene}_final_hdf5/frame.{frame}.color.hdf5'

        rgb_data = h5py.File(self.main_path + rgb_path)['dataset']

        return rgb_data

    def get_semantic(self, setting, scene, frame):
        # setting: 000_000
        # scene: 00
        # frame: 0000
        semantic_path = f'ai_{setting}/images/scene_cam_{scene}_geometry_hdf5/frame.{frame}.semantic.hdf5'

        semantic_data = h5py.File(self.main_path + semantic_path)['dataset']

        return semantic_data

    def get_depth(self, setting, scene, frame):
        # setting: 000_000
        # scene: 00
        # frame: 0000
        depth_path = f'ai_{setting}/images/scene_cam_{scene}_geometry_hdf5/frame.{frame}.depth_meters.hdf5'

        depth_data = h5py.File(self.main_path + depth_path)['dataset']

        return depth_data

    def get_scene_data(self, setting, scene, frame, rgb_flag=True, semantic_flag=True, depth_flag=True):

        scene_data = []

        if rgb_flag == True:
            rgb = self.get_rgb(setting, scene, frame)
            scene_data.append(rgb)
        
        if semantic_flag == True:
            semantic = self.get_semantic(setting, scene, frame)
            scene_data.append(semantic)

        if depth_flag == True:
            depth = self.get_depth(setting, scene, frame)
            scene_data.append(depth)

        return scene_data

# example use of the hypersim_dataloader class
dataloader = hypersim_dataloader('/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/hypersim/')
scene_data = dataloader.get_scene_data('013_007', '00', '0000', rgb_flag=False)
print(scene_data)
