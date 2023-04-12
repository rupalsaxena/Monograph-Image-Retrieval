import os
from dataloader import hypersim_config
from generate_scene_graph import config as graph_config
from dataloader.hypersim.dataloader import hypersim_dataloader as dataloader 
from generate_scene_graph.GenerateSceneGraph import GenerateSceneGraph as GSG
import pdb


class pipeline:
    def __init__(self):
        pass
    
    def run_pipeline(self):
        settings = hypersim_config.HYPERSIM_SETTINGS
        output_folder = hypersim_config.HYPERSIM_GRAPHS

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        for setting in settings:
            print("running for ai_", setting)
            # get img_data from setting
            input_data = hypersim_config.HYPERSIM_DATAPATH
            dl = dataloader(input_data)
            img_data = dl.get_dataset(setting)

            # get scene graphs from dataset
            graphs = {}
            for img_set in img_data:
                print(img_set.scene, img_set.frame)
                _gsg = GSG(img_set.depth, img_set.semantic)

                # skipping generating graphs if in vizualization mode, otherwise generate torch graphs
                if not graph_config.viz:
                    graph = _gsg.get_torch_graph()
                    scene_id = img_set.scene
                    if scene_id not in graphs.keys():
                        graphs[scene_id] = [graph]
                    else:
                        graphs[scene_id].append(graph)

            # skipping saving torch graph if in vizualization mode, otherwise saving torch graphs
            if not graph_config.viz:
                import torch
                for scene_id in graphs:
                    filename = os.path.join(output_folder, f'ai_{setting}_{scene_id}_graphs.pt')
                    torch.save(graphs[scene_id], filename)
                    print("graph saved in", f'ai_{setting}_{scene_id}_graphs.pt')
