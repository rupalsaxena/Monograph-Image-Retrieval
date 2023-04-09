import os
from dataloader import hypersim_config
from generate_scene_graph import config as graph_config
from dataloader.hypersim.dataloader import hypersim_dataloader as dataloader 
from generate_scene_graph.GenerateSceneGraph import GenerateSceneGraph as GSG


class pipeline:
    def __init__(self):
        pass
    
    def run_pipeline(self):
        # get img_data for one setting
        # TODO: do it for different settings
        setting = '013_007'
        input_data = hypersim_config.HYPERSIM_DATAPATH
        dl = dataloader(input_data)
        img_data = dl.get_dataset(setting)

        # # get scene graphs from dataset
        ## TODO: export for each scene seperately and not for entire setting in one file
        graphs = []
        for idx, img_set in enumerate(img_data):
            _gsg = GSG(img_set.depth, img_set.semantic)
            graph = _gsg.get_torch_graph()
            graphs.append(graph)
            print(graphs)

            # TODO: remove this later
            if idx==2:
                break

        # output graphs in files
        output_folder = hypersim_config.HYPERSIM_GRAPHS
        setting_output = os.path.join(output_folder, f'ai_{setting}')
        if not os.path.exists(setting_output):
            os.makedirs(setting_output)
        
        filename = os.path.join(setting_output, f'{setting}_graphs.pt')
        if not graph_config.viz:
            import torch
            torch.save(graphs, filename)

