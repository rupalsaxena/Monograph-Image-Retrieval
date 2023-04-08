from dataloader import hypersim_config
from dataloader.hypersim.dataloader import hypersim_dataloader as dataloader 
from generate_scene_graph.GenerateSceneGraph import GenerateSceneGraph as GSG


class pipeline:
    def __init__(self):
        pass
    
    def run_pipeline(self):
        # get img_data
        datapath = hypersim_config.HYPERSIM_DATAPATH
        dl = dataloader(datapath)
        img_data = dl.get_dataset('013_007')

        # # get scene graphs from dataset
        graph_data = []
        for img_set in img_data:
            _gsg = GSG(img_set.depth, img_set.semantic)
            graph = _gsg.get_graph()
            graph_data.append(graph)
            print(graph_data)
