import os
import argparse
from generate_scene_graph import config
from dataloader.hypersim.dataloader import hypersim_dataloader as dataloader 
from generate_scene_graph.GenerateSceneGraph import GenerateSceneGraph as GSG

def run_pipeline(settings):
    output_folder = config.HYPERSIM_GRAPHS

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for setting in settings:
        print("running for ai_", setting)
        # get img_data from setting
        input_data = config.HYPERSIM_DATAPATH
        dl = dataloader(input_data)
        img_data = dl.get_dataset(setting)

        # get scene graphs from dataset
        graphs = {}
        for img_obj in img_data:
            print(img_obj.scene, img_obj.frame)
            _gsg = GSG(img_obj)

            # skipping generating graphs if in vizualization mode, otherwise generate torch graphs
            if not config.viz:
                graph = _gsg.get_torch_graph()
                scene_id = img_obj.scene
                if scene_id not in graphs.keys():
                    graphs[scene_id] = [graph]
                else:
                    graphs[scene_id].append(graph)

        # skipping saving torch graph if in vizualization mode, otherwise saving torch graphs
        if not config.viz:
            import torch
            for scene_id in graphs:
                filename = "ai_"+setting+"_"+scene_id+"_graphs.pt"
                filename = os.path.join(output_folder, filename)
                torch.save(graphs[scene_id], filename)
                print("graph saved in", f'ai_{setting}_{scene_id}_graphs.pt')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hypersim_setting")
    args = parser.parse_args()
    if args.hypersim_setting is not None:
        settings = [args.hypersim_setting]
    else:
        settings = config.HYPERSIM_SETTINGS
    run_pipeline(settings)

if __name__ == '__main__':
    main()