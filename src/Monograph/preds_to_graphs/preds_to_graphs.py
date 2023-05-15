import os
import sys
import argparse
import config
import torch
from PredictionDataloader import PredictionDataloader as dataloader
sys.path.append("../generate_scene_graph/")
from GenerateSceneGraph import GenerateSceneGraph as GSG


def run_pipeline(settings):
    output_path = config.GRAPH_PATH
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for setting in settings:
        dl = dataloader(setting)
        img_set = dl.get_dataset()

        graphs = {}
        for img_obj in img_set:
            print(img_obj.scene, img_obj.frame)
            _gsg = GSG(img_obj)

            graph = _gsg.get_torch_graph()
            scene_id = img_obj.scene
            if scene_id not in graphs.keys():
                graphs[scene_id] = [graph]
            else:
                graphs[scene_id].append(graph)
            

        for scene_id in graphs:
            filename = "ai_"+setting+"_"+scene_id+"_graphs.pt"
            filename = os.path.join(output_path, filename)
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