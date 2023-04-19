"""
Created on Sun Apr 9 2023

@author: Levi Lingsch
"""
import torch

class pipeline_features():
    def __init__(self, model, device):
        # self.graph = graph
        self.model = torch.load(f'models/{model}').to(device)

    def get_features(self, graph):
        with torch.no_grad():
            edge_index = graph.edge_index - 1
            features = self.model(graph.x, edge_index)
            features = torch.max(features, 0).values
        
        return features

def run_pipeline_example():
    import os
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'

    path='/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/hypersim_graphs/'
    scene_files = os.listdir(path)
    scene = torch.load(f'{path}{scene_files[0]}')

    graph = scene[0].to(device)
    model = 'pretrained_on_3dssg'
    pipeline = pipeline_features(model, device)
    features = pipeline.get_features(graph)
    print(features)

run_pipeline_example()