"""
Created on Sun Apr 9 2023

@author: Levi Lingsch
"""
import torch

class PipelineFeatures():
    def __init__(self, model):
        # self.graph = graph
        if torch.cuda.is_available():
            self.device='cuda:0'
        else:
            self.device='cpu'

        self.model = torch.load(model).to(self.device)

    def get_features(self, graph):
        graph.to(self.device)
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
    model = 'models/pretrained_on_3dssg'
    pipeline = pipeline_features(model, device)
    features = pipeline.get_features(graph)
    print(features)

# run_pipeline_example()