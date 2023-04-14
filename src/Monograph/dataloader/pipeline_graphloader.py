import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import os
import pdb

class pipeline_graph_loader():
    def __init__(self, path='/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/hypersim_graphs/'):
        self.scene_files = os.listdir(path)
        self.scenes = []
        
        for scene in self.scene_files:
            self.scenes.append(torch.load(f'{path}{scene}'))
        

    def load_triplet_dataset(self, start, stop, batch_size=1, shuffle=False):
        # given:    a range of graphs
        # return:   a dataloader containing triplets
        num_scenes = stop - start

        data_set = []
        scenes_set = []
        for scene_idx in range(start, stop):
            current_graphs = self.scenes[scene_idx][:5]
            num_graphs = len(current_graphs)

            
            for graph_idx in range(num_graphs):
                anchor = current_graphs[graph_idx]

                graphs = torch.arange(num_graphs).float()
                weights = torch.ones_like(graphs)
                weights[graph_idx] = 0
                positive_idx = graphs[torch.multinomial(weights, num_samples=1)].int()
                positive = current_graphs[positive_idx.item()]

                negative, negative_scene = self.make_negative(scene_idx, start, stop, 1)
                
                triplet = [anchor, positive, negative]

                data_set.append(triplet)
                scenes_set.append([scene_idx, negative_scene])

        return DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)

    def make_negative(self, idx, start, stop, num_graphs):
        # given:    an index of a scene
        # return:   a set of random graphs from any other scenes

        # want to select a random scene excluding our current anchor-positive scene
        # num_scenes = stop - start
        scenes = torch.arange(stop).float()
        weights = torch.ones_like(scenes)
        weights[:start] = 0
        weights[idx] = 0
        
        # need to make 'num_graphs' number of negatives
        # choose some random scene
        scene_idx = torch.multinomial(weights, num_samples=num_graphs)
        random_scene = scenes[scene_idx].int()
        
        # choose some random graph from that scene
        random_scene = self.scenes[random_scene.item()]
        random_graph_idx = np.random.choice(len(random_scene))
        negative = random_scene[random_graph_idx]

        return negative, scene_idx.item()

def run_example():
    start=0
    stop=15
    path = '../../../data/hypersim_graphs/'
    
    loader = pipeline_graph_loader(path=path)
    loader.load_triplet_dataset(start, stop)

# run_example()