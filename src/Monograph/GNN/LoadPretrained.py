import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric as pyg
from torch.nn import TripletMarginLoss
from torch.nn import MaxPool1d
from torch_geometric.nn.models import GCN
from Adam import Adam
import os
import sys
sys.path.append('../dataloader/3dssg/')
from graphloader import ssg_graph_loader
sys.path.append('../dataloader/')
from pipeline_graphloader import pipeline_graph_loader
from timeit import default_timer
import pdb

class LoadPretrained():
    def __init__(self, path, device):
        self.model = torch.load(f'{path}').to(device)
        self.device = device
        self.data_source = 'pipeline'

        self.loss_function = TripletMarginLoss(margin=1, p=2)

    def compute_features(self, data):
        # given:    a Data object
        # return:   the output from the model

        if self.data_source == 'pipeline':
            edge_index = data.edge_index - 1
            features = self.model(data.x, edge_index)
        else:
            features = self.model(data.x, data.edge_index)
        
        features = torch.max(features, 0).values
        return features

    def compute_distance(self, anchor, comparison):
        # given:    two data objects; 1 to be queried against, 1 being compared to the query
        # return:   the distance between the two outputs of the model

        # anchor, comparison = self.pad_inputs(anchor, comparison)
        anchor_features = self.compute_features(anchor)
        comparison_features = self.compute_features(comparison)

        loss = torch.nn.MSELoss(reduction='sum')
        distance = loss(anchor_features, comparison_features)

        return distance
    
    def evaluate_accuracy(self, test_loader):
        with torch.no_grad():
            test_accuracy = 0
            num_test_examples = 0
            # pdb.set_trace()
            for triplet in test_loader:
                anc = triplet[0].to(self.device)
                pos = triplet[1].to(self.device)
                neg = triplet[2].to(self.device)

                distance_to_p = self.compute_distance(anc, pos)
                distance_to_n = self.compute_distance(anc, neg)

                num_test_examples += 1
                if distance_to_n > distance_to_p:
                    test_accuracy += 1

                print(f'{distance_to_p.item():.2f}, {distance_to_n.item(): .2f}, {bool(distance_to_p < distance_to_n)}')

            print(test_accuracy / num_test_examples)
    
def run_pretrained():
    # load a pretrained model and compare graphs
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'
    
    num_train = 29
    num_test = 2
    github_path = '../../../data/hypersim_graphs/'
    loader = pipeline_graph_loader(path=github_path)
    test_triplets = loader.load_triplet_dataset(num_train, num_train + num_test, batch_size=1, shuffle=False)
        

    model = 'pretrained_on_3dssg'
    pretrained = LoadPretrained(f'models/{model}', device)
    pretrained.evaluate_accuracy(test_triplets)

# run_pretrained()