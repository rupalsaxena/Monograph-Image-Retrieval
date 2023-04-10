import pickle
import numpy as np
import pdb
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class graph_loader():
    def __init__(self,path='/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/3dssg/'):
        # graph_file = open(path+'graph.npy', "rb")
        # self.data = pickle.load(graph_file)
        self.data = torch.load(f'{path}torch_graph.pt')

        # ["nyu40", "eigen13", "rio27", "global_id", "ply_color"]
    def load_selected(self, idx, nyu40=True, eigen13=True, rio27=True, global_id=True, ply=True):
        # given:    index of graph
        # return:   torch_geometric.Data object at that index with selected attributes
        assert idx < len(self.data), f'Index {idx} out of range for the given graph.'
        
        attributes = []

        if nyu40:
            attributes.append(0)
        if eigen13:
            attributes.append(1)
        if rio27:
            attributes.append(2)
        if global_id:
            attributes.append(3)
        if ply:
            attributes.append(4)
        selected_attributes = torch.tensor(attributes).int()
        
        x = torch.index_select(self.data[idx].x, 1, selected_attributes)

        selected_graph = Data(x=x, edge_index=self.data[idx].edge_index, edge_attr=self.data[idx].edge_attr)

        return selected_graph
    
    def load_index(self, idx):
        # given:    index of graph
        # return:   torch_geometric.Data object at that index
        assert idx < len(self.data), f'Index {idx} out of range for the given graph.'

        return self.data[idx]
    
    def make_anchor_positive(self, graph, offset=50):
        # given:    a graph and the data
        # return:   an achor, a positive, and a negative
        num_edges = graph.edge_index.shape[-1]//2
        random_indices = torch.randperm(num_edges)

        anchor_x = graph.x
        anchor_edge_index = torch.index_select(graph.edge_index, 1, random_indices[:num_edges//2])
        anchor_edge_attr = torch.index_select(graph.edge_attr, 0, random_indices[:num_edges//2])
        anchor = Data(x=anchor_x, edge_index=anchor_edge_index, edge_attr=anchor_edge_attr)

        positive_x = graph.x
        positive_edge_index = torch.index_select(graph.edge_index, 1, random_indices[offset:offset+num_edges//2])
        positive_edge_attr = torch.index_select(graph.edge_attr, 0, random_indices[offset:offset+num_edges//2])
        positive = Data(x=positive_x, edge_index=positive_edge_index, edge_attr=positive_edge_attr)

        return anchor, positive
    
    def make_negative(self, graph):
        # given:    a graph and the data
        # return:   an achor, a positive, and a negative
        num_edges = graph.edge_index.shape[-1]
        random_indices = torch.randperm(num_edges)

        negative_x = graph.x
        negative_edge_index = torch.index_select(graph.edge_index, 1, random_indices[:num_edges//2])
        negative_edge_attr = torch.index_select(graph.edge_attr, 0, random_indices[:num_edges//2])
        negative = Data(x=negative_x, edge_index=negative_edge_index, edge_attr=negative_edge_attr)

        return negative

    def load_triplet_dataset(self, start, stop, batch_size=1, shuffle=False, nyu=True, eig=True, rio=True, g_id=False, ply=True):
        # given:    a range of graphs
        # return:   a dataloader containing 
        data_set = []
        for index in range(start, stop):
            graph = self.load_selected(index, nyu40=nyu, eigen13=eig, rio27=rio, global_id=g_id, ply=ply)
            anchor, positive = self.make_anchor_positive(graph)
            graph = self.load_selected(index-1, nyu40=nyu, eigen13=eig, rio27=rio, global_id=g_id, ply=ply)
            negative = self.make_negative(graph)

            triplet = [anchor, positive, negative]
            data_set.append(triplet)

        return DataLoader(data_set, batch_size=batch_size, shuffle=shuffle)

    def load_standard_dataset(self, batchsize, start, stop):
        # given:    a start and stop for a range of values
        # return:   a torch_geometric.Dataset object for that range
        return DataLoader(self.data[start:stop], batch_size=batchsize)
       
def run_example():
    start = 0
    stop = 3
    shuffle = True
    batch_size = 1

    p = graph_loader(path='../../../../data/3dssg/')

    p.load_selected(0, global_id=False)
    test_loader = p.load_triplet_dataset(start, stop, batch_size=batch_size, shuffle=shuffle, nyu=True, eig=True, rio=True, g_id=False, ply=True)
    pdb.set_trace()
    for triplet in test_loader:
        anchor = triplet[0]
        positive = triplet[1]
        negative = triplet[2]
        
        # forward pass
        # calculate loss
        # back propagate

# run_example()