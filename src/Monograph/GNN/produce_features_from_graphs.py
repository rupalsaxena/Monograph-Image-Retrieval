import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn.models import GCN
from load_n_train import load_pretrained
import sys
sys.path.append('../dataloader/3dssg/')
from graphloader import *
from timeit import default_timer
import pdb

def pad_inputs(anchor, comparison):
    # give:     the anchor and comparison
    # return:   padded x of same size for both
    num_nodes = np.max([anchor.x.shape[0], comparison.x.shape[0], 30])
    num_attr = anchor.x.shape[1]
    a_x = torch.zeros((num_nodes, num_attr), dtype=torch.float)
    a_x[:anchor.x.shape[0], :] = anchor.x.float()
    anchor.x = a_x
    c_x = torch.zeros((num_nodes, num_attr), dtype=torch.float)
    c_x[:comparison.x.shape[0], :] = comparison.x.float()
    comparison.x = c_x
    return anchor, comparison

import os

files = os.scandir('../../../data/hypersim_graphs/')
file_names = os.listdir('../../../data/hypersim_graphs/')
graph = torch.load(f'../../../data/hypersim_graphs/{file_names[0]}')
graph_2 = torch.load(f'../../../data/hypersim_graphs/{file_names[1]}')

model = torch.load('models/GCN_model_1', map_location=torch.device('cpu'))

loss = torch.nn.MSELoss(reduction='sum')

anc, pos = pad_inputs(graph[0], graph[1])
pdb.set_trace()
out_1 = model(anc.x, anc.edge_index)
out_2 = model(pos.x, pos.edge_index)
distance = loss(out_1, out_2)

a, n = pad_inputs(graph[0], graph_2[0])
out_3 = model(anc.x, anc.edge_index)
out_4 = model(neg.x, neg.edge_index)
distance = loss(out_3, out_4)
