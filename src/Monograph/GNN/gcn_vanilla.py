#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 11:24:10 2023

@author: alexanderjurgens
"""
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn import models
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='data/Planetoid', name='Cora')
graph   = dataset[0]

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = pyg.nn.GCNConv(dataset.num_node_features, 16)
        self.conv2 = pyg.nn.GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)
    

def visualize(h, color):
    z = TSNE(n_components=2).fit_transform(h.detach().cpu().numpy())

    plt.figure(figsize=(10,10))
    plt.xticks([])
    plt.yticks([])

    plt.scatter(z[:, 0], z[:, 1], s=70, c=color, cmap="Set2")
    plt.show()

model = GCN()
# num_layers = 3
# model = models.GCN(dataset.num_node_features, TBD , num_layers)
# model = models.DeepGCNLayer()
# embedding_dim = 10
# model = models.LightGCN(dataset.num_node_features, embedding_dim, num_layers)

#Untrained 
model.eval()
out = model(graph)
visualize(out, color=graph.y)

#Start training
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()

def train(data):
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data)  # Perform a single forward pass.
      loss = criterion(out[data.train_mask], data.y[data.train_mask])  # Compute the loss solely based on the training nodes.
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss

def test(data):
      model.eval()
      out = model(data)
      pred = out.argmax(dim=1)  # Use the class with highest probability.
      test_correct = pred[data.test_mask] == data.y[data.test_mask]  # Check against ground-truth labels.
      test_acc = int(test_correct.sum()) / int(data.test_mask.sum())  # Derive ratio of correct predictions.
      return test_acc

for epoch in tqdm(range(1, 101)):
    loss = train(graph)

test_acc = test(graph)
print('Test Accuracy : '+str(test_acc))
