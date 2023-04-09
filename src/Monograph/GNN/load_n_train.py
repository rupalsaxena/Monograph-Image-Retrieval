"""
Created on Sun Apr 9 2023

@author: Levi Lingsch
"""
import torch
import numpy as np
import torch.nn.functional as F
import torch_geometric as pyg
from torch.nn import TripletMarginLoss
from torch_geometric.nn.models import GCN
from Adam import Adam
import sys
sys.path.append('../dataloader/3dssg/')
from graphloader import *
from timeit import default_timer
import pdb




# class my_GCN(torch.nn.Module):
#     def __init__(self):
#         super().__init__()
#         # self.conv1 = pyg.nn.GCNConv(4, 16)
#         # self.conv2 = pyg.nn.GCNConv(16, 5)
#         self.fc0 = torch.nn.Linear(5, 5)

#     def forward(self, data):
#         x, edge_index = data.x.float(), data.edge_index
#         pdb.set_trace()
#         x = self.conv1(x, edge_index)
#         x = F.relu(x)
#         x = F.dropout(x, training=self.training)
#         x = self.conv2(x, edge_index)
#         x = self.fc0(x)

#         return F.log_softmax(x, dim=1)

class load_n_train():
    def __init__(self, configs):
        # training parameters
        self.epochs = configs['epochs']
        self.num_train = configs['num_train']
        self.num_test = configs['num_test']
        self.batch_size = configs['batch_size']
        self.learning_rate = configs['learning_rate']
        self.scheduler_step = configs['scheduler_step']
        self.scheduler_gamma = configs['scheduler_gamma']

        # training loss function
        self.margin = configs['triplet_loss_margin']
        self.p = configs['triplet_loss_p']
        self.loss_function = TripletMarginLoss(margin=self.margin, p=self.p)

        # model
        self.model = GCN(-1, 32, 3, 16)

        # training optimizer
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma)


    def call_loader(self, path):
        # given:    a path to the torch.pt file of graphs
        # return:   the test and train loader for the triplet data from the graphloader.py file
        loader = graph_loader(path=path)
        train_triplets = loader.load_triplet_dataset(0, self.num_train, batch_size=self.batch_size, shuffle=False, nyu=True, eig=True, rio=True, g_id=False, ply=True)
        test_triplets = loader.load_triplet_dataset(self.num_train, self.num_train + self.num_test, batch_size=self.batch_size, shuffle=False, nyu=True, eig=True, rio=True, g_id=False, ply=True)

        return train_triplets, test_triplets

    def pad_inputs(self, a, p, n):
        num_nodes = np.max([a.shape[0], p.shape[0], n.shape[0]])
        num_attr = a.shape[1]

        a_x = torch.zeros((num_nodes, num_attr), dtype=torch.float)
        a_x[:a.shape[0], :] = a.float()

        p_x = torch.zeros((num_nodes, num_attr), dtype=torch.float)
        p_x[:p.shape[0], :] = p.float()

        n_x = torch.zeros((num_nodes, num_attr), dtype=torch.float)
        n_x[:n.shape[0], :] = n.float()

        return a_x, p_x, n_x

    def train(self):
        # given:    self variables
        # return:   the trained model
        train_loader, test_loader = self.call_loader(path='../../../data/3dssg/')
        
        for epoch in range(self.epochs):
            start_epoch = default_timer()   # it's helpful to time this

            train_loss = 0
            test_loss = 0
            for triplet in train_loader:
                a = triplet[0]
                p = triplet[1]
                n = triplet[2]

                a_x, p_x, n_x = self.pad_inputs(a.x, p.x, n.x)

                a_out = self.model(a_x, a.edge_index, edge_attr=a.edge_attr)
                p_out = self.model(p_x, p.edge_index, edge_attr=p.edge_attr)
                n_out = self.model(n_x, n.edge_index, edge_attr=n.edge_attr)

                loss = self.loss_function(a_out, p_out, n_out)
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            with torch.no_grad():
                for triplet in test_loader:
                    a = triplet[0]
                    p = triplet[1]
                    n = triplet[2]

                    a_x, p_x, n_x = self.pad_inputs(a.x, p.x, n.x)

                    a_out = self.model(a_x, a.edge_index, edge_attr=a.edge_attr)
                    p_out = self.model(p_x, p.edge_index, edge_attr=p.edge_attr)
                    n_out = self.model(n_x, n.edge_index, edge_attr=n.edge_attr)

                    test_loss += self.loss_function(a_out, p_out, n_out).item()

            stop_epoch = default_timer()
            epoch_time = stop_epoch - start_epoch

            print(epoch_time, train_loss, test_loss)
        self.scheduler.step()


def run_it():
    train_configs = {
        'learning_rate':   0.001,
        'epochs':          20,
        'batch_size':      1,
        'num_train':       2,
        'num_test':        2,
        'scheduler_step':  10,
        'scheduler_gamma': 0.5,
        'triplet_loss_margin': 1,
        'triplet_loss_p':   2
    }
    base = load_n_train(train_configs)

    base.train()

run_it()