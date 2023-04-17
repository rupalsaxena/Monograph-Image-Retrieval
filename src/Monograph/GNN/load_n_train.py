"""
Created on Sun Apr 9 2023

@author: Levi Lingsch
"""
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

class GNN():
    def __init__(self, configs):
        self.in_channels = configs['in_channels']
        self.hidden_channels = configs['hidden_channels']
        self.num_layers = configs['num_layers']
        self.out_channels = configs['out_channels']
        self.kernel_size = configs['kernel_size']

        self.gcn = GCN(self.in_channels,
                          self.hidden_channels,
                         self.num_layers,
                          self.out_channels)
        
        self.maxpool = MaxPool1d(self.kernel_size)
        
    def forward(self, x):
        
        x = self.gcn(x)
        # x = self.maxpool(x)
        x = torch.max(x, 0)
        return x

class load_n_train():
    def __init__(self, configs, device, model_name):
        self.device = device
        self.data_source = configs['data_source']

        # training parameters
        self.epochs = configs['epochs']
        self.num_train = configs['num_train']
        self.num_test = configs['num_test']
        self.batch_size = configs['batch_size']
        self.learning_rate = configs['learning_rate']
        self.scheduler_step = configs['scheduler_step']
        self.scheduler_gamma = configs['scheduler_gamma']
        self.pad = False

        # training loss function
        self.margin = configs['triplet_loss_margin']
        self.p = configs['triplet_loss_p']
        self.loss_function = TripletMarginLoss(margin=self.margin, p=self.p)

        # model
        if configs['use_pretrained']:
            self.model = torch.load(f'models/{model_name}')
        else:
            self.in_channels = configs['in_channels']
            self.hidden_channels = configs['hidden_channels']
            self.num_layers = configs['num_layers']
            self.out_channels = configs['out_channels']
            self.model = GCN(self.in_channels, self.hidden_channels, self.num_layers, self.out_channels).to(device)
        # self.model = GCN(configs).to(device)
        self.model_path = model_name
        self.save_model = configs['save_model']

        # training optimizer
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma)


    def call_loader(self, path):
        # given:    a path to the torch.pt file of graphs
        # return:   the test and train loader for the triplet data from the graphloader.py file
        
        if self.data_source == 'ssg':
            loader = ssg_graph_loader(path=path)
            train_triplets = loader.load_triplet_dataset(0, self.num_train, batch_size=self.batch_size, shuffle=False, nyu=True, eig=False, rio=False, g_id=False, ply=False)
            test_triplets = loader.load_triplet_dataset(self.num_train, self.num_train + self.num_test, batch_size=1, shuffle=False, nyu=True, eig=False, rio=False, g_id=False, ply=False)
        elif self.data_source == 'pipeline':
            loader = pipeline_graph_loader(path=path)
            train_triplets = loader.load_triplet_dataset(0, self.num_train, batch_size=self.batch_size, shuffle=False)
            test_triplets = loader.load_triplet_dataset(self.num_train, self.num_train + self.num_test, batch_size=1, shuffle=False)
        
        return train_triplets, test_triplets

    def pad_inputs(self, a, p, n):
        num_nodes = np.max([a.shape[0], p.shape[0], n.shape[0]])
        num_attr = a.shape[1]

        a_x = torch.zeros((num_nodes, num_attr), dtype=torch.float, device=self.device)
        a_x[:a.shape[0], :] = a.float()

        p_x = torch.zeros((num_nodes, num_attr), dtype=torch.float, device=self.device)
        p_x[:p.shape[0], :] = p.float()

        n_x = torch.zeros((num_nodes, num_attr), dtype=torch.float, device=self.device)
        n_x[:n.shape[0], :] = n.float()

        return a_x, p_x, n_x

    def train(self, path):
        # given:    self variables
        # return:   the trained model
        train_loader, test_loader = self.call_loader(path=path)
        self.model.train()
        
        for epoch in range(self.epochs):
            start_epoch = default_timer()   # it's helpful to time this
            # pdb.set_trace()

            train_loss = 0
            test_loss = 0
            num_train_examples = 0
            train_accuracy = 0
            for triplet in train_loader:
                anc = triplet[0].to(self.device)
                pos = triplet[1].to(self.device)
                neg = triplet[2].to(self.device)

                if self.data_source == 'pipeline':
                    anc.edge_index = anc.edge_index - 1
                    pos.edge_index = pos.edge_index - 1
                    neg.edge_index = neg.edge_index - 1

                if self.pad == True:
                    a_x, p_x, n_x = self.pad_inputs(anc.x, pos.x, neg.x)
                    a_out = self.model(a_x, anc.edge_index)
                    p_out = self.model(p_x, pos.edge_index)
                    n_out = self.model(n_x, neg.edge_index)
                else:
                    a_out = self.model(anc.x.float(), anc.edge_index)
                    p_out = self.model(pos.x.float(), pos.edge_index)
                    n_out = self.model(neg.x.float(), neg.edge_index)
                    # a_out = self.model(anc.x, anc.edge_index, edge_attr = anc.edge_attribute)
                    # p_out = self.model(pos.x, pos.edge_index, edge_attr = pos.edge_attribute)
                    # n_out = self.model(neg.x, neg.edge_index, edge_attr = neg.edge_attribute)

                    a_out = torch.max(a_out, 0).values
                    p_out = torch.max(p_out, 0).values
                    n_out = torch.max(n_out, 0).values
                
                num_train_examples += 1

                distnace_to_p = self.compute_distance(a_out, p_out)
                distance_to_n = self.compute_distance(a_out, n_out)

                if distance_to_n > distnace_to_p:
                    train_accuracy += 1


                loss = self.loss_function(a_out, p_out, n_out)
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            with torch.no_grad():
                test_accuracy = 0
                num_test_examples = 0
                for triplet in test_loader:
                    anc = triplet[0].to(self.device)
                    pos = triplet[1].to(self.device)
                    neg = triplet[2].to(self.device)

                    if self.data_source == 'pipeline':
                        anc.edge_index = anc.edge_index - 1
                        pos.edge_index = pos.edge_index - 1
                        neg.edge_index = neg.edge_index - 1

                    # a_out = self.model(a_x, a.edge_index, edge_attr=a.edge_attr)

                    if self.pad == True:
                        a_x, p_x, n_x = self.pad_inputs(anc.x, pos.x, neg.x)
                        a_out = self.model(a_x, anc.edge_index)
                        p_out = self.model(p_x, pos.edge_index)
                        n_out = self.model(n_x, neg.edge_index)
                    else:
                        a_out = self.model(anc.x.float(), anc.edge_index)
                        p_out = self.model(pos.x.float(), pos.edge_index)
                        n_out = self.model(neg.x.float(), neg.edge_index)
                        # a_out = self.model(anc.x, anc.edge_index, edge_attr = anc.edge_attribute)
                        # p_out = self.model(pos.x, pos.edge_index, edge_attr = pos.edge_attribute)
                        # n_out = self.model(neg.x, neg.edge_index, edge_attr = neg.edge_attribute)
                        
                        a_out = torch.max(a_out, 0).values
                        p_out = torch.max(p_out, 0).values
                        n_out = torch.max(n_out, 0).values

                    distnace_to_p = self.compute_distance(a_out, p_out)
                    distance_to_n = self.compute_distance(a_out, n_out)

                    num_test_examples += 1
                    if distance_to_n > distnace_to_p:
                        test_accuracy += 1


                    test_loss += self.loss_function(a_out, p_out, n_out).item()

            stop_epoch = default_timer()
            epoch_time = stop_epoch - start_epoch

            print(epoch, epoch_time, train_loss / num_train_examples, train_accuracy / num_train_examples, test_loss / num_test_examples, test_accuracy / num_test_examples)
            # print(num_train_examples, num_test_examples)
        self.scheduler.step()

        if self.save_model:
            torch.save(self.model, f'models/{self.model_path}')

    def compute_distance(self, anchor_features, comparison_features):
        # given:    two data objects; 1 to be queried against, 1 being compared to the query
        # return:   the distance between the two outputs of the model

        loss = torch.nn.MSELoss(reduction='sum')
        distance = loss(anchor_features, comparison_features)

        return distance

def run_trainer():
    # use the configs to train a GCN, saving it in 'models/'

    train_configs = {
        'learning_rate':   0.001,
        'epochs':          100,
        'batch_size':      1,
        'num_train':       20,
        'num_test':        10,
        'scheduler_step':  10,
        'scheduler_gamma': 0.9,
        'triplet_loss_margin': 1.0,
        'triplet_loss_p':   2,
        'in_channels': -1,
        'hidden_channels': 64,
        'num_layers': 5,
        'out_channels': 64, 
        'kernel_size': 1,
        'use_pretrained': True,
        'data_source':   'pipeline', # pipeline or ssg
        'save_model':   False
    }
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'

    model_name = 'pretrained_on_3dssg'
    trainer = load_n_train(train_configs, device, model_name)
    euler_path = '/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/3dssg/'
    singularity_path = '/mnt/datasets/3dssg/'
    github_path = '../../../data/hypersim_graphs/'
    trainer.train(github_path)

# run_trainer()

##############################################################################################
##############################################################################################
##############################################################################################

class load_pretrained():
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
        
        # features = self.model(data.x, data.edge_index, edge_attr=data.edge_attr.float())
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

                # test_loss += torch.max(distance_to_p - distance_to_n + 1)
                print(f'{distance_to_p.item():.2f}, {distance_to_n.item(): .2f}, {bool(distance_to_p < distance_to_n)}')

            print(test_accuracy / num_test_examples)
    
def run_pretrained():
    # load a pretrained model and compare graphs
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'
    
    # data_loader = graph_loader()
    # anchor = data_loader.load_selected(0, nyu40=True, eigen13=False, rio27=False, global_id=False, ply=True)
    # anchor, comparison_p = data_loader.make_anchor_positive(anchor, offset=100)
    # anchor = anchor.to(device)
    # comparison_p = comparison_p.to(device)
    # comparison_n = data_loader.load_selected(2, nyu40=True, eigen13=False, rio27=False, global_id=False, ply=True).to(device)
    num_train = 29
    num_test = 2
    github_path = '../../../data/hypersim_graphs/'
    loader = pipeline_graph_loader(path=github_path)
    # train_triplets = loader.load_triplet_dataset(0, self.num_train, batch_size=self.batch_size, shuffle=False)
    test_triplets = loader.load_triplet_dataset(num_train, num_train + num_test, batch_size=1, shuffle=False)
        

    model = 'pretrained_on_3dssg'
    pretrained = load_pretrained(f'models/{model}', device)
    pretrained.evaluate_accuracy(test_triplets)



    # distance_a_p = pretrained.compute_distance(anchor, comparison_p)
    # distance_a_n = pretrained.compute_distance(anchor, comparison_n)
    # print(distance_a_p)
    # print(distance_a_n)

# run_pretrained()

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