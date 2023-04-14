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
import sys
sys.path.append('GNN/')
from Adam import Adam
sys.path.append('dataloader/3dssg/')
from graphloader import ssg_graph_loader
sys.path.append('dataloader/')
from pipeline_graphloader import pipeline_graph_loader
from timeit import default_timer

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

        # training loss function
        self.margin = configs['triplet_loss_margin']
        self.p = configs['triplet_loss_p']
        self.loss_function = TripletMarginLoss(margin=self.margin, p=self.p)

        # model
        self.model = GCN(-1, 32, 4, 32).to(device)
        self.model_path = model_name

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
            train_loss = 0
            test_loss = 0
            for triplet in train_loader:
                a = triplet[0].to(self.device)
                p = triplet[1].to(self.device)
                n = triplet[2].to(self.device)

                a_x, p_x, n_x = self.pad_inputs(a.x, p.x, n.x)
                
                # a_out = self.model(a_x, a.edge_index, edge_attr=a.edge_attr)
                # p_out = self.model(p_x, p.edge_index, edge_attr=p.edge_attr)
                # n_out = self.model(n_x, n.edge_index, edge_attr=n.edge_attr)
                a_out = self.model(a_x, a.edge_index)
                p_out = self.model(p_x, p.edge_index)
                n_out = self.model(n_x, n.edge_index)

                loss = self.loss_function(a_out, p_out, n_out)
                train_loss += loss.item()

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            with torch.no_grad():
                for triplet in test_loader:
                    a = triplet[0].to(self.device)
                    p = triplet[1].to(self.device)
                    n = triplet[2].to(self.device)

                    a_x, p_x, n_x = self.pad_inputs(a.x, p.x, n.x)

                    # a_out = self.model(a_x, a.edge_index, edge_attr=a.edge_attr)
                    # p_out = self.model(p_x, p.edge_index, edge_attr=p.edge_attr)
                    # n_out = self.model(n_x, n.edge_index, edge_attr=n.edge_attr)
                    a_out = self.model(a_x, a.edge_index)
                    p_out = self.model(p_x, p.edge_index)
                    n_out = self.model(n_x, n.edge_index)

                    test_loss += self.loss_function(a_out, p_out, n_out).item()

            stop_epoch = default_timer()
            epoch_time = stop_epoch - start_epoch

            print(epoch, epoch_time, train_loss / self.num_train, test_loss / self.num_test)
        self.scheduler.step()
        torch.save(self.model, self.model_path)

def run_trainer():
    # use the configs to train a GCN, saving it in 'models/'

    train_configs = {
        'learning_rate':   0.005,
        'epochs':          50,
        'batch_size':      20,
        'num_train':       20,
        'num_test':        10,
        'scheduler_step':  10,
        'scheduler_gamma': 0.9,
        'triplet_loss_margin': 0.25,
        'triplet_loss_p':   2,
        'data_source':             'pipeline'
    }
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'

    model_name = 'models/pipeline'
    trainer = load_n_train(train_configs, device, model_name)
    euler_path = '/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/3dssg/'
    singularity_path = '/mnt/datasets/3dssg/'
    github_path = '../../../data/hypersim_graphs/'
    trainer.train(github_path)

#run_trainer()

##############################################################################################
##############################################################################################
##############################################################################################

class load_pretrained():
    def __init__(self, path, device):
        self.model = torch.load(f'{path}').to(device)
        self.device = device

    def compute_features(self, data):
        # given:    a Data object
        # return:   the output from the model
        
        features = self.model(data.x, data.edge_index, edge_attr=data.edge_attr.float())
        return features

    def pad_inputs(self, anchor, comparison):
        # give:     the anchor and comparison
        # return:   padded x of same size for both
        num_nodes = np.max([anchor.x.shape[0], comparison.x.shape[0]])
        num_attr = anchor.x.shape[1]

        a_x = torch.zeros((num_nodes, num_attr), dtype=torch.float, device=self.device)
        a_x[:anchor.x.shape[0], :] = anchor.x.float()
        anchor.x = a_x

        c_x = torch.zeros((num_nodes, num_attr), dtype=torch.float, device=self.device)
        c_x[:comparison.x.shape[0], :] = comparison.x.float()
        comparison.x = c_x

        return anchor, comparison

    def compute_distance(self, anchor, comparison):
        # given:    two data objects; 1 to be queried against, 1 being compared to the query
        # return:   the distance between the two outputs of the model

        anchor, comparison = self.pad_inputs(anchor, comparison)

        anchor_features = self.compute_features(anchor)
        comparison_features = self.compute_features(comparison)

        loss = torch.nn.MSELoss(reduction='sum')
        distance = loss(anchor_features, comparison_features)

        return distance
    
def run_pretrained():
    # load a pretrained model and compare graphs
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'
    
    data_loader = graph_loader()
    anchor = data_loader.load_selected(0, nyu40=True, eigen13=False, rio27=False, global_id=False, ply=True)
    anchor, comparison_p = data_loader.make_anchor_positive(anchor, offset=100)
    anchor = anchor.to(device)
    comparison_p = comparison_p.to(device)
    comparison_n = data_loader.load_selected(2, nyu40=True, eigen13=False, rio27=False, global_id=False, ply=True).to(device)

    pretrained = load_pretrained('models/GCN_model_1', device)
    distance_a_p = pretrained.compute_distance(anchor, comparison_p)
    distance_a_n = pretrained.compute_distance(anchor, comparison_n)
    print(distance_a_p)
    print(distance_a_n)

# run_pretrained()