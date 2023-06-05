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
import matplotlib.pyplot as plt
from Adam import Adam
import os
import sys
sys.path.append('../dataloader/3dssg/')
from graphloader import ssg_graph_loader
sys.path.append('../dataloader/')
from pipeline_graphloader import pipeline_graph_loader
from timeit import default_timer
import pdb

class LoadAndTrain():
    def __init__(self, configs, device, model_name):
        self.device = device
        self.data_source = configs['data_source']
        self.configs = configs
        # training parameters
        self.epochs = configs['epochs']
        self.num_train = configs['num_train']
        self.num_test = configs['num_test']
        self.batch_size = configs['batch_size']
        self.learning_rate = configs['learning_rate']
        self.scheduler_step = configs['scheduler_step']
        self.scheduler_gamma = configs['scheduler_gamma']
        self.threshold = configs['threshold']

        # training loss function
        self.margin = configs['triplet_loss_margin']
        self.p = configs['triplet_loss_p']
        self.loss_function = TripletMarginLoss(margin=self.margin, p=self.p)

        # model
        if configs['use_pretrained']:
            self.model = torch.load(f'models/{model_name}')
            self.in_channels = configs['in_channels']
            self.hidden_channels = configs['hidden_channels']
            self.num_layers = configs['num_layers']
            self.out_channels = configs['out_channels']
        else:
            self.in_channels = configs['in_channels']
            self.hidden_channels = configs['hidden_channels']
            self.num_layers = configs['num_layers']
            self.out_channels = configs['out_channels']
            self.model = GCN(self.in_channels, self.hidden_channels, self.num_layers, self.out_channels).to(device)
        # self.model = GCN(configs).to(device)

        self.model_path = model_name
        self.save_model = configs['save_model']
        self.training_figure = configs['training_figure']

        # training optimizer
        self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.scheduler_step, gamma=self.scheduler_gamma)


    def call_loader(self, path):
        # given:    a path to the torch.pt file of graphs
        # return:   the test and train loader for the triplet data from the graphloader.py file
        
        if self.data_source == 'ssg':
            loader = ssg_graph_loader(path=path)
            train_triplets = loader.load_triplet_dataset(0, self.num_train, batch_size=self.batch_size, shuffle=False, nyu=True, eig=False, rio=False, g_id=False, ply=True)
            test_triplets = loader.load_triplet_dataset(self.num_train, self.num_train + self.num_test, batch_size=1, shuffle=False, nyu=True, eig=False, rio=False, g_id=False, ply=True)
        elif self.data_source == 'pipeline':
            loader = pipeline_graph_loader(self.threshold, path=path)
            train_triplets = loader.load_triplet_dataset(0, self.num_train, batch_size=self.batch_size, shuffle=False)
            test_triplets = loader.load_triplet_dataset(self.num_train, self.num_train + self.num_test, batch_size=1, shuffle=False)
        
        return train_triplets, test_triplets

    def train(self, path):
        # given:    self variables
        # return:   the trained model
        
        print('Loading data...')
        train_loader, test_loader = self.call_loader(path=path)
        self.model.train()
        
        training_profile = np.zeros((self.epochs, 6))

        print('Beginning to train...')
        for epoch in range(self.epochs):
            start_epoch = default_timer()   # it's helpful to time this

            train_loss = 0
            test_loss = 0
            num_train_examples = 0
            train_accuracy = 0

            self.model.train()
            for triplet in train_loader:
                anc = triplet[0].to(self.device)
                pos = triplet[1].to(self.device)
                neg = triplet[2].to(self.device)

                anc.x[:,1:] = torch.clamp(anc.x[:,1:], min=0, max=1)
                pos.x[:,1:] = torch.clamp(pos.x[:,1:], min=0, max=1)
                neg.x[:,1:] = torch.clamp(neg.x[:,1:], min=0, max=1)

                a_out = self.model(anc.x.float(), anc.edge_index.int(), edge_attr = anc.edge_attr)
                p_out = self.model(pos.x.float(), pos.edge_index.int(), edge_attr = pos.edge_attr)
                n_out = self.model(neg.x.float(), neg.edge_index.int(), edge_attr = neg.edge_attr)

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
                
                if torch.isnan(a_out[0]).item():
                    print(anc.y)


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            with torch.no_grad():
                test_accuracy = 0
                num_test_examples = 0

                self.model.eval()
                for triplet in test_loader:
                    anc = triplet[0].to(self.device)
                    pos = triplet[1].to(self.device)
                    neg = triplet[2].to(self.device)

                    anc.x[:,1:] = torch.clamp(anc.x[:,1:], min=0, max=1)
                    pos.x[:,1:] = torch.clamp(pos.x[:,1:], min=0, max=1)
                    neg.x[:,1:] = torch.clamp(neg.x[:,1:], min=0, max=1)

                    a_out = self.model(anc.x.float(), anc.edge_index.int(), edge_attr = anc.edge_attr)
                    p_out = self.model(pos.x.float(), pos.edge_index.int(), edge_attr = pos.edge_attr)
                    n_out = self.model(neg.x.float(), neg.edge_index.int(), edge_attr = neg.edge_attr)
                        
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
            training_profile[epoch] = (epoch, epoch_time, train_loss / num_train_examples, train_accuracy / num_train_examples, test_loss / num_test_examples, test_accuracy / num_test_examples)
            print(num_train_examples, num_test_examples)

            self.scheduler.step()
        print(num_train_examples, num_test_examples)
        print('Training complete...')

        if self.save_model:
            print('Saving model...')
            torch.save(self.model, f'models/{self.model_path}')

        if self.training_figure:
            print('Generating figure...')

            epochs = training_profile[:,0]
            train_loss = training_profile[:,2]
            train_accuracy = training_profile[:,3]
            test_loss = training_profile[:,4]
            test_accuracy = training_profile[:,5]

            fig, ax1 = plt.subplots()

            color = 'tab:red'
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Log Triplet Loss', color=color)
            ax1.plot(epochs, np.log(train_loss), color=color, label='training loss')
            ax1.plot(epochs, np.log(test_loss), '-.', color=color, label='testing loss')
            ax1.tick_params(axis='y', labelcolor=color)
            ax1.legend(loc='upper left')

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

            color = 'tab:blue'
            ax2.set_ylabel('Accuracy', color=color)  # we already handled the x-label with ax1
            ax2.plot(epochs, train_accuracy, color=color, label='training accuracy')
            ax2.plot(epochs, test_accuracy, '-.', color=color, label='testing accuracy')
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            ax2.legend(loc='lower left')
            # plt.figtext(0,0,self.configs)
            plt.savefig(f"training_plots/{self.model_path}.png")
            np.save(f"training_text/{self.model_path}.txt", training_profile)

    def compute_distance(self, anchor_features, comparison_features):
        # given:    two data objects; 1 to be queried against, 1 being compared to the query
        # return:   the distance between the two outputs of the model

        loss = torch.nn.MSELoss(reduction='sum')
        distance = loss(anchor_features, comparison_features)

        return distance.item()

def run_trainer(t, p):
    # use the configs to train a GCN, saving it in 'models/'
    euler_path = '/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/3dssg/'
    singularity_path = '/mnt/datasets/3dssg/'
    train_configs1 = {
        'learning_rate':        0.001,
        'epochs':               10,
        'batch_size':           1,
        'num_train':            1100,
        'num_test':             200,
        'scheduler_step':       2,
        'scheduler_gamma':      0.9,
        'triplet_loss_margin':  1.0,
        'triplet_loss_p':       2,
        'in_channels':          -1,
        'hidden_channels':      128,
        'num_layers':           5,
        'out_channels':         64, 
        'kernel_size':          1,
        'use_pretrained':       False,
        'data_source':          'ssg', # pipeline or ssg
        'save_model':           True,
        'training_figure':      True,
        'threshold':            1,
        'path':                 '../../../data/3dssg/'
    }
    train_configs2 = {
        'learning_rate':        0.001,
        'epochs':               10,
        'batch_size':           1,
        'num_train':            150,
        'num_test':             50,
        'scheduler_step':       1,
        'scheduler_gamma':      0.97,
        'triplet_loss_margin':  1.0,
        'triplet_loss_p':       2,
        'in_channels':          -1,
        'hidden_channels':      128,
        'num_layers':           5,
        'out_channels':         64, 
        'kernel_size':          1,
        'use_pretrained':       False,
        'data_source':          'pipeline', # pipeline or ssg
        'save_model':           True,
        'training_figure':      True,
        'threshold':            1.0,
        'path':                 '../../../data/hypersim_graphs/'
    }
    train_configs3 = {
        'learning_rate':        0.001,
        'epochs':               10,
        'batch_size':           1,
        'num_train':            200,
        'num_test':             50,
        'scheduler_step':       1,
        'scheduler_gamma':      0.97,
        'triplet_loss_margin':  1.0,
        'triplet_loss_p':       2,
        'in_channels':          -1,
        'hidden_channels':      128,
        'num_layers':           5,
        'out_channels':         64, 
        'kernel_size':          1,
        'use_pretrained':       False,
        'data_source':          'pipeline', # pipeline or ssg
        'save_model':           True,
        'training_figure':      True,
        'threshold':            t,
        'path':                 p,
    }
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'

    # pdb.set_trace()
    train_configs = train_configs3
    if p == '../../../data/hypersim_graphs/':
        model_name = f'GT_threshold:{train_configs["threshold"]}'
    elif p == '/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/hypersim_graphs_sem_resnet50/':
        model_name = f'ResNet50_threshold:{train_configs["threshold"]}'
    else:
        model_name = "example_model"

    trainer = LoadAndTrain(train_configs, device, model_name)
    trainer.train(train_configs['path'])

run_trainer(int(sys.argv[1]), sys.argv[2])