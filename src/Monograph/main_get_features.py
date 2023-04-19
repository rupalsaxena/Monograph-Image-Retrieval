import torch
from configs import train as train_config
from GNN.pipeline_features import pipeline_features

def run_training(graph):
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'
    
    train_obj = pipeline_features(train_config.config["model_name"], device)
    train_obj.get_features(graph)

if __name__ == '__main__':
    run_training(graph)