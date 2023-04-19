import torch
from configs import get_features as config
from GNN.pipeline_features import pipeline_features

def get_features(graph):
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'
    
    train_obj = pipeline_features(config.MODEL_NAME, device)
    train_obj.get_features(graph)
