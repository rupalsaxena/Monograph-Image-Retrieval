import torch
from configs import train as train_config
from GNN.load_n_train import load_n_train

def run_training():
    if torch.cuda.is_available():
        device='cuda:0'
    else:
        device='cpu'
    
    train_obj = load_n_train(train_config.config, device, train_config.config["model_name"])
    train_obj.train(path=train_config.config["input_path"])

if __name__ == '__main__':
    run_training()