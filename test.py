import os 
import pdb
import json
import torch
from matplotlib import pyplot as plt
import numpy as np
from dataset import GeneratorDataset
import yaml

# generator_dataset = GeneratorDataset(root_dir="./data", image_size=(256,192))

# generator_dataset.__getitem__(0)

def test():
    config = {
    'device':'cuda',
    'learning_rate': 0.0002,
    'batch_size': 16,
    'num_epochs': 16,
    'num_workers':2,
    'image_size': [256,192],
    'training_steps': 1.5e3,
    'betas': [0.5,0.999],
    'layer_idx': [2,7,12,21,30],
    'lambda_i': [1./32.,1./16.,1./8.,1./4.,1.],
    'log_interval': 10,
    'checkpoint_dir': './checkpoints',
    'tensorboard_dir': './tensorboard',
    'data_root': './data',
    }
    with open("config.yaml", "w") as file:
        yaml.dump(config, file, default_flow_style=False)

if __name__ == "__main__":
    test()
        

 

    
