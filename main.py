import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import List, Optional, Any
from tqdm import tqdm
import os
import wandb
from copy import copy

from dataset import CustomDataset
from dataset import dataset as collate_fn
from model import RawNet2
from utils import eer_metric

from trainer import train


if __name__ == "__main__":
    SEED = 123
    torch.manual_seed(SEED)

    config_path = '/home/ubuntu/as_project/configs/config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    wandb.init(
        project="anti_spoof",
        config=config
    )

    # data
    train_set = CustomDataset(**config['dataset']['train'])
    eval_set = CustomDataset(**config['dataset']['val'])
    train_loader = DataLoader(train_set, collate_fn=collate_fn, batch_size=config['training']['batch_size'], shuffle=True, num_workers=4)
    eval_loader = DataLoader(eval_set, collate_fn=collate_fn, batch_size=config['training']['batch_size'], shuffle=False, num_workers=4)
    
    print("len val set", len(eval_set))
    print("len train set", len(train_set))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = RawNet2(**config['model']).to(device)
    print(model)

    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
    class_weights = torch.FloatTensor(eval(config['train']['ce_weights'])).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    train(model, optimizer, criterion, train_loader, eval_loader=eval_loader, save_path=config['train']['save_path'], num_epochs=config['train']['num_epochs'])
    wandb.finish()