import yaml
import torch
from torch import nn
from torch.utils.data import DataLoader
from datetime import datetime
from tqdm import tqdm
import os
import wandb
from copy import copy

from dataset import CustomDataset
from dataset import collate_fn
from model import RawNet2

from trainer import train


if __name__ == "__main__":
    SEED = 123
    torch.manual_seed(SEED)

    config_path = '/home/ubuntu/as_project/configs/config.yaml'
    with open(config_path) as f:
        config = yaml.safe_load(f)

    wandb.init(
        project="as_project",
        config=config
    )

    # data
    train_set = CustomDataset(**config['dataset']['train'])
    dev_set = CustomDataset(**config['dataset']['dev'])
    eval_set = CustomDataset(**config['dataset']['eval'])
    train_loader = DataLoader(train_set, collate_fn=collate_fn, batch_size=config['train']['batch_size'], shuffle=True, num_workers=4)
    dev_loader = DataLoader(dev_set, collate_fn=collate_fn, batch_size=config['dev']['batch_size'], shuffle=False, num_workers=4)
    eval_loader = DataLoader(eval_set, collate_fn=collate_fn, batch_size=config['eval']['batch_size'], shuffle=False, num_workers=4)
    
    print("len train set", len(train_set))
    print("len dev set", len(dev_set))
    print("len eval set", len(eval_set))

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = RawNet2(args=config['model']).to(device)
    print(device)
    print(model)

    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'], weight_decay=config['train']['weight_decay'])
    class_weights = torch.FloatTensor(config['train']['ce_weights']).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    load_path = config['train']['load_path']
    if load_path != 'none':
        model_dict = torch.load(load_path)
        model.load_state_dict(model_dict['model'])
        optimizer.load_state_dict(model_dict['optimizer'])
        print("model loaded")

    current_date = datetime.now()
    current_date_str = current_date.strftime("%d.%m-%H:%M:%S")
    print(current_date_str)

    os.mkdir(f"{config['train']['save_path']}/run_{current_date_str}")

    train(
        model=model,
        train_loader=train_loader, 
        optimizer=optimizer,
        criterion=criterion,
        eval_loader=eval_loader,
        dev_loader=dev_loader,
        num_epochs=config['train']['num_epochs'],
        device=device,
        save_path=f"{config['train']['save_path']}/run_{current_date_str}",
    )
    wandb.finish()