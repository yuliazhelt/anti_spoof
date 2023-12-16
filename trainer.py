import torch
from torch import nn
from tqdm import tqdm
import os
import wandb

from utils import eer_metric

@torch.no_grad()
def val_epoch(data_loader, model, criterion, device):
    running_loss = 0
    num_total = 0.0
    model.eval()

    labels = []
    model_scores = []
    
    
    for audio, label in tqdm(data_loader):
        audio = audio.to(device)
        label = label.view(-1).type(torch.int64).to(device)
  
        pred = model(audio, is_test=True)

        model_scores.append(pred)
        labels.append(label)

        loss = criterion(pred, label)
        
        running_loss += (loss.item() * audio.shape[0])
        num_total += audio.shape[0]
       
    running_loss /= num_total

    eer = eer_metric(torch.cat(model_scores)[:, 0], torch.cat(labels))
    
    return running_loss, eer


def train_epoch(data_loader, model, optimizer, criterion, device):
    running_loss = 0
    num_total = 0.0
    model.train()
    
    for audio, label in tqdm(data_loader):
        audio = audio.to(device)
        label = label.view(-1).type(torch.int64).to(device)
        
        pred = model(audio)
        
        loss = criterion(pred, label)
        running_loss += (loss.item() * audio.shape[0])
        
        wandb.log({"train_loss": loss.item()})

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
       
        num_total += audio.shape[0]

       
    running_loss /= num_total
    
    return running_loss


def train(
        model,
        train_loader,
        optimizer,
        criterion,
        dev_loader=None,
        eval_loader=None, 
        device='cpu',
        save_path='/saved',
        num_epochs=100
    ):
    for epoch in range(num_epochs):
        running_loss = train_epoch(train_loader, model, optimizer, criterion, device)

        print("train loss: ", running_loss)
        torch.save(
            {
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()
            }, 
            f"{save_path}/epoch={epoch}.pth"
        )
        dev_loss, dev_eer = val_epoch(dev_loader, model, criterion, device)
        eval_loss, eval_eer = val_epoch(eval_loader, model, criterion, device)

        print("dev", f"loss={dev_loss}", f"eer={dev_eer}")
        print("eval", f"loss={eval_loss}", f"eer={eval_eer}")
