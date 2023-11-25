import matplotlib.pyplot as plt
import numpy as np
import os
import itertools
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn import Module
import time
import torchvision.transforms as transforms
import torchvision

## define the general training steps
def train(model, device, dataloader, criterion, optimizer, epoch):
    
    print('\n')
    print(f'Training Epoch: {epoch}')
    
    model.train()
    running_loss = 0.0
    correct = 0.0
    for batch_idx, (data) in enumerate(dataloader, 1):
        # zero the parameter gradients
        optimizer.zero_grad()

        # send input and label data to the device
        inputs, labels = data[0].to(device), data[1].to(device)
        
        batch_size = inputs.size(0)
        
        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        #metrics and printouts
        running_loss += loss.item()
        pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(labels.view_as(pred)).sum().item()
        
        if batch_idx % 20 == 0:
            print(f'Epoch {epoch} - Train Loss: {running_loss/(batch_idx*batch_size) :.6f} - Train Top1 Accuracy: {100.*correct/(batch_idx*batch_size) :.6f}')
    
    print(f'Training Epoch: {epoch} Complete')
    print('\n')
    
    return running_loss/len(dataloader.dataset), 100.*correct/len(dataloader.dataset)

## define general valuation steps:
def evaluate(model, device, dataloader, criterion, epoch):
    
    print('\n')
    print(f'Eval Epoch: {epoch}')
    
    model.eval()
    running_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for batch_idx, (data) in enumerate(dataloader, 1):

            # send input and label data to the device
            inputs, labels = data[0].to(device), data[1].to(device)

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            #metrics and printouts
            running_loss += loss.item()
            pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability for top-1
            correct += pred.eq(labels.view_as(pred)).sum().item()

            
    print(f'Epoch {epoch} - Eval Loss: {running_loss/len(dataloader.dataset) :.6f} - Eval Top1 Accuracy: {100.*correct/len(dataloader.dataset) :.6f}')
    
    print(f'Eval Epoch: {epoch} Complete')
    print('\n')
            
    return running_loss/len(dataloader.dataset), 100.*correct/len(dataloader.dataset)

def get_cifar10_loaders(path, train_batch_size=512, test_batch_size=2048):
    
    train_ds = torchvision.datasets.CIFAR10(
            root=path,
            train=True,
            download=False,
            transform=transforms.Compose([
#                 transforms.Resize(224),
#                 transforms.RandomResizedCrop(224),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
                ]))
    test_ds = torchvision.datasets.CIFAR10(
            root=path,
            train=False,
            download=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465),
                                     (0.2023, 0.1994, 0.2010)),
                ]))
    
    
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=train_batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader