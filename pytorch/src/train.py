from __future__ import print_function, division
import os
import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
import logging

from random import shuffle
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from model import Network
from dataloader import KeenDataloader
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

config = {
    'train_path' : "C:\\Users\\Karthik\\Documents\\KEEN_DATA\\Training",
    'val_path' : "C:\\Users\\Karthik\\Documents\\KEEN_DATA\\Validation",
    'epochs' : 50,
    'lr' : 0.001,
    'wd' : 0.0001,
    'batch_size' : 16,
    'val_batch_size' : 8,
    'num_workers' : 4,
    'print_freq' : 200,
    'save_root' : "C:\\Users\\Karthik\\Desktop\\checkpoints",
    'checkpoint' : "C:\\Users\\Karthik\\Desktop\\checkpoints\\checkpoint",
    'logs_root' : "C:\\Users\\Karthik\\Desktop\\checkpoints\\logs",
    'resume' : "C:\\Users\\Karthik\\Desktop\\checkpoints\\ckpt_epoch_049_.pth",
    'save_freq' : 1000,
    'val_freq' : 5,
    'initial_eval' : False,
    'is_training' : False
}

def load_model(model, optimizer, config):
    if config["resume"] is not None : 
        checkpoint = torch.load(config['resume'], map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=False)
        #if 'optimizer' in checkpoint:
        #    optimizer.load_state_dict(checkpoint['optimizer'])

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def create_dataloader(config):
    train_set = KeenDataloader(config['train_path'], is_training=True)
    val_set = KeenDataloader(config['val_path'], is_training=False)
    tkwargs = {'batch_size': config['batch_size'],
               'num_workers': config['num_workers'],
               'pin_memory': True, 'drop_last': True}
    trainloader = DataLoader(train_set, **tkwargs)
    tkwargs = {'batch_size': config['val_batch_size'],
               'num_workers': config['num_workers'],
               'pin_memory': True, 'drop_last': True}
    testloader = DataLoader(val_set, **tkwargs)
    return trainloader, testloader

def create_model_and_optimizer():
    model = Network(2).to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['wd'])

    return model, criterion, optimizer

def train_step(model, criterion, optimizer, trainloader, epoch):
    model.train()
    running_loss = 0.0
    iteration = 0
    for data in tqdm(trainloader):
        iteration+=1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['image'].to(device), data['label'].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if iteration % config['print_freq'] == 0:
            logging.info('[%d, %5d] loss: %.3f' %
                (epoch + 1, iteration + 1, running_loss / iteration))
    logging.info('[%d, %5d] Epoch loss: %.3f' %
                    (epoch + 1, iteration + 1, running_loss/iteration))
    logging.info(f'Epoch {epoch} completed')
    return running_loss/iteration

def val_step(model, criterion, optimizer, valloader):
    model.eval()
    val_loss = 0.0
    iteration = 0
    for data in tqdm(valloader):
        iteration+=1
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data['image'].to(device), data['label'].to(device)

        # forward + backward + optimize
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # print statistics
        val_loss += loss.item()
        if iteration % config['print_freq'] == 0:
            logging.info('[%5d] loss: %.3f' %
                (iteration + 1, val_loss / iteration))
    logging.info('[%5d] Validation loss: %.3f' %
                    (iteration + 1, val_loss/iteration))
    logging.info(f'Validation completed')
    return val_loss/iteration

def train(epochs, model, criterion, optimizer, trainloader, valloader):
    load_model(model, optimizer, config)
    for epoch in range(epochs):  # loop over the dataset multiple times
        epoch_loss = train_step(model, criterion, optimizer, trainloader, epoch)
        if epoch % config['val_freq'] == 0:
            val_loss = val_step(model, criterion, optimizer, valloader)
        save_model(model, optimizer, epoch, config)
        logging.info(f"Model saved under : {os.path.join(config['save_root'], f'ckpt_epoch_{epoch}.pth')}")

def save_model(model, optimizer, epoch, config):
    # save checkpoint
    model_optim_state = {'epoch': epoch,
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict(),
                         }
    model_name = os.path.join(
        config['save_root'], 'ckpt_epoch_%03d_.pth' % (
            epoch))
    torch.save(model_optim_state, model_name)
    logging.info('saved model {}'.format(model_name))

def predict(model, optimizer, sample, config):
    # predict a single sample
    load_model(model, optimizer, config)
    model.eval()
    inputs, labels = sample['image'].to(device), sample['label']
    prediction = model(inputs)
    print(torch.nn.functional.softmax(prediction), labels)

if __name__ == '__main__':
    if config['is_training']:
        os.makedirs(config['save_root'], exist_ok=True)
        os.makedirs(config['logs_root'], exist_ok=True)
        os.makedirs(config['checkpoint'], exist_ok=True)

        logging.basicConfig(level=logging.DEBUG,
                        filename=os.path.join(config['logs_root'], 'stdout.log'),
                        format='%(asctime)s %(message)s')
        logging.info('Run configurations:')
        for item in config:
            logging.info(f"{item} : {config[item]}")
        logging.info("Creating dataloaders")
        trainloader, valloader = create_dataloader(config)
        logging.info("Creating model, optimizer and criterion functions")
        model, criterion, optimizer = create_model_and_optimizer()
    
        if config["initial_eval"]:
            val_loss = val_step(model, criterion, optimizer, valloader)
        logging.info(f"Training model on {device}:")
        train(config['epochs'], model, criterion, optimizer, trainloader, valloader)
        logging.info('Finished Training')
    else:
        sample_set = KeenDataloader(config['val_path'], is_training=False)
        tkwargs = {'batch_size': 1,
                'num_workers': 1,
                'pin_memory': True, 'drop_last': True}
        sampleloader = DataLoader(sample_set, **tkwargs)
        model, criterion, optimizer = create_model_and_optimizer()
        for i, sample in enumerate(sampleloader):
            predict(model, optimizer, sample, config)
            if i == 5:
                break

