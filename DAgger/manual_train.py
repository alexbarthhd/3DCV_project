from drive_dataset_2 import DriveDataset

from drive_model import DriveModel

from early_stopping import EarlyStopping

import json

import numpy as np

from pathlib import Path

import time

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

from torchvision import transforms

class ManualTrain:
    def __init__(self, width, height, channels, model_path = 'dagger.h5', data_path = 'data', validation_split = .2, batch_size = 32, num_workers = 2, epochs = 100):
        self.has_cuda = torch.cuda.is_available()

        self.model_path = Path(model_path)
        self.model = DriveModel(width, height, channels)
        if self.has_cuda:
            self.model = self.model.cuda()

        self.data_path = Path(data_path)

        self.validation_split = validation_split
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.epochs = epochs

    def train_model(self):
        '''
        Train a model for self.epochs and print validation loss after each epoch.
        '''
        train_loader, val_loader = self.prepare_train_and_val_loader()

        if self.has_cuda:
            self.model = self.model.cuda()

        optimizer = Adam(self.model.parameters())

        es = EarlyStopping()

        for epoch in range(self.epochs):
            self.model.train()
            start = time.process_time()
            for image, (steering, throttle) in train_loader:
                if self.has_cuda:
                    image, steering, throttle = image.cuda(), steering.cuda(), throttle.cuda()

                predicted_steering, predicted_throttle = self.model(image)

                loss = F.mse_loss(predicted_steering, steering) + F.mse_loss(predicted_throttle, throttle)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.save_model()

            with torch.no_grad():
                self.model.eval()
                val_loss = torch.tensor(0.0)
                for image, (steering, throttle) in val_loader:
                    if self.has_cuda:
                        image, steering, throttle = image.cuda(), steering.cuda(), throttle.cuda()

                    predicted_steering, predicted_throttle = self.model(image)
                    val_loss += F.mse_loss(predicted_steering, steering) + F.mse_loss(predicted_throttle, throttle)

                print(f'Epoch {epoch} in {time.process_time() - start:.0f}s with val loss: {val_loss.item():.5f}')
                if es.step(val_loss):
                    break

    def prepare_train_and_val_loader(self):
        '''
        Return a train and validation loaders for training.
        '''
        dataset = DriveDataset(
            self.data_path,
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        )
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(self.validation_split * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        train_loader = DataLoader(dataset, batch_size = self.batch_size, num_workers = self.num_workers, sampler=train_sampler)
        val_loader = DataLoader(dataset, batch_size = self.batch_size, num_workers = self.num_workers, sampler=val_sampler)
        return train_loader, val_loader

    def save_model(self):
        '''
        Save a self.model to a file.
        '''
        torch.save(self.model.state_dict(), str(self.model_path))

if __name__ == '__main__':
    mt = ManualTrain(288, 352, 3)
    mt.train_model()