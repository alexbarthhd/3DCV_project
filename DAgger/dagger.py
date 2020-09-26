from drive_dataset import DriveDataset

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

class Dagger:
    def __init__(self, expert, model_path = 'models/dagger.h5', data_path = 'data', validation_split = .2, batch_size = 1, num_workers = 1, epochs = 10):
        '''
        Dagger class used to drive the car.
        '''
        self.expert = expert

        self.has_cuda = torch.cuda.is_available()

        self.model_path = Path(model_path)
        self.model = DriveModel(70,160)
        self.load_model()
        if self.has_cuda:
            self.model = self.model.cuda()

        self.data_path = Path(data_path)

        self.validation_split = validation_split
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.epochs = epochs

    def load_model(self):
        '''
        Load a self.model from file.
        '''
        if self.model_path.exists():
            self.model.load_state_dict(torch.load(str(self.model_path)))

    def run(self):
        for episode in range(10):
            if episode == 0:
                # expert drives for one or two laps
                pass
            else:
                # model drives until failure
                self.query_expert()
            self.train_model()

    def query_expert(self):
        '''
        Load each record and pass the image associated with the record to the expert.
        Save the record with the adjusted labels.
        '''
        dataset = DriveDataset(self.data_path)
        dataloader = DataLoader(dataset, batch_size = 1, num_workers = 1)
        for image, _, record_path, record in dataloader:
            record['steering'], record['throttle'] = self.expert(image)
            record_path.write_text(json.dumps(record))

    def train_model(self):
        '''
        Train a model for self.epochs and print validation loss after each epoch.
        '''
        train_loader, val_loader = self.prepare_train_and_val_loader()

        self.model = DriveModel(70,160)
        if self.has_cuda:
            self.model = self.model.cuda()
            
        optimizer = Adam(self.model.parameters())

        es = EarlyStopping()

        for epoch in range(self.epochs):
            self.model.train()
            start = time.process_time()
            for image, (steering, throttle), _, _ in train_loader:
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
                for image, (steering, throttle), _, _ in val_loader:
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