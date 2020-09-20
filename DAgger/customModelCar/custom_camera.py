#from donkeycar.parts.dgym import DonkeyGymEnv
import os
import time
import numpy as np
import cv2
import gym
import gym_donkeycar

import json
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

import sys

def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

class CustomDonkeyGymEnv(object):
    def __init__(self, sim_path, host="127.0.0.1", port=9091, headless=0, env_name="donkey-generated-track-v0", sync="asynchronous", conf={}, delay=0):
        #super().__init__(sim_path, host, port, headless, env_name, sync, conf, delay)
        if sim_path != "remote":
            if not os.path.exists(sim_path):
                raise Exception("The path you provided for the sim does not exist.") 

            if not is_exe(sim_path):
                raise Exception("The path you provided is not an executable.") 

        conf["exe_path"] = sim_path
        conf["host"] = host
        conf["port"] = port
        self.env = gym.make(env_name, conf=conf)
        self.frame = self.env.reset()
        self.action = [0.0, 0.0]
        self.running = True
        self.info = { 'pos' : (0., 0., 0.)}
        self.delay = float(delay)

        self.done = False
        self.train = False

        self.model = CustomModel()
        self.model.eval()
        load_model('dagger', self.model, True)

        self.expert = CustomModel()
        self.expert.eval()
        load_model('all', self.expert, True)

    def transform_frame(self, frame, fs):
        for f in fs:
            frame = f(frame)
        return frame

    def update(self):        
        while self.running:
            self.frame, _, self.done, self.info = self.env.step(self.action)

    def ask_expert(self, record_path):
        record = json.loads(record_path.read_text())
        x = Image.open(record_path.parent/record['cam/image_array'])
        x = transforms.functional.to_tensor(x)
        x = transforms.functional.normalize(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).unsqueeze(0)
        steering_tensor, throttle_tensor = self.expert(x)
        steering, throttle = steering_tensor[0][0].item(), throttle_tensor[0][0].item()
        record['user/angle'] = steering
        record['user/throttle'] = throttle
        record_path.write_text(json.dumps(record))

    def run_threaded(self, steering, throttle): # ignore steering and throttle
        if self.delay > 0.0:
            time.sleep(self.delay / 1000.0)
        if self.train:
            print('TRAIN CALLED')
            for _ in range(3): # reset multiple times because once isn't reliable
                time.sleep(2)
                self.env.reset()
            print('resets done')

            # relabel the dataset by calling expert
            dataset_path = Path('data')
            records = []
            for tub in dataset_path.glob('tub*'):
                records += list(tub.glob('*record*[0-9]*'))
            for record_path in records:
                self.ask_expert(record_path)
            # train dagger on the relabeled dataset
            dataset = TubDataset(
                'data',
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                ])
            )
            dataset_size = len(dataset)
            indices = list(range(dataset_size))
            validation_split = .2
            split = int(np.floor(validation_split * dataset_size))
            np.random.shuffle(indices)
            train_indices, val_indices = indices[split:], indices[:split]

            train_sampler = SubsetRandomSampler(train_indices)
            val_sampler = SubsetRandomSampler(val_indices)

            train_loader = DataLoader(dataset, batch_size = 4096, num_workers = 3, sampler=train_sampler)
            val_loader = DataLoader(dataset, batch_size = 4096, num_workers = 3, sampler=val_sampler)

            train('dagger', train_loader, val_loader)
            load_model('dagger', self.model, True)

            self.frame = self.env.reset()
            self.train = False
        if self.done:
            print('DONE CALLED')
            steering, throttle = 0.0, 0.0
            self.train = True
        else:
            x = transforms.functional.to_tensor(self.frame[50:,:,:])
            x = transforms.functional.normalize(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).unsqueeze(0)
            steering_tensor, throttle_tensor = self.model(x)
            steering, throttle = steering_tensor[0][0].detach().numpy(), throttle_tensor[0][0].detach().numpy()

        self.action = [steering, throttle]
        return self.frame[50:,:,:]

    def shutdown(self):
        self.running = False
        time.sleep(0.2)
        self.env.close()




class TubDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        p = Path(root_dir)
        self.records = []
        for tub in p.glob('tub*'):
            self.records += list(tub.glob('*record*[0-9]*'))
        self.transform = transform
    
    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record_path = self.records[i]
        record = json.loads(record_path.read_text())
        img = Image.open(record_path.parent/record['cam/image_array'])

        if self.transform:
            img = self.transform(img)

        steering = torch.tensor(record['user/angle']).unsqueeze(-1)
        throttle = torch.tensor(record['user/throttle']).unsqueeze(-1)
        return img, (steering, throttle)

def load_model(model_name, model, inferrence = False):
    model_path = Path('models')
    all_models = list(model_path.iterdir())
    saved_model = [m for m in all_models if model_name in str(m)]
    if len(saved_model) == 0: # no saved models
        return 0
    elif model_name == 'dagger':
        if inferrence:
            model.load_state_dict(torch.load(f'models/dagger.h5', map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(f'models/dagger.h5'))
        return 0
    else:
        under_split = str(saved_model[0].stem).split('_')
        dot_split = under_split[-1].split('.')
        epoch = int(dot_split[0])
        if inferrence:
            model.load_state_dict(torch.load(f'models/{model_name}_{epoch}.h5', map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(f'models/{model_name}_{epoch}.h5'))
        return epoch + 1
    
class CustomModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential( # similar to donkeycars KerasLinear
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(1664, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
        )

        self.steering = nn.Linear(50, 1)
        self.throttle = nn.Linear(50, 1)

    def forward(self, x):
        x  = self.model(x)
        return self.steering(x), self.throttle(x)

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

def train(model_name, train_loader, val_loader, epochs = 100):
    f = CustomModel()
    if torch.cuda.is_available():
        f = f.cuda()

    lr_lambda = lambda epoch: 1 - 0.02*(epoch - 50) if epoch > 49 else 1
    optimizer = Adam(f.parameters())
    scheduler = LambdaLR(optimizer, lr_lambda = lr_lambda)

    start_epoch = load_model(model_name, f)
    es = EarlyStopping(patience=5)
    for epoch in range(start_epoch, epochs):
        f.train()
        start = time.process_time()
        for im, (steering, throttle) in train_loader:
            if torch.cuda.is_available():
                im = im.cuda()
                steering = steering.cuda()
                throttle = throttle.cuda()
            
            optimizer.zero_grad()
            psteering, pthrottle = f(im)

            loss = nn.functional.mse_loss(psteering, steering) + nn.functional.mse_loss(pthrottle, throttle)
            loss.backward()

            optimizer.step()

        scheduler.step()
        # val
        with torch.no_grad():
            f.eval()
            val_loss = torch.tensor(0.0)
            for im, (steering, throttle) in train_loader:
                if torch.cuda.is_available():
                    im = im.cuda()
                    steering = steering.cuda()
                    throttle = throttle.cuda()

                psteering, pthrottle = f(im)
                val_loss += nn.functional.mse_loss(psteering, steering) + nn.functional.mse_loss(pthrottle, throttle)
            
            print(f'val_loss: {val_loss.item()}')
            if es.step(val_loss):
                break

        # save after each epoch
        torch.save(f.state_dict(), f'models/{model_name}.h5')
        p = Path(f'models/{model_name}.h5')
        print(f'ep{epoch} in {time.process_time() - start:.0f}s.')

        # early stopping