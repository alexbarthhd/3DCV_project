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
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from custom.model import CustomModel, load_model
from custom.tubdataset import TubDataset
from custom.train import train

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
        load_model('track_199', self.expert, True)

        self.first_run = True
        self.no_progress_counter = 100


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
            self.first_run = False
        elif self.first_run:
            steering, throttle = 0.0, 0.25
        else:
            x = transforms.functional.to_tensor(self.frame)
            x = transforms.functional.normalize(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).unsqueeze(0)
            steering_tensor, throttle_tensor = self.model(x)
            steering, throttle = steering_tensor[0][0].detach().numpy(), throttle_tensor[0][0].detach().numpy()

        if throttle < 0.01 and not self.train: # minimum throttle to prevent stalling
            self.no_progress_counter -= 1

        if self.no_progress_counter == 0:
            print('NO PROGRESS MADE')
            steering, throttle = 0.0, 0.0
            self.train = True
            self.no_progress_counter = 100
        self.action = [steering, throttle]
        return self.frame

    def shutdown(self):
        self.running = False
        time.sleep(0.2)
        self.env.close()