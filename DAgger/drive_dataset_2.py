import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.records = []
        for tub in root_dir.glob('tub*'):
            self.records += list((root_dir/tub).iterdir())
        self.transform = transform
    
    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        img = Image.open(self.records[i])

        if self.transform:
            img = self.transform(img)

        parts = self.records[i].stem.split('-')
        if parts[-2] == '':
            steering = torch.tensor(-float(parts[-1])).unsqueeze(-1)
        else:
            steering = torch.tensor(float(parts[-1])).unsqueeze(-1)

        throttle = torch.tensor(18.5).unsqueeze(-1)
        return img, (steering, throttle)

