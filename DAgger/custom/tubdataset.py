import torch
from torch.utils.data import Dataset

from pathlib import Path
from PIL import Image
import json

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