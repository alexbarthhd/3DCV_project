import json
from PIL import Image
import torch
from torch.utils.data import Dataset

class DriveDataset(Dataset):
    '''
    TODO: specify record format.
    Currently 'record_<number>.json' should contain
    {
        "image": "image_<number>.ext",
        "steering": number in [-1.0, 1.0],
        "throttle": number in [-1.0, 1.0]
    }
    and with their corresponding images be located in folders named tub<some id>
    which themselves are in the root_dir.
    '''
    def __init__(self, root_dir, transform=None):
        self.records = []
        for tub in root_dir.glob('tub*'):
            self.records += list(tub.glob('*record*[0-9]*'))
        self.transform = transform
    
    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        record_path = self.records[i]
        record = json.loads(record_path.read_text())
        img = Image.open(record_path.parent/record['image'])

        if self.transform:
            img = self.transform(img)

        steering = torch.tensor(record['steering']).unsqueeze(-1)
        throttle = torch.tensor(record['throttle']).unsqueeze(-1)
        return img, (steering, throttle), record_path, record