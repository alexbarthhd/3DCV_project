import json
from pathlib import Path
from PIL import Image
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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
            model.load_state_dict(torch.load(f'models/{model_name}.h5', map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(f'models/{model_name}.h5'))
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

class CustomModelWrapper: # class that conforms to the donkeycar drive loop
    def __init__(self):
        self.model = CustomModel()
        self.model.eval()

    def load(self, model_name):
        load_model(model_name, self.model, True)

    def run(self, x):
        x = transforms.functional.to_tensor(x)
        x = transforms.functional.normalize(x, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)).unsqueeze(0)
        steering, throttle = self.model(x)
        steering_out, throttle_out = steering[0][0].detach().numpy(), throttle[0][0].detach().numpy()
        #print(steering_out, throttle_out)
        return steering_out, throttle_out

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
            nn.Linear(6656, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
        )

        self.steering = nn.Linear(50, 1)
        self.throttle = nn.Linear(50, 1)

    def forward(self, x):
        x  = self.model(x)
        return self.steering(x), self.throttle(x)

def train(model_name, dataloader, epochs = 200):
    f = CustomModel()
    if torch.cuda.is_available():
        f = f.cuda()

    lr_lambda = lambda epoch: 1 - 0.01*(epoch - 100) if epoch > 99 else 1
    optimizer = Adam(f.parameters())
    scheduler = LambdaLR(optimizer, lr_lambda = lr_lambda)

    start_epoch = load_model(model_name, f)
    for epoch in range(start_epoch, epochs):
        start = time.process_time()
        for im, (steering, throttle) in dataloader:
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
        # save after each epoch
        torch.save(f.state_dict(), f'models/{model_name}_{epoch}.h5')
        p = Path(f'models/{model_name}_{epoch - 1}.h5')
        if p.exists():
            p.unlink()
        print(f'ep{epoch} in {time.process_time() - start:.0f}s.')