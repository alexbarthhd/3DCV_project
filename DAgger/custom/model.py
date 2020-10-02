import torch.nn as nn
from pathlib import Path
import torch

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
        if inferrence:
            model.load_state_dict(torch.load(f'models/{model_name}.h5', map_location=torch.device('cpu')))
        else:
            model.load_state_dict(torch.load(f'models/{model_name}.h5'))
        return 0