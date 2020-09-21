import torch.nn as nn

class DriveModel(nn.Module):
    '''
    TODO: 
    '''
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