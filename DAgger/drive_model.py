import torch.nn as nn

class DriveModel(nn.Module):
    '''
    TODO: 
    '''
    def __init__(self, width, height, channels=3):
        super().__init__()
        self.width = width
        self.height = height

        ks = [(5,2), (5,2), (5,2), (3,1), (3,1)]

        cwidth = width
        cheight = height

        for k, s in ks:
            cwidth = int((cwidth - k)/s+1)
            cheight = int((cheight - k)/s+1)

        self.model = nn.Sequential( # similar to donkeycars KerasLinear
            nn.Conv2d(in_channels=channels, out_channels=24, kernel_size=ks[0][0], stride=ks[0][1]),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=ks[1][0], stride=ks[1][1]),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=ks[2][0], stride=ks[2][1]),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=ks[3][0], stride=ks[3][1]),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=ks[4][0], stride=ks[4][1]),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(cwidth * cheight * 64, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 50),
            nn.ReLU(inplace=True),
        )

        self.steering = nn.Linear(50, 1)
        self.throttle = nn.Linear(50, 1)

    def forward(self, x):
        x  = self.model(x)
        return self.steering(x), self.throttle(x)