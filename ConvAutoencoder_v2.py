from myImports import *


class ConvAutoencoder_v2(nn.Module):
    def __init__(self):
        super(ConvAutoencoder_v2, self).__init__()
        self.encoder = nn.Sequential(  # in- (N,3,512,512)
            nn.Conv2d(in_channels=3,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1), nn.ReLU(True),
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=64,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=2,
                      padding=1), nn.ReLU(True),
            nn.Conv2d(in_channels=128,
                      out_channels=128,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=0), nn.ReLU(True), nn.MaxPool2d(2, stride=2),
            nn.Conv2d(in_channels=128,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=2,
                      padding=1), nn.ReLU(True),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1), nn.ReLU(True),
            nn.Conv2d(in_channels=256,
                      out_channels=256,
                      kernel_size=(3, 3),
                      stride=1,
                      padding=1), nn.ReLU(True), nn.MaxPool2d(2, stride=2))
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=256,
                               kernel_size=(3, 3),
                               stride=1,
                               padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=256,
                               out_channels=128,
                               kernel_size=(3, 3),
                               stride=2,
                               padding=0),
            nn.ConvTranspose2d(in_channels=128,
                               out_channels=64,
                               kernel_size=(3, 3),
                               stride=2,
                               padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=64,
                               out_channels=32,
                               kernel_size=(3, 3),
                               stride=2,
                               padding=1),
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=32,
                               kernel_size=(3, 3),
                               stride=2,
                               padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(in_channels=32,
                               out_channels=3,
                               kernel_size=(4, 4),
                               stride=2,
                               padding=2), nn.Tanh())

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x