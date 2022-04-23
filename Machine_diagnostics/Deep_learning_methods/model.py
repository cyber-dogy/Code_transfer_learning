from torch import nn

class MyCNN(nn.Module):

    def __init__(self):
        super(MyCNN, self).__init__()
        # conv1: Conv2d -> BN -> ReLU -> MaxPool
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16,
                      kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # conv2: Conv2d -> BN -> ReLU -> MaxPool
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64,
                      kernel_size=3, stride=1,
                      padding=1),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 2, 0),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        # fully connected layer
        self.fc = nn.Linear(64*2*2, 256)
        self.fc = nn.Linear(256, 128)
    def forward(self, x):
        """
        input: N * 3 * image_size * image_size
        output: N * num_classes
        """
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # view(x.size(0), -1): change tensor size from (N ,H , W) to (N, H*W)
        x = x.view(x.size(0), -1)
        output = self.fc(x)
        return output