
import torch
import torch.nn as nn



#%%模型设定
class Convolution(nn.Module):
    def __init__(self):
        super(Convolution, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=9,
                kernel_size=3,
                stride=1
            ),

            nn.BatchNorm2d(9),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=1)


        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=9,
                out_channels=18,
                kernel_size=3,
                stride=1
            ),

            nn.BatchNorm2d(18),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)

        )

        self.fc1 = nn.Sequential(
            nn.Linear(1*188*9, 256),
            nn.ReLU()
        )
        self.fc2 = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU()
        )
        self.fc3 = nn.Linear(64, 2)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

model = Convolution()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
loss_f = nn.BCELoss()


