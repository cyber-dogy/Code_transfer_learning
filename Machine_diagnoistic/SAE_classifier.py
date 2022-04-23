import torch
from torch import nn


#%%模型设定
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.en_cov = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.Tanh(),
            nn.Conv2d(32, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.Tanh()
        )

        self.en_fc = nn.Linear(400, 64)
        self.de_fc = nn.Linear(64, 400)
        self.de_conv = nn.Sequential(
            nn.ConvTranspose2d(16, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.Tanh(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        en = self.en_cov(x)
        code = self.en_fc(en.view(en.size(0), -1))
        de = self.de_fc(code)
        decoded = self.de_conv(de.view(de.size(0), 16, 5, 5))
        return code, decoded

model = AutoEncoder()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_f = nn.MSELoss()

