import torch
from torch import nn


class ResNet(nn.Module):

    def __init__(self):
        super(ResNet, self).__init__()

        self.func = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2),
        )

        self.linear = nn.Sequential(
            nn.Linear(128 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(True),
            nn.Dropout(p=0.5),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),

            nn.Linear(128, 10)
        )

    def forward(self, x):
        batchsz = x.size(0)
        x = self.func(x)
        x = x.view(batchsz, 128 * 7 * 7)
        logits = self.linear(x)

        return logits


def main():
    data = torch.randn(2, 1, 28, 28)
    net = ResNet()

    out = net(data)
    print(out.shape)


if __name__ == '__main__':
    main()
