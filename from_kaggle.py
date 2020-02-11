import torch
from torch import nn


class CNN_Kaggle(nn.Module):

    def __init__(self):
        super(CNN_Kaggle, self).__init__()

        self.models_1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 5, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.Dropout(0.4),

            nn.Conv2d(32, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 5, 2, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.Dropout(0.4)
        )

        self.models_2 = nn.Sequential(
            nn.Linear(64*6*6, 128),
            nn.ReLU(True),
            nn.BatchNorm1d(128),  # 切记这里是 BatchNorm1d 不是 BatchNorm2d
            nn.Dropout(0.4),
            nn.Linear(128, 10)
        )

    def forward(self, x):

        batchsz = x.size(0)
        x = self.models_1(x)
        x = x.view(batchsz, 64*6*6)
        logits = self.models_2(x)

        return logits


def main():
    data = torch.randn(2, 1, 28, 28)
    net = CNN_Kaggle()

    out = net(data)
    print(out.shape)


if __name__ == '__main__':
    main()