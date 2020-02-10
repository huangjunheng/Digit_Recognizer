"""
https://blog.csdn.net/laplacebh/article/details/97648824
抄的这位老哥的网络
acc:0.9917 比我自己的好一点
"""
import torch
from torch import nn


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(  # (1, 28, 28)
            nn.Conv2d(
                in_channels=1,  # 输入通道数，若图片为RGB则为3通道
                out_channels=32,  # 输出通道数，即多少个卷积核一起卷积
                kernel_size=3,  # 卷积核大小
                stride=1,  # 卷积核移动步长
                padding=1,  # 边缘增加的像素，使得得到的图片长宽没有变化
            ),  # (32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),  # (32, 28, 28)
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),  # 池化 (32, 14, 14)
        )
        self.conv3 = nn.Sequential(  # (32, 14, 14)
            nn.Conv2d(32, 64, 3, 1, 1),  # (64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),  # (64, 14, 14)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # (64, 7, 7)
        )
        self.out = nn.Sequential(
            nn.Dropout(p=0.5),  # 抑制过拟合
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)  # (batch_size, 64*7*7)
        output = self.out(x)
        return output
