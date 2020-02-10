import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from torchvision.models import resnet18, resnet50
from mnist import MydataSet
from mlp import MLP
from cnnNet import ResNet
from utils import Flatten

"""
这个文件主要采用 迁移学习的方法，使用resnet18网络，去掉最后一层，进行分类

resnet18，acc 为 0.9758, 不如自己写的 Resnet
resnet50，acc 为 0.9805, 不如自己写的 Resnet
"""


batchsz = 64
lr = 1e-3
epochs = 20

data = pd.read_csv('./data/train.csv')
test = pd.read_csv('./data/test.csv')

train, valid = train_test_split(data, stratify=data.label, test_size=0.2, random_state=211)

train_dataset = MydataSet(df=train, mode='train')
train_datasetload = DataLoader(train_dataset, batch_size=batchsz, shuffle=True)

valid_dataset = MydataSet(df=valid, mode='train')
valid_datasetload = DataLoader(valid_dataset, batch_size=batchsz)

device = torch.device('cuda:0')

train_model = resnet50(pretrained=True)
net = nn.Sequential(
    *list(train_model.children())[:-1],
    Flatten(),
    nn.Linear(2048, 10)
).to(device)


# net = MLP().to(device)
# net = ResNet().to(device)

optimizer = optim.Adam(net.parameters(), lr=lr)
criteon = nn.CrossEntropyLoss()

for epoch in range(epochs):

    for batch_id, (x, y) in enumerate(train_datasetload):

        x, y = x.to(device), y.to(device)

        # x = x.view(x.size(0), 28 * 28)

        logits = net(x)

        loss = criteon(logits, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_id % 100 == 0:
            print('epoch:[{}/{}], loss:{:.4f}'.format(epoch, epochs, loss))

net.eval()

with torch.no_grad():
    correct = 0
    total = 0

    for x, y in valid_datasetload:
        x = x.to(device)
        y = y.to(device)
        # x = x.view(x.size(0), 28 * 28)

        logits = net(x)

        pred = logits.argmax(dim=1)

        total += len(y)

        correct += pred.eq(y).sum().float().item()

print('acc:{:.4f}'.format(correct / total))

test_dataset = MydataSet(df=test, mode='test')
test_datasetload = DataLoader(test_dataset, batch_size=batchsz)

net.eval()
pred = []
for x in test_datasetload:
    x = x.to(device)
    # x = x.view(x.size(0), 28*28)

    logits = net(x)
    pre = logits.argmax(dim=1).cpu().numpy()
    pred += list(pre)

submit = pd.read_csv('./data/sample_submission.csv')
submit['Label'] = pred
submit.to_csv('submission_mlp.csv', index=False)
