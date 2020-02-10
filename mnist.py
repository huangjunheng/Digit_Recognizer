import torch
from torch.utils.data import Dataset, DataLoader  # Dataset 为大写
from torchvision import transforms
import pandas as pd
from sklearn.model_selection import train_test_split


class MydataSet(Dataset):

    def __init__(self, df, mode):
        super(MydataSet, self).__init__()

        self.df = df.values
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
        ])

        if self.mode == 'train':
            label = self.df[item, 0]
            img = torch.FloatTensor(self.df[item, 1:]).view(28, 28).unsqueeze(0)
            img = tf(img)
            # torch.Size([1, 28, 28])
            img = torch.stack([img, img, img], dim=0).squeeze(1)  # 实验迁移学习 跑MLP和自己写的cnnNet时，这行去掉

            return img, label
        else:
            img = torch.FloatTensor(self.df[item, :]).view(28, 28).unsqueeze(0)
            img = tf(img)
            img = torch.stack([img, img, img], dim=0).squeeze(1)  # 实验迁移学习 为了把单通道变为3通道
            return img


def main():
    data = pd.read_csv('./data/train.csv')
    test = pd.read_csv('./data/test.csv')

    train, valid = train_test_split(data, stratify=data.label, test_size=0.2, random_state=211)

    train_dataset = MydataSet(df=train, mode='train')
    img, label = train_dataset.__getitem__(1)
    print(img.shape, label)
    train_datasetload = DataLoader(train_dataset, batch_size=32, shuffle=True)


if __name__ == '__main__':
    main()
