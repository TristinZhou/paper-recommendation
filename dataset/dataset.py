import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from dataset import padding


class TripletDataset(Dataset):
    def __init__(self, file_path='../data/train.csv',
                 x1_column='description_text', x2_column='title_abstract',
                 transform=transforms.Compose([
                     padding,
                 ])):
        self.df = pd.read_csv(file_path)
        self.x1_column = x1_column
        self.x2_column = x2_column
        self.transform = transform

    def __getitem__(self, index):
        anchor = self.df[self.x1_column].iloc[index]
        positive = self.df[self.x2_column].iloc[index]
        negative = positive
        while positive == negative:
            negative = self.df.sample(1)[self.x2_column].values[0]
        triplet = anchor, positive, negative
        if self.transform:
            triplet = list(map(self.transform, triplet))
        return triplet

    def __len__(self):
        return len(self.df)

    def dataframe(self):
        return self.df


class TestDataset(Dataset):
    def __init__(self, file_path='../data/test.csv',
                 transform=transforms.Compose([
                     padding,
                 ])):
        self.df = pd.read_csv(file_path)
        self.transform = transform

    def __getitem__(self, index):
        items = self.df.iloc[index].values
        if self.transform:
            items = list(map(self.transform, items))
        return items

    def __len__(self):
        return len(self.df)

    def dataframe(self):
        return self.df


class EvalDataset(Dataset):
    def __init__(self, df,
                 id,
                 column,
                 transform=transforms.Compose([
                     padding])):
        self.df = df
        self.column = column
        self.id = id
        self.transform = transform

    def __getitem__(self, index):
        text = self.df.loc[index, self.column]
        id = self.df.loc[index, self.id]
        if self.transform:
            text = self.transform(text)
        return id, text

    def __len__(self):
        return len(self.df)

    def dataframe(self):
        return self.df
