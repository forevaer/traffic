import pandas as pd
from utils import image
from torch.utils.data import Dataset


class Item(dict):

    def __init__(self, path, classify):
        super().__init__()
        self.path = path
        self.image = None
        self.classify = classify
        self.parse()
        self.save()

    def parse(self):
        if (self.path is not None) and isinstance(self.path, str):
            self.image = image.load_image(self.path)
        self.classify = int(self.classify)

    def save(self):
        super().__setitem__('path', self.path)
        super().__setitem__('image', self.image)
        super().__setitem__('classify', self.classify)

    def trans(self, transform):
        self.image = transform(self.image)
        self.save()
        return self


class TrafficCSVLoader(object):

    def __init__(self, path, single_image=False):
        self.sing_image = single_image
        self.path = path
        self.data = None
        self.load_csv()

    def load_csv(self):
        if self.sing_image:
            self.data = pd.DataFrame({
                'path': self.path,
                'classify': [-1 for _ in range(len(self.path))]
            })
        else:
            self.data = pd.read_csv(self.path)


class TrafficDataSet(Dataset):

    def __init__(self, loader: TrafficCSVLoader, transform=None):
        self.transform = transform
        self.data = loader.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item_data = self.data.iloc[idx]
        return Item(item_data['path'], item_data['classify']).trans(self.transform)
