import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Subset
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


class CustomDataset(Dataset):

    def __init__(self, train_transform=None, test_transform=None):

        self.train_transform = train_transform if train_transform else self.default_transform()
        self.test_transform = test_transform if test_transform else self.default_transform()

        self.path_train = 'data/Training'
        self.path_test = 'data/Testing'

        self.data_train = datasets.ImageFolder(root=self.path_train, transform=self.train_transform)
        self.data_test = datasets.ImageFolder(root=self.path_test, transform=self.test_transform)

    def default_transform(self):
        return transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
    
    def get_train_loader(self, test_split=0.2, **loader_kwargs):
        
        loader_defaults = {'batch_size': 32, 'shuffle': True, 'num_workers': 4}
        loader_defaults.update(loader_kwargs)

        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_split, random_state=42)
        targets = np.array(self.data_train.targets)

        for train_index, val_index in sss.split(np.zeros(len(targets)), targets):
            train_subset = Subset(self.data_train, train_index)
            val_subset = Subset(self.data_train, val_index)

        train_loader = DataLoader(train_subset, **loader_defaults)
        val_loader   = DataLoader(val_subset, **loader_defaults)
        
        return train_loader, val_loader

    def get_test_loader(self, **loader_kwargs):
        loader_defaults = {'batch_size': 32, 'shuffle': False, 'num_workers': 4}
        loader_defaults.update(loader_kwargs)

        test_loader = DataLoader(self.data_test, **loader_defaults)
        return test_loader
