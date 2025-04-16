import os
import pickle
import torch
import math

import numpy as np
import pandas as pd
import anndata as ad

from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

class PeakDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(os.path.join(path, 'sc_mat.txt'), sep='\t', index_col=0)
        # self.data = self.data[self.data.sum(axis=1) != 0]  # Drop rows with all zeros
        self.sample_list = self.data.columns  # Convert to list for indexing
        self.len = len(self.sample_list)
        self.shape = self._calculate_shape(self.data.shape[0])
        self.label = self._load_labels(os.path.join(path, 'labels.txt'))
        self.label_onehot = self._one_hot_encode(self.label)
        self.drop_prob = self._calculate_drop_prob()

    def _calculate_shape(self, n):
        return n if n % 80 == 0 else n + 80 - n % 80

    def _load_labels(self, label_path):
        label_df = pd.read_csv(label_path, sep='\t', header=None)
        label_df.index = [f'cell_{i}' for i in range(len(label_df))]
        return label_df[0]

    def _one_hot_encode(self, labels):
        label_dict = {label: idx for idx, label in enumerate(labels.unique())}
        return labels.map(label_dict)

    def _calculate_drop_prob(self):
        zero_ratio = (self.data == 0).sum(axis=1) / self.data.shape[1]  # Use shape[1] for sum of columns
        positive_mean = self.data[self.data > 0].mean(axis=1).fillna(0)
        drop_prob = zero_ratio / (positive_mean + 1)
        return drop_prob / drop_prob.max()

    def normalize(self, x):
        x = np.clip(x, 0, 100)  # Clip values to max 100
        return np.log2(x + 1) / np.log2(100)  # Normalize to [0, 1]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        # try:
        #     sample = self.sample_list[index]
        #     data = self.data[sample]
        #     label = self.label.loc[sample].item()
        #     label_onehot = self.label_onehot[index]
        # except:
        #     print(sample)
        sample = self.sample_list[index]
        data = self.data[sample]
        label = self.label.loc[sample]
        label_onehot = self.label_onehot[index]
        data = torch.tensor(data.values, dtype=torch.float32)
        # padding 
        data = F.pad(data, (0, self.shape - data.shape[0]), 'constant', 0)
        # normalize
        data = self.normalize(data)
        return data, label, sample, label_onehot

# TODO: anndata dataset

class AnnDataset(Dataset):
    pass