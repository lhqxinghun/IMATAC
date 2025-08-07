import os
import pickle
import torch
import math

import numpy as np
import pandas as pd
import anndata as ad

from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

import time


class PeakDataset(Dataset):
    def __init__(self, path):
        self.path = path
        self.data = pd.read_csv(os.path.join(path, 'sc_mat.txt'), sep='\t', index_col=0)
        # self.data = self.data[self.data.sum(axis=1) != 0]  # Drop rows with all zeros
        self.sample_list = self.data.columns  # Convert to list for indexing
        self.len = len(self.sample_list)
        self.raw_shape = self.data.shape[0]  
        self.shape = self._calculate_shape(self.data.shape[0])
        self.label = self._load_labels(os.path.join(path, 'labels.txt'))
        self.label_onehot = self._one_hot_encode(self.label)
        self.num_classes = len(self.label.unique())
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
        print("Calculating drop probabilities...")
        tic = time.time()
        zero_ratio = np.mean(self.data == 0, axis=0)
        positive_mean = np.where(self.data > 0, self.data, np.nan)
        positive_mean = np.nanmean(positive_mean, axis=0)
        positive_mean = np.nan_to_num(positive_mean)  
        with np.errstate(divide='ignore', invalid='ignore'):
            drop_prob = zero_ratio / (positive_mean + 1)
        max_prob = np.nanmax(drop_prob)
        if max_prob > 0:
            drop_prob /= max_prob
        else:
            drop_prob = np.zeros_like(drop_prob)
        toc = time.time()
        print(f"Drop probability calculation took {toc - tic:.2f} seconds.")
        return drop_prob
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
        label_onehot = self.label_onehot.iloc[index]
        data = torch.tensor(data.values, dtype=torch.float32)
        # padding 
        data = F.pad(data, (0, self.shape - data.shape[0]), 'constant', 0)
        # normalize
        data = self.normalize(data)
        return data, label, sample, label_onehot

class H5adPeakDataset(Dataset):
    def __init__(self, h5ad_path):
        self.h5ad_path = h5ad_path
        self.adata = ad.read_h5ad(h5ad_path)
        self.data = pd.DataFrame(self.adata.X.toarray() if hasattr(self.adata.X, 'toarray') else self.adata.X,
                                 index=self.adata.obs_names, columns=self.adata.var_names)
        self.sample_list = self.data.index.tolist()
        self.len = len(self.sample_list)
        self.raw_shape = self.data.shape[1]  
        self.shape = self._calculate_shape(self.data.shape[1])
        if 'label' in self.adata.obs:
            self.label = self.adata.obs['label']
        else:
            self.label = pd.Series([0]*self.len, index=self.sample_list)
        self.label_onehot = self._one_hot_encode(self.label)
        self.num_classes = len(pd.Series(self.label).unique())
        self.drop_prob = self._calculate_drop_prob()

    def _calculate_shape(self, n):
        return n if n % 80 == 0 else n + 80 - n % 80

    def _one_hot_encode(self, labels):
        label_dict = {label: idx for idx, label in enumerate(pd.Series(labels).unique())}
        return pd.Series(labels).map(label_dict)

    def _calculate_drop_prob(self):
        print("Calculating drop probabilities...")
        tic = time.time()
        zero_ratio = np.mean(self.data == 0, axis=0)
        positive_mean = np.where(self.data > 0, self.data, np.nan)
        positive_mean = np.nanmean(positive_mean, axis=0)
        positive_mean = np.nan_to_num(positive_mean)  
        with np.errstate(divide='ignore', invalid='ignore'):
            drop_prob = zero_ratio / (positive_mean + 1)
        max_prob = np.nanmax(drop_prob)
        if max_prob > 0:
            drop_prob /= max_prob
        else:
            drop_prob = np.zeros_like(drop_prob)
        toc = time.time()
        print(f"Drop probability calculation took {toc - tic:.2f} seconds.")
        return drop_prob
    def normalize(self, x):
        x = np.clip(x, 0, 100)
        return np.log2(x + 1) / np.log2(100)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        sample = self.sample_list[index]
        data = self.data.loc[sample]
        label = self.label.loc[sample] if sample in self.label.index else 0
        label_onehot = self.label_onehot.iloc[index]
        data = torch.tensor(data.values, dtype=torch.float32)
        data = F.pad(data, (0, self.shape - data.shape[0]), 'constant', 0)
        data = self.normalize(data)
        return data, label, sample, label_onehot

