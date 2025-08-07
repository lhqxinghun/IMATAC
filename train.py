from IMATAC import IMATAC, PeakDataset, H5adPeakDataset
import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch
import json

def run(args):
    seed = 18
    np.random.seed(seed)
    torch.manual_seed(seed)
    path = args.dataset
    mark = args.mark
    epoch = args.epoch
    with open(args.config, 'r', encoding='utf-8') as file:
        params = json.load(file)
    if args.format == 'h5ad':
        dataset = H5adPeakDataset(path)
    elif args.format == 'count matrix':
        dataset = PeakDataset(path)
    loader = DataLoader(dataset, 32, shuffle=True, num_workers=4)
    dropout_rate = torch.tensor(dataset.drop_prob, dtype=torch.float32).to('cuda:0')
    model = IMATAC(in_feature=dataset.shape, num_components=dataset.num_classes, dropout_rate=dropout_rate, params=params, dropout=True).to('cuda:0')
    print(model)
    model.fit(loader, epochs=epoch, device='cuda:0')
    torch.save(model.state_dict(), f'/root/autodl-tmp/IMATAC-main/output/IMATAC_{mark}.pth')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an IMATAC3D model on a specified dataset.')
    parser.add_argument('--config', type=str,
                        default='/root/autodl-tmp/IMATAC-main/config.JSON',
                        help='Path to the network parameters.')
    parser.add_argument('--dataset', type=str, 
                        default='/root/autodl-fs/data/data_drop40_716/data_drop40_716.h5ad',
                        help='Path to the input dataset.')
    parser.add_argument('--epoch', type=int, 
                        default=100, 
                        help='Number of training epochs.')
    parser.add_argument('--mark', type=str, 
                        default='no_mask1', 
                        help='Identifier for the output files.')
    parser.add_argument('--format', type=str,
                        default='h5ad',
                        help='Format of the input files.')
    args = parser.parse_args()
    run(args)
