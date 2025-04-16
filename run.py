from IMATAC import IMATAC, PeakDataset
import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch
# from IMATAC.dataset import PeakDataset
import json



def run(args):
    seed = 18
    np.random.seed(seed)
    torch.manual_seed(seed)
    path = args.dataset
    mark = args.mark
    num_components = args.num_components
    epoch = args.epoch
    with open(args.config, 'r', encoding='utf-8') as file:
        params = json.load(file)
    dataset = PeakDataset(path)
    loader = DataLoader(dataset, 32, shuffle = True, num_workers = 4)
    dropout_rate = torch.tensor(dataset.drop_prob, dtype=torch.float32).to('cuda:1')
    model = IMATAC(in_feature=dataset.shape, num_components=num_components, dropout_rate=dropout_rate, params=params, dropout=True).to('cuda:1')
    print(model)
    model.fit(loader, epochs=epoch, device='cuda:1')
    torch.save(model.state_dict(), f'./output/IMATAC_{mark}.pth')
    # model.load_state_dict(torch.load('model_sim_716.pth'))
    model = model.eval()
    imputed = model.imputation(loader, 'cuda:1')
    imputed.to_csv(f'./output/imputed_IMATAC_{mark}.txt', sep='\t', index = False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train an IMATAC model on a specified dataset.')
    parser.add_argument('--config', type=str,
                        default='./config.JSON',
                        help='Path to the network parameters.')
    parser.add_argument('--dataset', type=str, 
                        default='',
                        help='Path to the input dataset.')
    parser.add_argument('--epoch', type=int, 
                        default=200, 
                        help='Number of training epochs.')
    parser.add_argument('--mark', type=str, 
                        default='', 
                        help='Identifier for the output files.')
    parser.add_argument('--num_components', type=int, 
                        default=1, 
                        help='Number of components for the classifier.')
    args = parser.parse_args()
    run(args)
