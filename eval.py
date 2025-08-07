from IMATAC import IMATAC, PeakDataset, H5adPeakDataset
import argparse
import numpy as np
from torch.utils.data import DataLoader
import torch
import json
import anndata as ad
from scipy.sparse import csr_matrix

def run(args):
    seed = 18
    np.random.seed(seed)
    torch.manual_seed(seed)
    path = args.dataset
    mark = args.mark
    with open(args.config, 'r', encoding='utf-8') as file:
        params = json.load(file)
    if args.format == 'h5ad':
        dataset = H5adPeakDataset(path)
    elif args.format == 'count matrix':
        dataset = PeakDataset(path)
    loader = DataLoader(dataset, 32, shuffle=False, num_workers=4)
    dropout_rate = torch.tensor(dataset.drop_prob, dtype=torch.float32).to('cuda:0')
    model = IMATAC(in_feature=dataset.shape, num_components=dataset.num_classes, dropout_rate=dropout_rate, params=params, dropout=True).to('cuda:0')
    model.load_state_dict(torch.load(args.model_path, map_location='cuda:0'))
    model = model.eval()
    imputed = model.imputation(loader, 'cuda:0', output_scale=args.output_scale)
    mat = csr_matrix(imputed.values)
    adata = ad.AnnData(mat)
    adata.var_names = imputed.columns
    adata.obs_names = imputed.index
    adata.write_h5ad(f'/root/autodl-fs/imputed_IMATAC_{mark}.h5ad')
    # imputed.to_csv(f'/root/autodl-fs/IMATAC-main/output/imputed_IMATAC_{mark}.txt', sep='\t', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Evaluate IMATAC3D model and output imputed results.')
    parser.add_argument('--config', type=str,
                        default='/root/autodl-tmp/IMATAC-main/config.JSON',
                        help='Path to the network parameters.')
    parser.add_argument('--dataset', type=str, 
                        default='/root/autodl-fs/data/data_drop40_716/data_drop40_716.h5ad',
                        help='Path to the input dataset.')
    parser.add_argument('--mark', type=str, 
                        default='no_mask1', 
                        help='Identifier for the output files.')
    parser.add_argument('--format', type=str,
                        default='h5ad',
                        help='Format of the input files.')
    parser.add_argument('--model_path', type=str,
                        default='/root/autodl-tmp/IMATAC-main/output/IMATAC_no_mask1.pth',
                        help='Path to the trained model weights.')
    parser.add_argument('--output_scale', type=str,
                        choices=['log', 'raw'],
                        default='log',
                        help='Output scale: "log" for log-normalized, "raw" for original scale.')
    args = parser.parse_args()
    run(args)
