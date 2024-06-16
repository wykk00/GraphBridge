from ogb.nodeproppred import PygNodePropPredDataset
import torch_geometric.transforms as T
import torch
import pandas as pd
import numpy as np

PATH = './datasets/'

def get_raw_text_arxiv(use_text=False, seed=0):

    dataset = PygNodePropPredDataset(root=PATH,
        name='ogbn-arxiv', transform=T.ToSparseTensor())
    data = dataset[0]
    
    idx_splits = dataset.get_idx_split()
    train_mask = torch.zeros(data.num_nodes).bool()
    val_mask = torch.zeros(data.num_nodes).bool()
    test_mask = torch.zeros(data.num_nodes).bool()
    
    train_idx = idx_splits['train']
    val_idx = idx_splits['valid']
    test_idx = idx_splits['test']
    #train_idx = train_idx[torch.randperm(int(train_idx.size(0) * 0.2))]
    #val_idx = val_idx[torch.randperm(int(val_idx.size(0) * 0.2))]
    #test_idx = test_idx[torch.randperm(int(test_idx.size(0) * 0.2))]  
    
    
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    data.train_mask = train_mask
    data.val_mask = val_mask
    data.test_mask = test_mask
    data.y = data.y.squeeze()

    data.edge_index = data.adj_t.to_symmetric()
    if not use_text:
        return data, None

    nodeidx2paperid = pd.read_csv(
        './datasets/ogbn_arxiv/mapping/nodeidx2paperid.csv.gz', compression='gzip')

    raw_text = pd.read_csv('./datasets/ogbn_arxiv/raw/titleabs.tsv',
                           sep='\t', header=None, names=['paper id', 'title', 'abs'], skiprows=[0])
    raw_text = raw_text.dropna()

    # nodeidx2paperid['paper id'] = nodeidx2paperid['paper id'].astype('int64')
    raw_text['paper id'] = raw_text['paper id'].astype('int64')
    df = pd.merge(nodeidx2paperid, raw_text, on='paper id')

    text = []
    for ti, ab in zip(df['title'], df['abs']):
        t = 'Title: ' + ti + '\n' + 'Abstract: ' + ab
        text.append(t)
    return data, text
