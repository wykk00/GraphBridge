import torch
import os.path as osp
import pandas as pd
import torch.nn.functional as F
from torch_geometric.utils import add_self_loops, to_undirected


def get_raw_text_photo(use_text=False, seed=0):
    if osp.exists(f"./datasets/photo/photo_orig/photo.pt"):
        data = torch.load(f"./datasets/photo/photo_orig/photo.pt", map_location='cpu')
        data.y = data.label
        data.x = data.x.float() # Half into Float
        edge_index = to_undirected(data.edge_index)
        # edge_index, _ = add_self_loops(data.edge_index)
        data.edge_index = edge_index
        text = pd.read_csv('./datasets/photo/photo_orig/Electronics-Photo.txt', sep='\t', header=None)
        text = text[0]
        raw_texts = text.tolist()
        return data, raw_texts
    else:
        raise NotImplementedError('No existing photo dataset!')