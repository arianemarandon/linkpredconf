import numpy as np
import torch
import random
import math
from torch_geometric.utils import negative_sampling 
from torch_geometric.data import DataLoader 

from .SEAL import SEALDataset

def do_edge_split(dataset, val_ratio=0.05, test_ratio=0.1, 
                directed=False, random_state=None):
    """
    split list of true/false edges into train / test / val samples
    """
    data = dataset[0]
    #random.seed(234)
    #torch.manual_seed(234)

    if random_state is not None: 
        random.seed(random_state)
        torch.manual_seed(random_state)

    
    num_nodes = data.num_nodes
    row, col = data.edge_index
    # Return upper triangular portion.
    if not directed:   
        mask = row < col
        row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0))) #row.size(0) = total number of edges (not repeated)
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    
    # Negative edges 
    num_neg_samples=int(row.size(0))
    neg_edge_index = negative_sampling(
                data.edge_index, num_nodes=num_nodes,
                num_neg_samples=num_neg_samples, 
                force_undirected= True if not directed else False)
    data.val_neg_edge_index = neg_edge_index[:, :n_v]
    data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t] 
    data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    return split_edge


def make_split_edge_conf(dataset, split_edge, calib_size=1000, directed=True): 
    
    """
    used to build calibration set for conformal link prediction
    
    this resamples all the false edges to make split into train / test / calib /val 
    """

    data = dataset[0]
    
    ntr = len(split_edge['train']['edge'])
    ntest = len(split_edge['test']['edge'])

    num_neg_samples = calib_size + ntr + ntest

    neg_edge_index = negative_sampling(
                data.edge_index, 
                num_nodes=data.num_nodes,
                num_neg_samples=num_neg_samples,
                method='dense', force_undirected= True if not directed else False)

    
    calib_edges = neg_edge_index[:, :calib_size]
    train_neg_edge_index = neg_edge_index[:, calib_size:calib_size+ntr]
    test_neg_edge_index = neg_edge_index[:, calib_size+ntr:] 

    split_edge['train']['edge_neg'] = train_neg_edge_index.t()
    split_edge['test']['edge_neg'] = test_neg_edge_index.t()
    
    split_edge['calib'] = {'edge':None, 'edge_neg':None}
    split_edge['calib']['edge'] = torch.Tensor([[]])
    split_edge['calib']['edge_neg'] = calib_edges.t()


def make_loader_SEAL(path, data, split_edge, split, num_hops, directed=False):
    """
    for SEAL
    saves dataset at path
    !!!if already exists, loads from path!!!
    
    Extracts enclosing subgraphs using data.edge_index
    """
    dataset = SEALDataset(path, data, split_edge, num_hops, percent=100, split=split, 
                 use_coalesce=False, node_label='drnl', 
                 ratio_per_hop=1.0, max_nodes_per_hop=None, directed=directed)
    
    shuffle=True if split=='train' else False 
    return DataLoader(dataset, batch_size=32, shuffle=shuffle)

def make_loaders_SEAL(path, data, split_edge, num_hops, directed=False):
    """
    for SEAL
    """
    #Extract local enclosing subgraphs 
    train_loader = make_loader_SEAL(path, data, split_edge, split='train', num_hops=num_hops, directed=directed)
    test_loader = make_loader_SEAL(path, data, split_edge, split='test', num_hops=num_hops, directed=directed)
    calib_loader = make_loader_SEAL(path, data, split_edge, split='calib', num_hops=num_hops, directed=directed)
    val_loader=None
    
    return train_loader, test_loader, calib_loader, val_loader

def get_fdp(ytrue, rejection_set):
    """
    ytrue: vector of size m indicating for each test point whether it is a null (0) or a non-null (1) 
    rejection_set: a list of the indexes (corresponding to rejections of some procedure) 

    Return: the FDP and the TDP for the rejection set <rejection_set>
    """

    if rejection_set.size:
        fdp = np.sum(ytrue[rejection_set] == 0) / len(rejection_set)
        tdp = np.sum(ytrue[rejection_set] == 1) / np.sum(ytrue==1)
    else: 
        fdp=0
        tdp=0
    return fdp, tdp