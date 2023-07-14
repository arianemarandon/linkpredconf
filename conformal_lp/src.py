import numpy as np
import torch


from .utils import do_edge_split, make_split_edge_conf

def process(dataset, calib_size, val_ratio=0.,test_ratio=0.1,directed=False):
    """
    dataset: pytorch geometric Dataset
    """
 
    split_edge = do_edge_split(dataset, 
                                val_ratio=val_ratio, test_ratio=test_ratio, directed=directed)

    make_split_edge_conf(dataset, split_edge,
                         calib_size=calib_size, directed=directed)

    data = dataset[0] #copy

    if directed:
        data.edge_index = split_edge['train']['edge'].t()
    else:
        train_edge_index = split_edge['train']['edge'].t()
        train_edge_index_comp = torch.stack([train_edge_index[1], train_edge_index[0]], dim=0)
        data.edge_index = torch.cat([train_edge_index, train_edge_index_comp], dim=1)
        
    data.num_nodes = dataset[0].num_nodes #to ensure number of nodes stays the same (when setting edge_index, num_nodes is updated)

    return data, split_edge



def conformal_link_prediction(train_loader, test_loader, calib_loader, val_loader, model, level):

    """
    model: has a .train() method, and .test() method returning the tuple (scores for true test edges, scores for false test edges)

    returns: indexes of rejections along with corresponding labels
    """
    #train model 
    model.train(train_loader=train_loader, val_loader=val_loader)

    #get scores
    pos_test_pred, neg_test_pred = model.test(test_loader)
    test_scores= np.concatenate([pos_test_pred, neg_test_pred])
    test_labels = np.array([1]*len(pos_test_pred) + [0]*len(neg_test_pred))

    _, null_scores = model.test(calib_loader)
    
    rej_set = adaptiveEmpBH(null_scores, test_scores, level=level, correction_type=None)

    return rej_set, test_labels


#then do fdp, tdp = get_fdp(test_labels, rej_set)


def adaptiveEmpBH(null_statistics, test_statistics, level, correction_type='storey', storey_threshold=0.5):

    pvalues = np.array([compute_pvalue(x, null_statistics) for x in test_statistics])

    if correction_type == "storey": 
        null_prop= storey_estimator(pvalues=pvalues, threshold=storey_threshold)
    elif correction_type == "quantile":
        null_prop= quantile_estimator(pvalues=pvalues, k0=len(pvalues)//2)
    else:
        null_prop=1

    lvl_corr = level/null_prop
 
    return BH(pvalues=pvalues, level= lvl_corr) 

def compute_pvalue(test_statistic, null_statistics):
    return (1 + np.sum(null_statistics >= test_statistic)) / (len(null_statistics)+1)

def storey_estimator(pvalues, threshold): 
    return (1 + np.sum(pvalues >= threshold))/ (len(pvalues)*(1-threshold)) 

def quantile_estimator(pvalues, k0): #eg k0=m/2
    m = len(pvalues)
    pvalues_sorted = np.sort(pvalues)
    return (m-k0+1)/ (m*(1-pvalues_sorted[k0]))

def BH(pvalues, level): 
    """
    Benjamini-Hochberg procedure. 
    """
    n = len(pvalues)
    pvalues_sort_ind = np.argsort(pvalues) 
    pvalues_sort = np.sort(pvalues) #p(1) < p(2) < .... < p(n)

    comp = pvalues_sort <= (level* np.arange(1,n+1)/n) 
    #get first location i0 at which p(k) <= level * k / n
    comp = comp[::-1] 
    comp_true_ind = np.nonzero(comp)[0] 
    i0 = comp_true_ind[0] if comp_true_ind.size > 0 else n 
    nb_rej = n - i0

    return pvalues_sort_ind[:nb_rej]