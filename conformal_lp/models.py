import numpy as np 
import scipy.sparse as ssp
import torch
from torch.nn import BCEWithLogitsLoss
from torch_geometric.data import Data, DataLoader
from tqdm import tqdm

from .SEAL import get_pos_neg_edges

class SEAL_wrapper(object):
    def __init__(self, model, use_feature=True, use_edge_weight=False, emb=None, 
                num_epochs=100, lr=1e-3, log=False, num_log_epoch=10):
        self.model = model #pointed reference
        self.emb=emb
        self.use_feature=use_feature
        self.use_edge_weight=use_edge_weight

        self.num_epochs=num_epochs
        self.lr = lr
        self.log=log
        self.num_log_epoch=num_log_epoch
    
    def train(self, train_loader, val_loader=None):

        optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, self.model.parameters()), lr=self.lr)

        for epoch in range(self.num_epochs):

            self.model.train()

            avg_loss = 0
            for batch_idx, data in enumerate(train_loader):
                optimizer.zero_grad()
                x = data.x if self.use_feature else None
                edge_weight = data.edge_weight if self.use_edge_weight else None
                node_id = data.node_id if self.emb else None
                logits = self.model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
                loss = BCEWithLogitsLoss()(logits.view(-1), data.y.to(torch.float))
                loss.backward()
                optimizer.step()
                avg_loss += loss.item() 

            avg_loss /= batch_idx + 1

            if self.log: 
                if epoch==0 or (epoch+1)%self.num_log_epoch==0: 

                    if val_loader is not None:
                        val_acc = self.evaluate(val_loader)
                        print('Epoch: ', epoch+1, '; Avg loss: ', avg_loss, "; Val acc:", val_acc)
                    else: print('Epoch: ', epoch+1, '; Avg loss: ', avg_loss)


    def test(self, test_loader):

        self.model.eval()

        y_pred, y_true = [], []

        with torch.no_grad(): 
            for data in test_loader:
                x = data.x if self.use_feature else None
                edge_weight = data.edge_weight if self.use_edge_weight else None
                node_id = data.node_id if self.emb else None
                logits = self.model(data.z, data.edge_index, data.batch, x, edge_weight, node_id)
                y_pred.append(logits.view(-1))
                y_true.append(data.y.view(-1).to(torch.float))
        test_pred, test_true = torch.cat(y_pred), torch.cat(y_true)
        pos_test_pred = test_pred[test_true==1]
        neg_test_pred = test_pred[test_true==0]

        return pos_test_pred.numpy(), neg_test_pred.numpy()

    def evaluate(self, test_loader):
        self.model.eval()
        pos_test_pred, neg_test_pred = self.test(test_loader)

        pos_test_pred_y = np.where(pos_test_pred> 0., 1, 0)
        neg_test_pred_y = np.where(neg_test_pred> 0., 1, 0)

        acc = (np.sum(neg_test_pred_y==0) + np.sum(pos_test_pred_y==1))/(len(pos_test_pred)+len(neg_test_pred))

        return acc







def CN(A, edge_index, batch_size=100000):
    if edge_index.nelement()==0: return torch.Tensor([]), edge_index
    # The Common Neighbor heuristic score.
    link_loader = DataLoader(range(edge_index.size(1)), batch_size)
    scores = []
    for ind in tqdm(link_loader):
        src, dst = edge_index[0, ind], edge_index[1, ind]
        cur_scores = np.array(np.sum(A[src].multiply(A[dst]), 1)).flatten()
        scores.append(cur_scores)
    return torch.FloatTensor(np.concatenate(scores, 0)), edge_index



class CommonNeighbors(object):
    def __init__(self, data, split_edge): 
        
        self.data =data
        self.split_edge=split_edge
        edge_weight = torch.ones(data.edge_index.size(1), dtype=int)
        self.A_train = ssp.csr_matrix((edge_weight, (data.edge_index[0], data.edge_index[1])), 
                       shape=(data.num_nodes, data.num_nodes))

    def train(self, train_loader=None, val_loader=None):
        pass

    def test(self, split):
        pos_test_edge, neg_test_edge = get_pos_neg_edges(split, self.split_edge, 
                                                     self.data.edge_index, 
                                                     self.data.num_nodes)
        pos_test_pred, _ = CN(self.A_train, pos_test_edge)
        neg_test_pred, _ = CN(self.A_train, neg_test_edge)

        return pos_test_pred.numpy(), neg_test_pred.numpy()