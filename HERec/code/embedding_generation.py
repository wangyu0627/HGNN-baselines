from dgl.nn.pytorch import DeepWalk
import numpy as np
import scipy.sparse as sp
import torch
import dgl
import torch
from torch.optim import SparseAdam
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

device = 'cuda:0'

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def tensorToDGL(G):
    gs = []
    for tensor in G:
        tensor = tensor.coalesce()
        indices = tensor.indices()
        values = tensor.values()
        graph = dgl.graph((indices[0], indices[1]), num_nodes=tensor.shape[0])
        graph.edata['edge_weight'] = values
        gs.append(graph)
    return gs

def deepwalk_acm():
    path = "../data/acm/"
    pap = sp.load_npz(path + "pap.npz")
    psp = sp.load_npz(path + "psp.npz")
    pap = sparse_mx_to_torch_sparse_tensor(normalize_adj(pap))
    psp = sparse_mx_to_torch_sparse_tensor(normalize_adj(psp))
    G = [pap, psp]
    gs = tensorToDGL(G)

    X = []
    for i, g in enumerate(gs):
        model = DeepWalk(g, walk_length=5, window_size=3).to(device)
        dataloader = DataLoader(torch.arange(g.num_nodes()), batch_size=128,
                                shuffle=True, collate_fn=model.sample)
        optimizer = SparseAdam(model.parameters(), lr=0.01)
        num_epochs = 20
        for epoch in range(num_epochs):
            for batch_walk in dataloader:
                batch_walk = batch_walk.to(device)
                loss = model(batch_walk)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            print('meta-path {:d} | Epoch {:d} | Loss {:.4f}'.format(i, epoch, loss))
        X.append(model.node_embed.weight.detach())

    torch.save(X, '../embedding/acm_node_embdeedings.pkl')

if __name__ == "__main__":
    deepwalk_acm()