import numpy as np
import scipy.sparse as sp
import torch as th
from sklearn.preprocessing import OneHotEncoder


def encode_onehot(labels):
    labels = labels.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(labels)
    labels_onehot = enc.transform(labels).toarray()
    return labels_onehot


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()


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
    indices = th.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = th.from_numpy(sparse_mx.data)
    shape = th.Size(sparse_mx.shape)
    return th.sparse.FloatTensor(indices, values, shape)


def load_acm(ratio, type_num):
    # The order of node types: 0 p 1 a 2 s
    path = "../data/acm/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return label, train, val, test


def load_dblp(ratio, type_num):
    # The order of node types: 0 a 1 p 2 c 3 t
    path = "../data/dblp/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]
    
    label = th.FloatTensor(label)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return label, train, val, test


def load_aminer(ratio, type_num):
    # The order of node types: 0 p 1 a 2 r
    path = "../data/aminer/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return label, train, val, test


def load_freebase(ratio, type_num):
    # The order of node types: 0 m 1 d 2 a 3 w
    path = "../data/freebase/"
    label = np.load(path + "labels.npy").astype('int32')
    label = encode_onehot(label)
    train = [np.load(path + "train_" + str(i) + ".npy") for i in ratio]
    test = [np.load(path + "test_" + str(i) + ".npy") for i in ratio]
    val = [np.load(path + "val_" + str(i) + ".npy") for i in ratio]

    label = th.FloatTensor(label)
    train = [th.LongTensor(i) for i in train]
    val = [th.LongTensor(i) for i in val]
    test = [th.LongTensor(i) for i in test]
    return label, train, val, test


def load_data(dataset, ratio, type_num):
    if dataset == "acm":
        data = load_acm(ratio, type_num)
    elif dataset == "dblp":
        data = load_dblp(ratio, type_num)
    elif dataset == "aminer":
        data = load_aminer(ratio, type_num)
    elif dataset == "freebase":
        data = load_freebase(ratio, type_num)
    return data
