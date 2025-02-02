import torch
import torch.nn as nn
from torch.nn import Module
import torch.nn.functional as F
import math
from torch.nn.parameter import Parameter

def adj_matrix_weight_merge(A, adj_weight):
    """
    Multiplex Relation Aggregation
    """

    A_t = torch.stack(A, dim=2).to_dense()

    temp = torch.matmul(A_t, adj_weight)
    temp = torch.squeeze(temp, 2)

    return temp + temp.transpose(0, 1)


class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))#鏉冮噸鐭╅樀
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))#鍋忕Щ鍚戦噺
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        try:
            input = input.float()
        except:
            pass
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output
    #
    # def __repr__(self):
    #     return self.__class__.__name__ + ' (' \
    #            + str(self.in_features) + ' -> ' \
    #            + str(self.out_features) + ')'

class GCN(nn.Module):
    """
    A Two-layer GCN.
    """
    def __init__(self, nfeat, nhid, nclass, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x, adj, use_relu=True):
        x = self.gc1(x, adj)
        if use_relu:
            x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x

class MHGCN(nn.Module):
    def __init__(self, num_metapath, nfeat, nlayer, out, dropout):
        super(MHGCN, self).__init__()
        """
        # Multilayer Graph Convolution
        """
        self.gc_layers = nn.ModuleList()
        self.num_metapath = num_metapath
        self.gc1 = GraphConvolution(nfeat, out)
        for i in range(nlayer-1):
            self.gc_layers.append(GraphConvolution(out, out))

        self.dropout = dropout

        """
        Set the trainable weight of adjacency matrix aggregation
        """
        self.weight_b = torch.nn.Parameter(torch.FloatTensor(self.num_metapath, 1), requires_grad=True)
        torch.nn.init.uniform_(self.weight_b,a = 0,b = 0.1)

    def forward(self, feature, A, use_relu=True):
        final_A = adj_matrix_weight_merge(A, self.weight_b)
        try:
            feature = torch.tensor(feature.astype(float).toarray())
        except:
            try:
                feature = torch.from_numpy(feature.toarray())
            except:
                pass

        U = []
        # Output of single-layer GCN
        U1 = self.gc1(feature, final_A)
        U.append(U1)
        x = U1
        for layer in self.gc_layers:
            x = layer(x, final_A)
            U.append(x)
        Us = torch.stack(U, dim=1)  # (N, M, D * K)
        Us = torch.mean(Us, dim=1)

        # Average pooling
        return Us