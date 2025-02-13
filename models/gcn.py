import torch.nn as nn
import torch.optim as optim
import torch
from torch.nn.parameter import Parameter
import math

class GCN(nn.Module):
    def __init__(self, nfeat, nhid, nout, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nout)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, adj):
        # x: [b, n_sensors, nfeat]
        # adj: adjacent matrix
        x = self.relu(self.gc1(x, adj)) # [b, n_sensors, nhid]
        x = self.dropout(x)
        x = self.gc2(x, adj) # [b, n_sensors, nout]
        return x.permute(0, 2, 1) # [b, nout, n_sensors]


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features), requires_grad=True)
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features), requires_grad=True)
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight) # [b, n_sensor, out_features]
        output = torch.matmul(adj, support)  # [b, n_sensor, out_features]
        if self.bias is not None:
            return output + self.bias
        else:
            return output