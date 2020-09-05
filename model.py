import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import graphAttentionHead
from encoder import graphEncoder

class GAT(nn.Module):

    def __init__(self, dim, n_nodes, nclass, dropout, alpha, nheads, n_units):

        super(GAT, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.dim = dim
        self.node_embedding = nn.Embedding(n_nodes, dim)

        self.encoder_units = [graphEncoder(dim, dropout, alpha, nheads) for _ in range(n_units)]
        for i, enc_unit in enumerate(self.encoder_units):
            self.add_module('encoder-unit_{}'.format(i), enc_unit)

        self.att_weight = nn.Parameter(torch.zeros(size=(dim, 1)))
        nn.init.xavier_uniform_(self.att_weight.data, gain=1.414)
        self.linear2 = nn.Linear(dim, nclass, bias=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, nodes, adj):
        nodes = self.node_embedding(nodes)

        for enc_unit in self.encoder_units:
            nodes = enc_unit(nodes, adj)

        return nodes


    def classify(self, input):
        weight = self.leakyrelu(torch.mm(input, self.att_weight))
        weight = F.softmax(torch.transpose(weight, 0, 1), dim = 1)
        out = torch.mm(weight, input)
        out = self.linear2(out)
        return out
