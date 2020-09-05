import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import graphAttentionHead
import pickle

class graphEncoder(nn.Module):

    def __init__(self, dim, dropout, alpha, nheads):

        super(graphEncoder, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.dim = dim

        self.attentions = [graphAttentionHead(dim, dropout, alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.linear1 = nn.Linear(dim * nheads, dim, bias=True)
        self.layernorm = nn.LayerNorm(dim)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, nodes, adj):
        #node = F.dropout(nodes, self.dropout, training=self.training)
        nodes_n = torch.cat([att(nodes, adj) for att in self.attentions], dim=1)
        nodes_n = F.dropout(nodes_n, self.dropout, training=self.training)
        nodes_n = F.elu(self.linear1(nodes_n))
        nodes = nodes + nodes_n
        nodes = self.layernorm(nodes)
        return nodes