import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import graphAttentionHead
import pickle

class GAT(nn.Module):

    def __init__(self, dim, n_nodes, nclass, dropout, alpha, nheads):

        super(GAT, self).__init__()
        self.dropout = dropout
        self.alpha = alpha
        self.node_embedding = nn.Embedding(n_nodes, dim)

        self.attentions = [graphAttentionHead(dim, dropout, alpha) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

        self.linear1 = nn.Linear(dim * nheads, dim, bias=True)
        self.att_weight = nn.Parameter(torch.zeros(size=(dim, 1)))
        nn.init.xavier_uniform_(self.att_weight.data, gain=1.414)
        self.linear2 = nn.Linear(dim, nclass, bias=True)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, node, adj):
        node = self.node_embedding(node)
        adj = self.node_embedding(adj)

        node = F.dropout(node, self.dropout, training=self.training)
        node = torch.cat([att(node, adj) for att in self.attentions], dim=1)
        node = F.dropout(node, self.dropout, training=self.training)
        node = F.elu(self.linear1(node))
        return node

    def classify(self, input):
        weight = self.leakyrelu(torch.mm(input, self.att_weight))
        weight = F.softmax(torch.transpose(weight, 0, 1), dim = 1)
        out = torch.mm(weight, input)
        out = self.linear2(out)
        return out
