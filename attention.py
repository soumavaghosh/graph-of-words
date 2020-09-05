import torch
import torch.nn as nn
import torch.nn.functional as F

class graphAttentionHead(nn.Module):
    def __init__(self, dim, dropout, alpha):
        super(graphAttentionHead, self).__init__()
        self.dropout = dropout
        self.dim = dim
        self.alpha = alpha

        # self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2 * self.dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, node, adj):
        #h = torch.mm(input, self.W)
        N = adj.size()[0]

        a_input = torch.cat([node.repeat(N, 1), adj], dim=1)
        e = self.leakyrelu(torch.matmul(a_input, self.a))

        attention = F.softmax(e, dim=0)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        attention = torch.transpose(attention, 0,1)
        h_prime = torch.matmul(attention, adj)

        return F.elu(h_prime)