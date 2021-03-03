import torch
import torch.nn as nn
import config
import torch.nn.functional as F

class graphAttentionHead(nn.Module):
    def __init__(self, dim, dropout, alpha):
        super(graphAttentionHead, self).__init__()
        self.dropout = dropout
        self.dim = dim
        self.alpha = alpha

        self.a = nn.Parameter(torch.zeros(size=(2 * self.dim, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        # self.q = nn.Parameter(torch.zeros(size=(self.dim, self.dim), dtype=torch.float32))
        # nn.init.xavier_uniform_(self.q.data, gain=1.414)
        # self.k = nn.Parameter(torch.zeros(size=(self.dim, self.dim), dtype=torch.float32))
        # nn.init.xavier_uniform_(self.k.data, gain=1.414)
        # self.v = nn.Parameter(torch.zeros(size=(self.dim, self.dim), dtype=torch.float32))
        # nn.init.xavier_uniform_(self.v.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    # def forward(self, node, adj):
    #     #h = torch.mm(input, self.W)
    #     N = adj.size()[0]
    #
    #     a_input = torch.cat([node.repeat(N, 1), adj], dim=1)
    #     e = self.leakyrelu(torch.matmul(a_input, self.a))
    #
    #     attention = F.softmax(e, dim=0)
    #     #attention = F.dropout(attention, self.dropout, training=self.training)
    #     attention = torch.transpose(attention, 0,1)
    #     h_prime = torch.matmul(attention, adj)
    #
    #     return F.elu(h_prime)

    def forward(self, nodes, dist):
        N = nodes.size()[0]

        # q = torch.matmul(nodes, self.q)
        # k = torch.matmul(nodes, self.k)
        # v = torch.matmul(nodes, self.v)
        #
        # e = torch.matmul(q, torch.transpose(k, 0, 1))/pow(self.dim, 0.5)

        a_input = torch.cat([nodes.repeat(1, N).view(N * N, -1), nodes.repeat(N, 1)], dim=1).view(N, -1, 2 * self.dim)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(dist < config.dist_thresh, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        #attention = F.dropout(attention, self.dropout, training=self.training)
        h_prime = torch.matmul(attention, nodes)

        return h_prime
