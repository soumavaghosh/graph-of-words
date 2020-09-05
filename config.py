import os

files = os.listdir('../data')
dataset = '20ng'

epoch = 20
embedding_dim = 128
window = 2
sub_graph_range = 2

dim = 64
dropout = 0.2
alpha = 0.2
nheads = 8
weight_decay = 5e-4
lr = 0.005