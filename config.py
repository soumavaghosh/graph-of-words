import os

files = os.listdir('../data')
dataset = 'r52'

epoch = 20
embedding_dim = 128
window = 2
sub_graph_range = 2

dim = 128
dropout = 0.2
alpha = 0.2
nheads = 8
n_units = 2
weight_decay = 5e-4
lr = 0.005