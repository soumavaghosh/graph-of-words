import os

files = os.listdir('../data')
dataset = '20ng'

epoch = 50
window = 3
sub_graph_range = 2

dim = 64
dropout = 0.2
alpha = 0.2
nheads = 8
n_units = 2
weight_decay = 5e-4
lr = 0.005
dist_thresh = 0.5