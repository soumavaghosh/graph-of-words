from input_data import InputData
import os, pickle, config
from graph_struct import Graph
from tqdm import tqdm
from joblib import Parallel, delayed
import pandas as pd
from collections import Counter
from model import GAT
import torch
import torch.nn as nn
import torch.optim as optim
from load_data import read_data
from random import shuffle

class graph_of_words:
    def __init__(self):
        # files = config.files
        # if 'input_data.p' in files:
        #     with open('../data/input_data.p', 'rb') as f:
        #         self.inputdata = pickle.load(f)
        # else:
        #     data, label = self.read_imdb_data()
        #     self.inputdata = Parallel(n_jobs=-1)(delayed(InputData)(data[i], label[i]) for i in tqdm(range(len(label)), desc='Cleaning data'))
        #     with open('../data/input_data.p', 'wb') as f:
        #         pickle.dump(self.inputdata, f)
        #
        # if 'graph_data.p' in files:
        #     with open('../data/graph_data.p', 'rb') as f:
        #         self.graph_data = pickle.load(f)
        # else:
            # manager = Manager()
            # word2id = manager.list(word2id)
            # word_set = set()
            # for w in self.inputdata:
            #     word_set = word_set.union(w.words)
            # word2id = {j: i for i, j in enumerate(list(word_set))}
            # word2id = [word2id]*len(self.inputdata)
        #     self.graph_data = Parallel(n_jobs=-1)(delayed(Graph)(d.data) for d in tqdm(self.inputdata, desc='Creating graphs and sub-graphs'))
        #     with open('../data/graph_data.p', 'wb') as f:
        #         pickle.dump(self.graph_data, f)
        # sub_dict = []
        # for g in self.graph_data:
        #     sub_dict.extend(g.sub_graph_mem.values())
        # self.sub_dict = Counter(sub_dict)
        # print('done')

        data_train, data_test, n_nodes, nclass = read_data(config.dataset)
        print(f'Number of nodes - {n_nodes}\n Number of class - {nclass}')
        self.train_graph_data = Parallel(n_jobs=-1)(delayed(Graph)(d) for d in tqdm(data_train, desc='Creating graphs train'))
        #self.test_graph_data = Parallel(n_jobs=-1)(delayed(Graph)(d) for d in tqdm(data_test, desc='Creating graphs test'))

        #self.train_graph_data = [g for g in self.train_graph_data if len(g.words) <= 200]
        #self.test_graph_data = [g for g in self.test_graph_data if len(g.words) <= 200]

        self.model = GAT(config.dim, n_nodes, nclass, config.dropout, config.alpha, config.nheads, config.n_units)
        optimizer = optim.Adam(self.model.parameters(), lr = config.lr, weight_decay = config.weight_decay)
        loss_fn = nn.CrossEntropyLoss()

        print(self.model)

        cuda = 'cuda' if torch.cuda.is_available() else 'cpu'
        if cuda=='cuda':
            self.model.cuda()

        #print(f'Accuracy train - {self.evaluate(self.train_graph_data, cuda)}')
        #print(f'Accuracy test - {self.evaluate(self.test_graph_data, cuda)}')
        for e in range(1, config.epoch+1):
            self.model.train()
            shuffle(self.train_graph_data)
            error_lst = []
            for g in tqdm(self.train_graph_data, desc = f'Training epoch {e}'):
                optimizer.zero_grad()
                label = g.label
                node_emb = self.model(torch.tensor(g.words, device=cuda), torch.tensor(g.adj, device=cuda),
                                      torch.tensor(g.fiedler_encoding, device=cuda))
                out = self.model.classify(node_emb)
                loss = loss_fn(out, torch.tensor([label], dtype=torch.long, device=cuda))
                loss.backward()
                optimizer.step()
                if cuda=='cuda':
                    error_lst.append(loss.data.cpu().numpy())
                else:
                    error_lst.append(loss.data.numpy())
            print(f'Epoch {e} loss - {sum(error_lst)/len(error_lst)}')
            print(f'Accuracy train - {self.evaluate(self.train_graph_data, cuda)}')
            #print(f'Accuracy test - {self.evaluate(self.test_graph_data, cuda)}')

    def evaluate(self, data, cuda):
        self.model.eval()
        acc = []
        for g in tqdm(data, desc=f'Evaluating'):
            label = g.label
            node_emb = self.model(torch.tensor(g.words, device=cuda), torch.tensor(g.adj, device=cuda),
                                  torch.tensor(g.fiedler_encoding, dtype = torch.float32, device=cuda))
            out = self.model.classify(node_emb)
            out = torch.argmax(out)
            if cuda=='cuda':
                out = int(out.data.cpu().numpy())
            else:
                out = int(out.data.numpy())
            acc.append(out==label)
        return sum(acc)/len(acc)

    def read_amazon_data(self):
        with open('../data/train.ft.txt', 'r', encoding='utf-8') as f:
            data = f.readlines()

        with open('../data/test.ft.txt', 'r', encoding='utf-8') as f:
            data.extend(f.readlines())

        data = data[:10000]
        data = [x[11:] for x in data]
        label = [0 if x.startswith('__label__1 ') else 1 for x in data]
        return data, label

    def read_imdb_data(self):
        data = pd.read_csv('../data/IMDB Dataset.csv', header = 0)
        label = list(data['sentiment'])
        label = [0 if x=='positive' else 1 for x in label]
        data = list(data['review'])

        return data, label

if __name__=='__main__':
    model = graph_of_words()

