from input_data import InputData
import os, pickle, config
from graph_struct import Graph
from tqdm import tqdm
from joblib import Parallel, delayed

class graph_of_words:
    def __init__(self):
        files = config.files
        if 'input_data.p' in files:
            with open('../data/input_data.p', 'rb') as f:
                self.inputdata = pickle.load(f)
        else:
            data, label = self.read_amazon_data()
            self.inputdata = Parallel(n_jobs=2)(delayed(InputData)(x, label) for x in tqdm(data, desc='Cleaning data'))
            word_set = set()
            for w in self.inputdata.words:
                word_set = word_set.union(w)
            word2id = {i:j for i,j in enumerate(list(word_set))}
            Parallel(n_jobs=-1)(delayed(x.index_words)(word2id) for x in tqdm(self.inputdata, desc='Indexing data'))
            with open('../data/input_data.p', 'wb') as f:
                pickle.dump(self.inputdata, f)

        if 'graph_data.p' in files:
            with open('../data/graph_data.p', 'rb') as f:
                self.inputdata = pickle.load(f)
        else:
            self.graph_data = [Graph(d) for d in tqdm(self.inputdata.data, desc='Creating graphs and sub-graphs')]
            with open('../data/graph_data.p', 'wb') as f:
                pickle.dump(self.graph_data, f)

    def read_amazon_data(self):
        with open('../data/train.ft.txt', 'r', encoding='utf-8') as f:
            data = f.readlines()

        with open('../data/test.ft.txt', 'r', encoding='utf-8') as f:
            data.extend(f.readlines())

        label = [0 if x.startswith('__label__1 ') else 1 for x in data]
        return data, label

if __name__=='__main__':
    model = graph_of_words()

