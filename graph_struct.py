import config
from scipy.sparse.csgraph import laplacian
from numpy.linalg import eig
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

class Graph:
    def __init__(self, data):
        self.window = config.window
        self.sub_graph_range = config.sub_graph_range
        #self.text = self.index_words(text, word2id)
        self.text = data[1]
        self.label = data[0]
        self.words = sorted([0]+list(set(self.text)))
        self.graph = {i:set() for i in self.words}
        self.adj = np.zeros((len(self.words), len(self.words)))
        self.adj_map = {j:i for i,j in enumerate(self.words)}
        self.add_edges()
        self.lap_mat = laplacian(self.adj, normed = True)
        self.fiedler_encoding = np.zeros((len(self.words), config.dim))
        self.dist_mat = self.get_dist_mat()
        self.fiedler_encoding = self.fiedler_encoding.astype(np.float32)

        # self.sub_graph_mem = {}
        #
        # for n in self.words:
        #     for d in range(1, self.sub_graph_range+1):
        #         self.visited = {i:False for i in self.words}
        #         self.sub_graph_mem[(n,d)] = self.get_subgraph(n, d)

    # def index_words(self, text, word2id):
    #     lst = []
    #     for d in text:
    #         lst.append(word2id[d])
    #     return lst

    def add_edges(self):
        n = len(self.text)
        for i in range(n):
            for j in range(i-self.window, i+self.window+1):
                if 0<=j<n:
                    #if i!=j:
                    self.graph[self.text[i]].add(self.text[j])
                    r, c = self.adj_map[self.text[i]], self.adj_map[self.text[j]]
                    self.adj[r][c]+=1
        self.adj[0,:] = 1
        self.adj[:,0] = 1

    def get_dist_mat(self):
        w, v = eig(self.lap_mat)
        v = np.real(v)
        w = [(w[i], i) for i in range(len(w))]
        w = sorted(w)
        val = list(v[:, w[1][1]])

        for i, k in enumerate(val):
            for j in range(config.dim//2):
                self.fiedler_encoding[i][2*j] =  np.sin(10*k/(0.5**(2*j/config.dim)))
                self.fiedler_encoding[i][2*j+1] = np.cos(10*k/(0.5**(2*j/config.dim)))

        thresh = int(np.sqrt(len(w)+1))
        col = [x[1] for x in w[1:]]
        v = v[:, col]

        dist = euclidean_distances(v[:, :thresh], v[:, :thresh])
        return dist

    # def get_subgraph(self, n, d):
    #     self.visited[n] = True
    #     neigh = sorted([x for x in self.graph[n] if self.visited[x]==False])
    #     res = str(n)
    #
    #     if d==0:
    #         return res
    #     for i in neigh:
    #         self.visited[i] = True
    #     for i in neigh:
    #         tmp = self.get_subgraph(i,d-1)
    #         res+=f'({tmp})'
    #
    #     return res

