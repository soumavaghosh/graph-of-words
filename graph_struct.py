import config

class Graph:
    def __init__(self, data):
        self.window = config.window
        self.sub_graph_range = config.sub_graph_range
        #self.text = self.index_words(text, word2id)
        self.text = data[1]
        self.label = data[0]
        self.words = sorted(set(self.text))
        self.graph = {i:set() for i in self.words}
        self.adj = [[0]*len(self.words) for _ in range(len(self.words))]
        self.adj_map = {j:i for i,j in enumerate(self.words)}
        self.add_edges()
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
                    self.graph[self.text[i]].add(self.text[j])
                    r, c = self.adj_map[self.text[i]], self.adj_map[self.text[j]]
                    self.adj[r][c]+=1

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

