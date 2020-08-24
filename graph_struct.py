import config

class Graph:
    def __init__(self, text):
        self.window = config.window
        self.sub_graph_range = config.sub_graph_range
        self.text = text
        self.words = set(text)
        self.graph = {i:set() for i in self.words}
        self.add_edges()
        self.sub_graph_mem = {}

        for n in self.words:
            for d in range(1, self.sub_graph_range+1):
                self.visited = {i:False for i in self.words}
                self.sub_graph_mem[(n,d)] = self.get_subgraph(n, d)

    def add_edges(self):
        n = len(self.text)
        for i in range(n):
            for j in range(i-self.window, i+self.window+1):
                if 0<=j<n:
                    if self.text[i]!=self.text[j]:
                        self.graph[self.text[i]].add(self.text[j])

    def get_subgraph(self, n, d):
        self.visited[n] = True
        neigh = sorted([x for x in self.graph[n] if self.visited[x]==False])
        res = str(n)

        if d==0:
            return res
        for i in neigh:
            self.visited[i] = True
        for i in neigh:
            tmp = self.get_subgraph(i,d-1)
            res+=f'({tmp})'

        return res

