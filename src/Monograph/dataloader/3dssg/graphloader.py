import pickle
import numpy as np
import pdb

class graph_loader:
    def __init__(self,path='/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/3dssg/'):
        graph_file = open(path+'graph.npy', "rb")
        self.data = pickle.load(graph_file)

        # ["nyu40", "eigen13", "rio27", "ply_color"]
    def load_selected(self, num_examples=0, nyu40=True, eigen13=True, rio27=True, ply=True):
        if num_examples == 0:
            num_examples = len(self.data)

        indices = [0]
        if nyu40 == True:
            indices.append(1)
        if eigen13 == True:
            indices.append(2)
        if rio27 == True:
            indices.append(3)
        if ply == True:
            indices.append(4)
        
        selected_graphs = []
        for graph in self.data:
            selected_graphs.append(np.take(graph, indices, axis=0))

        return selected_graphs
        
def run_example():
    p = graph_loader(path='../../../../data/3dssg/')
    nyu_data = p.load_selected(nyu40=True, eigen13=False, rio27=False, ply=False)
    print(nyu_data[0])

# run_example()