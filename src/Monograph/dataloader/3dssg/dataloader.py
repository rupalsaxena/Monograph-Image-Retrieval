import json
import pandas as pd
import numpy as np
from timeit import default_timer
import pdb
import pickle

class ssg_loader:
    def __init__(self, data_path, relationships_file, objects_file):
        self.data_path = data_path
        self.df_relationships = self.load_json(relationships_file)
        self.df_objects = self.load_json(objects_file)

    def load_json(self, file_name):
        file = open(f'{self.data_path}{file_name}')
        json_data = json.load(file)

        # normalize the relationships file to a pandas array
        pd_file = pd.json_normalize(json_data, record_path=['scans'])

        return pd_file

    def create_graph_from_relationships(self, scan, df):
        number_edges = len(df['relationships'][scan])
        graph = np.zeros((2, number_edges), int)
        for edge in range(number_edges):
            graph[:, edge] = df['relationships'][scan][edge][:2]

        return graph

    def create_descriptor_graph(self, graph, scan, df, descriptor):
        descriptor_graph = np.empty_like(graph, dtype='object')
        number_edges = descriptor_graph.shape[1]
        
        for object in range(number_edges):
            current_df = pd.DataFrame(df.loc[scan]['objects'])

            start_object_id = graph[0, object]
            descriptor_graph[0, object] = current_df.loc[current_df['id']==str(start_object_id)][descriptor].item()

            end_object_id = graph[1, object]
            descriptor_graph[1, object] = current_df.loc[current_df['id']==str(end_object_id)][descriptor].item()

        return descriptor_graph

    def create_scan_graphs(self, descriptor_list):
        scan_graphs = []

        for scan in self.df_relationships['scan'][:].items():
            if not (self.df_objects.loc[self.df_objects['scan']==scan[1]].index).empty:
                object_index = self.df_objects.loc[self.df_objects['scan']==scan[1]].index.item()
                
                current_scan_graph = []
                edge_graph = self.create_graph_from_relationships(scan[0], self.df_relationships)
                current_scan_graph.append(edge_graph)

                for descriptor in descriptor_list:    
                    descriptor_graph = self.create_descriptor_graph(edge_graph, object_index, self.df_objects, descriptor)
                    current_scan_graph.append(descriptor_graph)
                
                current_scan_graph = np.asarray(current_scan_graph)

                scan_graphs.append(current_scan_graph)
        
        return scan_graphs



# get the scan id from the relationships file and use it to search for the corresponding scan in the objects file
# discard the entry if the corresponding scan does not exist
def main():
    # data_path = '../../../../data/3dssg/' # toy files
    data_path = '/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/3dssg/'
    t1 = default_timer()
    loader = ssg_loader(data_path, 'toy_relationships.json', 'toy_objects.json')
    descriptor_list = ["nyu40", "eigen13", "rio27", "ply_color"]
    graphs = loader.create_scan_graphs(descriptor_list)
    graph_file = f'{data_path}graph.npy'
    t2 = default_timer()

    with open(graph_file, 'wb') as fp:
        pickle.dump(graphs, fp)
    t3 = default_timer()
    # p = open(graph_file, "rb")
    # b = pickle.load(p)
    print(t2-t1)
    print(t3-t2)

    

main()