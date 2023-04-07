import json
import pandas as pd
import numpy as np
from timeit import default_timer
import pdb
import pickle
import torch
from torch_geometric.data import Data


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

    def create_edge_index_attributes(self, scan, df, correspondance_matrix):
        # given: scan, dataframe
        # return:   edge_index in COO format with shape [2, num_edges]
        #           edge_attributes with shape [num_edges, num_edge_features]
        number_edges = len(df.loc[df.scan==scan]['relationships'].iat[-1])
        edge_index = np.zeros((2, number_edges), int)
        edge_attr = np.zeros((number_edges))
        for edge in range(number_edges):
            edge_ids = df.loc[df.scan==scan]['relationships'][1][edge][:2]

            edge_index_start = np.where(correspondance_matrix[1,:]==edge_ids[0])[0][0]
            edge_index[0, edge] = edge_index_start

            edge_index_end = np.where(correspondance_matrix[1,:]==edge_ids[1])[0][0]
            edge_index[1, edge] = edge_index_end
            
            edge_attr[edge] = df.loc[df.scan==scan]['relationships'][1][edge][2]

        return edge_index, edge_attr
    
    def create_node_feature_matrix(self, scan, df):
        # given: scan, dataframe
        # return:   x node feature matrix with shape [num_nodes, num_node_features]
        num_nodes = len(df.loc[df.scan==scan]['objects'].iat[-1])
        num_node_features = 5
        x = np.zeros((num_nodes, num_node_features))

        correspondance_matrix = np.zeros((2, num_nodes))

        for object in range(num_nodes):
            correspondance_matrix[0, object] = object
            correspondance_matrix[1, object] = int(df.loc[df.scan==scan]['objects'][0][object]['id'])

            x[object, 0] = int(df.loc[df.scan==scan]['objects'][0][object]['nyu40'])
            x[object, 1] = int(df.loc[df.scan==scan]['objects'][0][object]['eigen13'])
            x[object, 2] = int(df.loc[df.scan==scan]['objects'][0][object]['rio27'])
            x[object, 3] = int(df.loc[df.scan==scan]['objects'][0][object]['global_id'])
            x[object, 4] = int(df.loc[df.scan==scan]['objects'][0][object]['ply_color'][1:], 16)
        
        return x, correspondance_matrix
    
    def create_geometric_graphs(self):
        geometric_list = []

        for scan in self.df_relationships['scan'][:].items():
            if not (self.df_objects.loc[self.df_objects['scan']==scan[1]].index).empty:
                pdb.set_trace()
                x, cor_mat = self.create_node_feature_matrix(scan[1], self.df_objects)
                edge_index, edge_attributes = self.create_edge_index_attributes(scan[1], self.df_relationships, cor_mat)






# get the scan id from the relationships file and use it to search for the corresponding scan in the objects file
# discard the entry if the corresponding scan does not exist
def main():
    data_path = '../../../../data/3dssg/' # toy files
    # data_path = '/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/3dssg/'

    t1 = default_timer()
    loader = ssg_loader(data_path, 'toy_relationships.json', 'toy_objects.json')

    loader.create_geometric_graphs()


    # loader = ssg_loader(data_path, 'relationships.json', 'objects.json')
    # descriptor_list = ["nyu40", "eigen13", "rio27", "ply_color"]
    # graphs = loader.create_scan_graphs(descriptor_list)
    # graph_file = f'{data_path}graph.npy'
    # t2 = default_timer()


    # with open(graph_file, 'wb') as fp:
    #     pickle.dump(graphs, fp)
    # t3 = default_timer()
    # p = open(graph_file, "rb")
    # b = pickle.load(p)
    # print(t2-t1)
    # print(t3-t2)

    

main()