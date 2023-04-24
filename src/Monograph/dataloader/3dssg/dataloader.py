import json
import pandas as pd
import numpy as np
from timeit import default_timer
import pdb
import pickle
import torch
from PIL import ImageColor
from torch_geometric.data import Data


class ssg_loader:
    def __init__(self, data_path, relationships_file, objects_file, remove_semantics=True, nan=-1):
        self.data_path = data_path
        self.df_relationships = self.load_json(relationships_file)
        self.df_objects = self.load_json(objects_file)
        self.nan=nan
        self.remove_semantics=remove_semantics
        self.rels={'attached to' : 0.3,
            'behind': 1.2,
            'bigger than': nan,
            'brighter than': nan,
            'build in': .3,
            'close by': .6,
            'darker than': nan,
            'front': 1.2,
            'hanging in':.4,
            'hanging on':.4,
            'higher than':1,
            'leaning against':.3,
            'left':1.2,
            'lower than':1,
            'lying on':.3,
            'right':1.2,
            'same color': nan,
            'same material': nan,
            'same object type': nan,
            'same shape': nan,
            'same state': nan,
            'same texture': nan,
            'smaller than': nan,
            'standing in': .3,
            'standing on': .4,
            'supported by': .3,
            'part of': .3,
            'more comfortable than':nan,
            'inside':.3,
            'same symmetry as':nan,
            'same as':nan,
            'connected to':.3,
            'belonging to':nan,
            'cover':.3,
            'lying in':.3,
            'messier than':.3,
            'cleaner than':.3,
            'fuller than':nan,
            'more closed':nan,
            'more open':nan,
            }


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
        number_edges = len(df.loc[df.scan==scan]['relationships'].iat[0])
        edge_index = np.zeros((2, number_edges), int)
        edge_attr = np.zeros((number_edges))
        if self.remove_semantics:
            for edge in range(number_edges):
                attribute=self.rels[df.loc[df.scan==scan]['relationships'].iat[0][edge][3]]
                edge_attr[edge] = attribute
                if attribute==self.nan:
                    continue
                edge_ids = df.loc[df.scan==scan]['relationships'].iat[0][edge][:2]
                edge_index_start = np.where(correspondance_matrix[1,:]==edge_ids[0])[0][0]
                edge_index[0, edge] = edge_index_start
    
                edge_index_end = np.where(correspondance_matrix[1,:]==edge_ids[1])[0][0]
                edge_index[1, edge] = edge_index_end
            idx=edge_attr!=self.nan
            edge_index=edge_index.T[idx].T
            edge_attr =edge_attr[idx]            
            
        else:
            for edge in range(number_edges):
                edge_ids = df.loc[df.scan==scan]['relationships'].iat[0][edge][:2]
                edge_index_start = np.where(correspondance_matrix[1,:]==edge_ids[0])[0][0]
                edge_index[0, edge] = edge_index_start
    
                edge_index_end = np.where(correspondance_matrix[1,:]==edge_ids[1])[0][0]
                edge_index[1, edge] = edge_index_end
                
                edge_attr[edge] = int(df.loc[df.scan==scan]['relationships'].iat[0][edge][2])
        
        return torch.tensor(edge_index, dtype=torch.float), torch.tensor(edge_attr, dtype=torch.float)
    
    def create_node_feature_matrix(self, scan, df):
        # given: scan, dataframe
        # return:   x node feature matrix with shape [num_nodes, num_node_features]
        num_nodes = len(df.loc[df.scan==scan]['objects'].iat[0])
        num_node_features = 7
        x = np.zeros((num_nodes, num_node_features))

        correspondance_matrix = np.zeros((2, num_nodes))
        for object in range(num_nodes):
            correspondance_matrix[0, object] = object
            correspondance_matrix[1, object] = int(df.loc[df.scan==scan]['objects'].iat[0][object]['id'])

            x[object, 0] = int(df.loc[df.scan==scan]['objects'].iat[0][object]['nyu40'])
            x[object, 1] = int(df.loc[df.scan==scan]['objects'].iat[0][object]['eigen13'])
            x[object, 2] = int(df.loc[df.scan==scan]['objects'].iat[0][object]['rio27'])
            x[object, 3] = int(df.loc[df.scan==scan]['objects'].iat[0][object]['global_id'])
            rgb = ImageColor.getcolor(df.loc[df.scan==scan]['objects'].iat[0][object]['ply_color'], 'RGB')
            x[object, 4:] = rgb
        
        return torch.tensor(x, dtype=torch.float), correspondance_matrix
    
    def create_geometric_graphs(self):
        # given:    class attributes
        # return:   an array of torch_geometric.Data objects
        geometric_list = []
        pdb.set_trace()
        for scan in self.df_relationships['scan'][:].items():
            if not (self.df_objects.loc[self.df_objects['scan']==scan[1]].index).empty:
                x, cor_mat = self.create_node_feature_matrix(scan[1], self.df_objects)
                edge_index, edge_attributes = self.create_edge_index_attributes(scan[1], self.df_relationships, cor_mat)

                graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attributes)

                geometric_list.append(graph_data)
        
        return geometric_list






# get the scan id from the relationships file and use it to search for the corresponding scan in the objects file
# discard the entry if the corresponding scan does not exist
def save_geometric_graphs():
    # data_path = '../../../../data/3dssg/' # toy files
    # loader = ssg_loader(data_path, 'toy_relationships.json', 'toy_objects.json')

    # data_path = '/cluster/home/juergeal/Monograph-Image-Retrieval/data/3dssg/'
    data_path = '/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/3dssg/'
    loader = ssg_loader(data_path, 'relationships.json', 'objects.json')

    graphs = loader.create_geometric_graphs()

    graph_file = f'{data_path}geometric_graph.pt'

    torch.save(graphs, graph_file)
    # new_graphs = torch.load(graph_file)

save_geometric_graphs()
