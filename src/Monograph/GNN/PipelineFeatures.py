"""
Created on Sun Apr 9 2023

@author: Levi Lingsch
"""
import torch
import pandas as pd
import pdb

class PipelineFeatures():
    def __init__(self, model):
        # self.graph = graph
        if torch.cuda.is_available():
            self.device='cuda:0'
        else:
            self.device='cpu'

        self.model = torch.load(model).to(self.device)
        self.loss = torch.nn.MSELoss()

    def get_features(self, graph):
        graph.to(self.device)
        self.model.eval()
        with torch.no_grad():
            features = self.model(graph.x, graph.edge_index, edge_attr = graph.edge_attribute)
            features = torch.max(features, 0).values
        
        return [features, graph.y]

    def compute_distance(self, anchor_features, comparison_features):
        # given:    two data objects; 1 to be queried against, 1 being compared to the query
        # return:   the distance between the two outputs of the model

        distance = self.loss(anchor_features, comparison_features)

        return distance.item()

    def n_best_from_features(self, test_features, database_features, n=5):
        # given:    the features being queried and the database of all features
        # return:   the best n matches

        test_scene_id = test_features[1]
        test_features_values = test_features[0]

        # use a tensor and then just sort for the lowest values
        # distances = torch.zeros(len(database_features))
        distance_df = []
        for iter, graph_features in enumerate(database_features):
            graph_features_values = graph_features[0]
            distance = self.compute_distance(test_features_values, graph_features_values)
            
            scene_id = graph_features[1][0]
            mid_id = graph_features[1][1]
            frame_id = graph_features[1][2]

            distance_df.append(
                {
                'Distance': distance,
                'Scene': scene_id,
                'Mid':  mid_id,
                'Frame': frame_id
                }
            )
            
        
        distance_df = pd.DataFrame(distance_df)
        distance_df = distance_df.sort_values(by=['Distance'])
        # sorted_distance, sorted_indeces = torch.sort(distances)
    
        return distance_df.head(n)
            
    def feature_database(self, database_graphs):
        # given:    the graph to find similar features of and the database of all graphs
        # return:   the best match
        database_features = []
        for graph in database_graphs:
            database_features.append(self.get_features(graph))
        return database_features


        

def run_pipeline_example():
    import os
    from timeit import default_timer
    # from torch_geometric.loader import DataLoader
    
    # load the model
    print('Loading model...')
    model = 'models/pretrained_on_3dssg'
    pipeline = PipelineFeatures(model)

    path='../../../data/hypersim_graphs/'
    scene_files = os.listdir(path)

    # create several "test" scenes where we want to search for images of the same room
    print('Creating test scenes...')
    test_scene_1 = torch.load(f'{path}{scene_files[-1]}')[:10]
    test_scene_2 = torch.load(f'{path}{scene_files[-2]}')[:10]
    test_scene_3 = torch.load(f'{path}{scene_files[-3]}')[:10]
    test_scenes = test_scene_1
    test_scenes.extend(test_scene_2)
    test_scenes.extend(test_scene_3)
    print(f"Number of test scenes: {len(test_scenes)}")

    # create the scenes to create the database from which we will need to search
    print('Creating scene database...')
    database_scenes = torch.load(f'{path}{scene_files[-1]}')[-40:]       # has matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-2]}')[-40:]) # has matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-3]}')[-40:]) # has matches
    database_scenes.extend(torch.load(f'{path}{scene_files[0]}')[5:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[1]}')[5:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[2]}')[5:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[3]}')[5:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[4]}')[5:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[5]}')[5:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[6]}')[5:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[7]}')[5:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[8]}')[5:] ) # has no matches
    print(f"Number of database scenes: {len(database_scenes)}")
    

    # create a dataset
    t1 = default_timer()
    feature_database = pipeline.feature_database(database_scenes)
    test_features = pipeline.get_features(test_scene_1[0])
    t2 = default_timer()
    print(f"Time to get database feature: {t2-t1:4f}")

    best_match = pipeline.n_best_from_features(test_features, feature_database)
    t3 = default_timer()
    print(f"Time to get best matches: {t3-t2:5f}")
    
    
    # print(f"Test image id: {test_features[1]}")
    # print(f"Best matches: {best_match}")

    num_tested = 0
    num_correct_1 = 0
    num_correct_2 = 0
    num_correct_3 = 0
    num_correct_4 = 0
    num_correct_5 = 0
    for test_graph in test_scenes:
        test_features = pipeline.get_features(test_graph)
        best_match = pipeline.n_best_from_features(test_features, feature_database)

        # top_match_scene = best_match['Scene'].iloc[0]
        test_graph_scene = test_graph.y[0]

        if best_match['Scene'].iloc[0] == test_graph_scene:
            num_correct_1 += 1
        else:
            print(f"{test_graph.y} confused for {best_match.iloc[0]}")
        if best_match['Scene'].iloc[1] == test_graph_scene:
            num_correct_2 += 1
        if best_match['Scene'].iloc[2] == test_graph_scene:
            num_correct_3 += 1
        if best_match['Scene'].iloc[3] == test_graph_scene:
            num_correct_4 += 1
        if best_match['Scene'].iloc[4] == test_graph_scene:
            num_correct_5 += 1
            
        num_tested += 1

    print(f"Accuracy against top match: {100*(num_correct_1 / num_tested):2f}%")
    print(f"Accuracy against match 2: {100*(num_correct_2 / num_tested):2f}%")
    print(f"Accuracy against match 3: {100*(num_correct_3 / num_tested):2f}%")
    print(f"Accuracy against match 4: {100*(num_correct_4 / num_tested):2f}%")
    print(f"Accuracy against match 5: {100*(num_correct_5 / num_tested):2f}%")
        


run_pipeline_example()