"""
Created on Sun Apr 9 2023

@author: Levi Lingsch
"""
import torch
import pandas as pd
import numpy as np
import pdb

class PipelineFeatures():
    def __init__(self, model, threshold):
        # self.graph = graph
        if torch.cuda.is_available():
            self.device='cuda:0'
        else:
            self.device='cpu'

        self.model = torch.load(model).to(self.device)
        self.loss = torch.nn.MSELoss()

        self.threshold = threshold

    def threshold_filter(self, graph):
        # given:    a graph
        # return:   a new graph with edge indices where the edge length is lower than some threshold
        indices = torch.where(graph.edge_attribute < self.threshold)[1]
        new_edge_attr = torch.index_select(graph.edge_attribute, 1, indices)
        new_edge_index = torch.index_select(graph.edge_index, 1, indices)

        graph.edge_attribute = new_edge_attr
        graph.edge_index = new_edge_index

        return graph

    def get_features(self, graph):
        graph.to(self.device)
        graph = self.threshold_filter(graph)
        self.model.eval()
        with torch.no_grad():
            features = self.model(graph.x, graph.edge_index, edge_attr = graph.edge_attribute)
            if len(features) == 0:
                print(graph.y)
                return -1
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
            features = self.get_features(graph)
            if features != -1:
                database_features.append(features)
        return database_features


def clean_up_graphs():
    import os
    from timeit import default_timer
    # from torch_geometric.loader import DataLoader
    
    # load the model
    print('Loading model...')
    model = 'models/trained_on_large_hypersim_set'
    pipeline = PipelineFeatures(model)

    path='../../../data/hypersim_graphs/'
    scene_files = os.listdir(path)


    # saving non empty graphs for single example
    # scene = 'ai_026_018_00_graphs.pt'
    # scene_graphs = torch.load(f'{path}{scene}')
    # new_graphs = []
    # for graph in scene_graphs:
    #     if graph.x.shape[0] > 0:
    #         new_graphs.append(graph)
    #     else:
    #         print(f'skipping {graph.y}')
    # pdb.set_trace()
    # print(len(new_graphs))
    # torch.save(new_graphs, f'{path}{scene}')

    # find settings with small graphs
    # for scene in scene_files:
    #     scene_graphs = torch.load(f'{path}{scene}')
    #     small_scene_count = 0
    #     for graph in scene_graphs:
    #         if graph.x.shape[0] < 4:
    #             small_scene_count += 1
    #     if small_scene_count >= 20:
    #         print(scene)

    # save graphs, removing the ones which are empty
    # for scene in scene_files:
    #     scene_graphs = torch.load(f'{path}{scene}')
    #     new_graphs = []

    #     for graph in scene_graphs:
    #         if graph.x.shape[0] > 0:
    #             new_graphs.append(graph)
    #         else:
    #             print(f'skipping {graph.y}')

    #     torch.save(new_graphs, f'{path}{scene}')
# clean_up_graphs()

def run_pipeline_example(threshold, path):
    import os
    from timeit import default_timer
    import matplotlib.pyplot as plt

    test_samples = 10
    positive_samples = 30
    
    # load the model
    print('Loading model...')
    model = 'pretrained_on_3dssg'
    # threshold = 1
    # model = 'trained_on_large_hypersim_set'
    pipeline = PipelineFeatures(f'models/{model}', threshold)

    # path='../../../data/hypersim_graphs/'
    # path = '/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/hypersim_graphs_sem_resnet50/'
    scene_files = os.listdir(path)
    # create several "test" scenes where we want to search for images of the same room
    print('Creating test scenes...')
    test_scene_1 = torch.load(f'{path}{scene_files[-1]}')[:test_samples]
    test_scene_2 = torch.load(f'{path}{scene_files[-5]}')[:test_samples]
    test_scene_3 = torch.load(f'{path}{scene_files[-10]}')[:test_samples]
    test_scenes = test_scene_1
    test_scenes.extend(test_scene_2)
    test_scenes.extend(test_scene_3)
    print(f"Number of test scenes: {len(test_scenes)}")

    # create the scenes to create the database from which we will need to search
    print('Creating scene database...')
    database_scenes = torch.load(f'{path}{scene_files[-1]}')[-positive_samples:]       # has matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-5]}')[-positive_samples:]) # has matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-10]}')[-positive_samples:]) # has matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-3]}')[:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-4]}')[:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-6]}')[:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-7]}')[:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-14]}')[:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-15]}')[:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-16]}')[:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-17]}')[:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-18]}')[:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-19]}')[:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-20]}')[:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-21]}')[:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-22]}')[:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-23]}')[:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-24]}')[:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-25]}')[:] ) # has no matches
    database_scenes.extend(torch.load(f'{path}{scene_files[-26]}')[:] ) # has no matches
    print(f"Number of database scenes: {len(database_scenes)}")
    

    # create a dataset
    t1 = default_timer()
    feature_database = pipeline.feature_database(database_scenes)
    test_features = pipeline.get_features(test_scene_1[0])
    t2 = default_timer()
    print(f"Time to get database feature: {t2-t1:4f}")


    max_k = 10
    precision_at_k = np.zeros(max_k)
    recall_at_k = np.zeros(max_k)
    maximum_recall_at_k = np.zeros(max_k)
    for k in range(max_k):
        num_tested = 0
        for test_graph in test_scenes:
            # retrieve the features and k best matches
            test_features = pipeline.get_features(test_graph)
            best_match = pipeline.n_best_from_features(test_features, feature_database, n=k+1)

            test_graph_scene = test_graph.y[0]
            if k == 9:
                print("scene to match")
                print(test_graph.y)
                print("predicted matches")

            local_recall = 0
            for match_index in range(k+1):
                # maximum_recall_at_k[k] += 1/(positive_samples)
                if best_match['Scene'].iloc[match_index] == test_graph_scene:
                    precision_at_k[k] += 1/(k+1)
                    # recall_at_k[k] += 1/(positive_samples)
                    local_recall = 1

            if k == 9:
                print(best_match)
                    # print(best_match['Scene'].iloc[match_index] best_match['Mid'].iloc[match_index] best_match['Frame'].iloc[match_index])
            
            recall_at_k[k] += local_recall
            
            
            num_tested += 1
        precision_at_k[k] /= num_tested
        recall_at_k[k] /= num_tested
        maximum_recall_at_k[k] /= num_tested

    # print(precision_at_k)
    # print(recall_at_k)
    # print(maximum_recall_at_k)

    # k = np.arange(1,11)
    # plt.figure()
    # # plt.plot(k, precision_at_k, color='blue', label='average precision')
    # plt.plot(k, recall_at_k, color='red', label='average recall')
    # # plt.plot(k, maximum_recall_at_k, '-.',color='red', label='maximum average recall')
    # plt.legend()
    # plt.grid()
    # plt.xlabel('k')
    # plt.ylabel('Value as a Fraction')
    # plt.title(f'Performance of {model} Model')
    # plt.savefig(f'training_plots/precision_and_recall_at_k_for_{model}.png')

    return recall_at_k, precision_at_k
        

thresholds = [2]
models = ['GT', 'ResNet50']
paths = ['../../../data/hypersim_graphs/', 
             '/cluster/project/infk/courses/252-0579-00L/group11_2023/datasets/hypersim_graphs_sem_resnet50/']
recalls = np.zeros((len(thresholds), 2, 10))
precision = np.zeros((len(thresholds), 2, 10))
for col in range (2):
    for row, threshold in enumerate(thresholds):
        path  = paths[col]
        model = f'{models[col]}_threshold:{threshold}'


        
        print(threshold)
        print(model)
        print(path)
        recalls[row, col, :], precision[row,col,:] = run_pipeline_example(threshold, path)
        # print('Recall at k')
        # for index in range(10):
            # print(recalls[row, col, index])

        # print('Precision at k')
        # for index in range(10):
            # print(precision[row, col, index])
