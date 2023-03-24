import json
import pandas as pd
import numpy as np
# class ssg_loader:
#     def __init__(self, path):
#         self.main_path = path

#     def load_object():
#         return 
    
# f = open('/scratch/userdata/llingsch/monograph/3DSSG/objects.json')
# objects = json.load(f)
# print(objects['scans'][0]['objects'][0]['nyu40'])
# f.close()

# f = open('/scratch/userdata/llingsch/monograph/3DSSG/relationships.json')
# relationships = json.load(f)
# print(relationships['scans'][0]['relationships'][0])
# f.close()



#####################
# I want to construct a simple data loader to get the nyu40 ID's for a list of relationships

# normalize the edge relationships along the scans
f_relationships = open('/scratch/userdata/llingsch/monograph/3DSSG/relationships.json')
relationships = json.load(f_relationships)
df_r = pd.json_normalize(relationships, record_path=['scans'])
df_r = df_r.sort_values('scan')
number_edges = len(df_r['relationships'][0])
graph = np.zeros((2, number_edges), int)
for edge in range(number_edges):
    graph[:,edge] = df_r['relationships'][0][edge][:2]
print(graph)
print(df_r.scan[0])

# normalize the node objects along the scans
f_objects = open('/scratch/userdata/llingsch/monograph/3DSSG/objects.json')
objects = json.load(f_objects)
df_o = pd.json_normalize(objects, record_path=['scans'])
df_o = df_o.sort_values('scan')
number_edges = len(df_o['objects'][0])
graph = np.zeros((2, number_edges), int)
for edge in range(number_edges):
    graph[:,edge] = df_o['objects'][0][edge][:2]
print(graph)

print(df_o.loc[df_o['scan']==df_r.scan[0]].objects)
current_objects = df_o.loc[df_o['scan']==df_r.scan[0]]['objects'].item()
current_df = pd.DataFrame(current_objects.item())
# len(current_objects)

current_df.loc[current_df['id']=='1']['eigen13'].item()