import json

# class ssg_loader:
#     def __init__(self, path):
#         self.main_path = path

#     def load_object():
#         return 
    
f = open('/scratch/userdata/llingsch/monograph/3DSSG/objects.json')
objects = json.load(f)
print(objects['scans'][0]['objects'][0]['nyu40'])
f.close()

f = open('/scratch/userdata/llingsch/monograph/3DSSG/relationships.json')
relationships = json.load(f)
print(relationships['scans'][0]['relationships'][0])
f.close()