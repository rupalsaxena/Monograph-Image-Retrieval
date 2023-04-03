import h5py
import numpy as np
from matplotlib.pyplot import imshow
from matplotlib import pyplot as plt
import open3d as o3d

# file paths
rgb_img = "data/frame.0003.color.hdf5"
depth_img = "data/frame.0003.depth_meters.hdf5"
semantic_img = "data/frame.0003.semantic.hdf5"

# reading files
with h5py.File(rgb_img, "r") as f: rgb = f["dataset"][:].astype("float32")
with h5py.File(depth_img, "r") as f: depth = f["dataset"][:].astype("float32")
with h5py.File(semantic_img, "r") as f: semantic = f["dataset"][:].astype("float32")

# store graph data in form of object
class graph_obj:
    def __init__(self, fr_cord, to_cord, fr_id, to_id, frame_id=None, scene_id=None):
        self.from_coord = fr_cord
        self.to_coord = to_cord
        self.from_id = fr_id
        self.to_id = to_id
        self.frame_id = frame_id
        self.scene_id = scene_id

# params
focal_len = 26.2484493

# cal principal point of image
height, width = depth.shape
x0 = width//2
y0 = height//2

# finding unique ids in image
sem_uniq = list(set(np.matrix.flatten(semantic)))
if -1 in sem_uniq:
    sem_uniq.remove(-1)

# create dict for each sem id point cloud
ids_3d_points = {}
for id in sem_uniq:
    ids_3d_points[id] = []

# converting depth data to point cloud within semantic id
pcd = []
for i in range(height):
   for j in range(width):
        z = depth[i][j]
        x = (j - x0) * z / focal_len
        y = (i - y0) * z / focal_len
        pcd.append([x, y, z])

        sem_id = semantic[i][j]
        if sem_id != -1:
            ids_3d_points[sem_id].append([x,y,z])

# find medians of each semantic ids
medians = {}
for id in sem_uniq:
    xs = np.array(ids_3d_points[id])[:,0]
    ys = np.array(ids_3d_points[id])[:,1]
    zs = np.array(ids_3d_points[id])[:,2]
    medians[id] = [np.median(xs), np.median(ys), np.median(zs-2)]

# find distance between 2 medians and generate graph based on threshold
graphs = []
thresh_dist = 60
for fr_idx, fr_id in enumerate(sem_uniq):
    for to_idx, to_id in enumerate(sem_uniq):
        if to_idx>fr_idx:
            sq_dist = np.sum((np.array(medians[fr_id]) - np.array(medians[to_id]))**2)
            dist = np.sqrt(sq_dist)
            if dist < thresh_dist:
                graph = graph_obj(medians[fr_id], medians[to_id], fr_id, to_id)
                graphs.append(graph)


print(len(graphs))
# draw lines 
o3d_obj = []
for graph in graphs:
    point1 = graph.from_coord
    point2 = graph.to_coord
    points = [point1, point2]
    lines = [[1,2]]
    colors = [[1,0,0]]
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)
    o3d_obj.append(line_set)

# # Visualize:
pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
medians_o3d = o3d.geometry.PointCloud()  # create point cloud object

pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
medians_o3d.points = o3d.utility.Vector3dVector(list(medians.values()))  # set pcd_np as the point cloud points
med_color = [0,0,1]
median_colors = [med_color for i in range(len(medians.values()))]
medians_o3d.colors = o3d.utility.Vector3dVector(median_colors)

o3d_obj.append(pcd_o3d)
o3d_obj.append(medians_o3d)


o3d.visualization.draw_geometries(o3d_obj)