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

# params
focal_len = 1000

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
medians = []
for id in sem_uniq:
    xs = np.array(ids_3d_points[id])[:,0]
    ys = np.array(ids_3d_points[id])[:,1]
    zs = np.array(ids_3d_points[id])[:,2]
    medians.append([np.median(xs), np.median(ys), np.median(zs)-2])

# Visualize:
pcd_o3d = o3d.geometry.PointCloud()  # create point cloud object
medians_o3d = o3d.geometry.PointCloud()  # create point cloud object

pcd_o3d.points = o3d.utility.Vector3dVector(pcd)  # set pcd_np as the point cloud points
medians_o3d.points = o3d.utility.Vector3dVector(medians)  # set pcd_np as the point cloud points

o3d.visualization.draw_geometries([pcd_o3d, medians_o3d])