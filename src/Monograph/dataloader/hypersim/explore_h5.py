import h5py
import numpy as np
import matplotlib.pyplot as plt

# # color
# path = '/scratch/userdata/llingsch/monograph/hypersim/downloads/ai_001_002/images/scene_cam_00_final_hdf5/'
# color = h5py.File(path+'frame.0000.color.hdf5')
# print(list(color.keys()))
# color_dset = color['dataset']
# print(color_dset.shape)
# print(color_dset[0,0,1])
# color_dset = np.array(color_dset)*255
# color_dset = color_dset.astype(np.uint8)
# plt.imshow(color_dset[:,:,:], interpolation='nearest')
# plt.show()

# color
path = '/scratch/userdata/llingsch/monograph/hypersim/downloads/ai_001_002/images/scene_cam_00_final_hdf5/'
color = h5py.File(path+'frame.0000.color.hdf5')
# print(list(color.keys()))
color_dset = h5py.File(path+'frame.0000.color.hdf5')['dataset']
print(color_dset.shape)
print(color_dset[0,0,1])
color_dset = np.array(color_dset)*255
color_dset = color_dset.astype(np.uint8)
plt.imshow(color_dset[:,:,:], interpolation='nearest')
plt.show()

## depth
# path = '/scratch/userdata/llingsch/monograph/hypersim/downloads/ai_001_001/images/scene_cam_00_geometry_hdf5/'
# depth = h5py.File(path+'frame.0000.depth_meters.hdf5')
# print(list(depth.keys()))
# depth_dset = depth['dataset']
# print(depth_dset.shape)
# print(depth_dset[0,0])
# plt.imshow(depth_dset, interpolation='nearest')
# plt.show()

## render id
# path = '/scratch/userdata/llingsch/monograph/hypersim/downloads/ai_001_001/images/scene_cam_00_geometry_hdf5/'
# render = h5py.File(path+'frame.0000.render_entity_id.hdf5')
# print(list(render.keys()))
# render_dset = render['dataset']
# print(render_dset.shape)
# print(render_dset)
# plt.imshow(render_dset, interpolation='nearest')
# plt.show()

# semantic
# path = '/scratch/userdata/llingsch/monograph/hypersim/downloads/ai_001_001/images/scene_cam_00_geometry_hdf5/'
# semantic = h5py.File(path+'frame.0000.semantic.hdf5')
# print(list(semantic.keys()))
# semantic_dset = semantic['dataset']
# print(semantic_dset.shape)
# print(semantic_dset[0,400])
# plt.imshow(semantic_dset, interpolation='nearest')
# plt.show()

# semantic_instance
# path = '/scratch/userdata/llingsch/monograph/hypersim/downloads/ai_001_001/images/scene_cam_00_geometry_hdf5/'
# semantic_instance = h5py.File(path+'frame.0000.semantic_instance.hdf5')
# print(list(semantic_instance.keys()))
# semantic_instance_dset = semantic_instance['dataset']
# print(semantic_instance_dset.shape)
# print(semantic_instance_dset[0,0])
# plt.imshow(semantic_instance_dset, interpolation='nearest')
# plt.show()
