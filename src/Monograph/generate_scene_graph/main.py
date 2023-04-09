import h5py
from GenerateSceneGraph import GenerateSceneGraph as GSG

def main():
    # img paths
    rgb_img = "tutorials/data/frame.0003.color.hdf5"
    depth_img = "tutorials/data/frame.0003.depth_meters.hdf5"
    semantic_img = "tutorials/data/frame.0003.semantic.hdf5"

    # reading files
    # with h5py.File(rgb_img, "r") as f: rgb = f["dataset"][:].astype("float32")
    with h5py.File(depth_img, "r") as f: depth = f["dataset"][:].astype("float32")
    with h5py.File(semantic_img, "r") as f: semantic = f["dataset"][:].astype("float32")

    _gsg = GSG(depth, semantic)
    graph = _gsg.get_torch_graph()
    print(graph)

if __name__ == '__main__':
    main()