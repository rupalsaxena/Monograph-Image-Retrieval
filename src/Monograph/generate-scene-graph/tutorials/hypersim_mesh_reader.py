import h5py

def read_hdf5(path):
    with h5py.File(path,   "r") as f: variable_name   = f["dataset"][:]
    return variable_name

if __name__ == '__main__':
    nyu_labels_path = "data/mesh_objects_si.hdf5"
    nyu_labels = read_hdf5(nyu_labels_path)
    print("nyu label")
    print(nyu_labels)

    semantic_id_path = "data/mesh_objects_sii.hdf5"
    semantic_id = read_hdf5(semantic_id_path)
    print("semantic id")
    print(semantic_id)

