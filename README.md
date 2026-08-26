# ENDB-C

All network data are extracted using Python’s `pickle` library, as shown below:
with open(data_file_path, 'rb') as f:
        G = pickle.load(f)
