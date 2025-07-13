# import h5py
# import numpy as np

# batch_file = "D:/model_training/cache/batches/physio-deep-v1/for-training/real_batch1_raw.h5"

# with h5py.File(batch_file, "r") as f:
#     if 'dataset_label' in f.attrs:
#         label_str = f.attrs['dataset_label']
#         if isinstance(label_str, (bytes, np.bytes_)):
#             label_str = label_str.decode('utf-8')
    
#     print(label_str)

#==========================================================================================

import h5py
import sys

def print_attrs(obj, indent=0):
    for key, val in obj.attrs.items():
        print(" " * indent + f"  [attr] {key}: {val}")

def visit_hdf5(name, obj, indent=0):
    indent_str = " " * indent
    if isinstance(obj, h5py.Group):
        print(f"{indent_str}[Group] {name}")
        print_attrs(obj, indent + 2)
    elif isinstance(obj, h5py.Dataset):
        print(f"{indent_str}[Dataset] {name} shape={obj.shape} dtype={obj.dtype}")
        print_attrs(obj, indent + 2)

def print_structure(filename):
    print(f"\n== Structure of {filename} ==\n")
    with h5py.File(filename, "r") as f:
        def visitor(name, obj):
            depth = name.count('/')
            visit_hdf5(name, obj, indent=depth*2)
        f.visititems(visitor)
    print("\n== END OF STRUCTURE ==\n")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python print_hdf5_structure.py path/to/file.h5")
        sys.exit(1)
    print_structure(sys.argv[1])
