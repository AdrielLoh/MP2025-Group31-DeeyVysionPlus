import numpy as np
import os
import glob
import random

BATCH_DIR = r'C:\deepvysion\preprocessed-1'

# List all batch files
batch_files = sorted(glob.glob(os.path.join(BATCH_DIR, 'batch_*_frames.npy')))
num_batches = len(batch_files)

# Shuffle and split
random.seed(42)
indices = list(range(num_batches))
random.shuffle(indices)

train_split = int(0.8 * num_batches)
val_split = int(0.9 * num_batches)

train_idx = indices[:train_split]
val_idx = indices[train_split:val_split]
test_idx = indices[val_split:]

# Save splits for reproducibility
np.save(os.path.join(BATCH_DIR, 'train_indices.npy'), train_idx)
np.save(os.path.join(BATCH_DIR, 'val_indices.npy'), val_idx)
np.save(os.path.join(BATCH_DIR, 'test_indices.npy'), test_idx)

print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")
