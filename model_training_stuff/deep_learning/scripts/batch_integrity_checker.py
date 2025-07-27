import numpy as np
import os

BATCH_DIR = r'C:\deepvysion\preprocessed_batches'

def inspect_batch(batch_idx):
    frames = np.load(os.path.join(BATCH_DIR, f'batch_{batch_idx}_frames.npy'), allow_pickle=True)
    labels = np.load(os.path.join(BATCH_DIR, f'batch_{batch_idx}_labels.npy'), allow_pickle=True)
    print(f"Batch {batch_idx}: {len(frames)} videos, Labels: {labels.shape}")
    if len(frames) > 0:
        print(f"First video shape: {frames[0].shape if hasattr(frames[0], 'shape') else 'None'}")

# Inspect a few batches
for idx in [0, 100, 674]:
    inspect_batch(idx)
