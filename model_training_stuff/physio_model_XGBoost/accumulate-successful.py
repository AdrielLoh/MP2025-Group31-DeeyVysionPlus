import os
import glob

def combine_success_logs(cache_dir, label, output_file=None):
    """
    Accumulate all lines from success_{label}_batch_*.txt in cache_dir into one deduped file.
    Does not overwrite previous entries in the accumulated log (it merges).
    Args:
        cache_dir (str): Directory where batch logs are located.
        label (str): The label, e.g. 'real' or 'fake'.
        output_file (str): Optional, defaults to success_{label}_all.txt in cache_dir.
    """
    pattern = os.path.join(cache_dir, f"success_{label}_batch_*.txt")
    output_file = output_file or os.path.join(cache_dir, f"success_{label}_all.txt")
    
    all_lines = set()
    
    # First, load lines from the existing accumulated log (if present)
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                clean = line.strip()
                if clean:
                    all_lines.add(clean)
        print(f"Loaded {len(all_lines)} entries from existing {output_file}")
    
    # Now, add lines from all batch files
    files = glob.glob(pattern)
    print(f"Found {len(files)} batch log files matching pattern: {pattern}")

    for fname in files:
        with open(fname, "r") as f:
            for line in f:
                clean = line.strip()
                if clean:
                    all_lines.add(clean)
    
    sorted_lines = sorted(all_lines)
    with open(output_file, "w") as f:
        for line in sorted_lines:
            f.write(line + "\n")
    print(f"Wrote {len(sorted_lines)} unique entries to {output_file}")

if __name__ == "__main__":
    # Example usage
    # Set these according to your cache directory and label
    cache_dir = "C:/model_training/physio_ml/fake"
    label = "fake"
    combine_success_logs(cache_dir, label)
