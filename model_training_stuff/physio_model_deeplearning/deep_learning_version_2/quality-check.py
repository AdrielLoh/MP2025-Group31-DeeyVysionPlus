import os
import argparse
import numpy as np

EXPECTED_KEYS = ['signals', 'masks', 'videos', 'win_idxs', 'label']
EXPECTED_SIGNAL_SHAPE = (5, 150)  # channels, window
EXPECTED_MASK_SHAPE = (150,)
EXPECTED_MASK_VALUES = [0.0, 1.0]

def check_npz_file(path, fix_labels=False, dry_run=True):
    errors = []
    warnings = []
    fix_label_needed = False

    try:
        with np.load(path, allow_pickle=True) as d:
            # 1. Key presence
            for key in EXPECTED_KEYS:
                if key not in d:
                    errors.append(f"Missing key '{key}'")
            
            # 2. Shape & dtype checks
            if 'signals' in d:
                signals = d['signals']
                if signals.ndim != 3:
                    errors.append(f"'signals' is not 3D, got {signals.shape}")
                else:
                    n = signals.shape[0]
                    if signals.shape[1:] != EXPECTED_SIGNAL_SHAPE:
                        errors.append(f"'signals' has wrong shape: {signals.shape[1:]}, expected {EXPECTED_SIGNAL_SHAPE}")
                    if not np.issubdtype(signals.dtype, np.floating):
                        errors.append(f"'signals' has non-float dtype: {signals.dtype}")
                    # Value range & NaN/Inf
                    if np.isnan(signals).any():
                        errors.append(f"'signals' contains NaNs")
                    if np.isinf(signals).any():
                        errors.append(f"'signals' contains Infs")
                    sig_min, sig_max = np.min(signals), np.max(signals)
                    if sig_min < -10 or sig_max > 10:
                        warnings.append(f"'signals' values out of expected range [-10, 10]: min {sig_min}, max {sig_max}")

            else:
                n = None

            if 'masks' in d:
                masks = d['masks']
                if masks.ndim != 2:
                    errors.append(f"'masks' is not 2D, got {masks.shape}")
                elif n is not None and masks.shape[0] != n:
                    errors.append(f"'masks' first dim {masks.shape[0]} != signals batch size {n}")
                elif masks.shape[1] != EXPECTED_MASK_SHAPE[0]:
                    errors.append(f"'masks' has wrong shape: {masks.shape[1]}, expected {EXPECTED_MASK_SHAPE[0]}")
                if not np.issubdtype(masks.dtype, np.floating):
                    errors.append(f"'masks' has non-float dtype: {masks.dtype}")
                # NaN/Inf and mask values
                if np.isnan(masks).any():
                    errors.append(f"'masks' contains NaNs")
                if np.isinf(masks).any():
                    errors.append(f"'masks' contains Infs")
                mask_vals = np.unique(masks)
                if not all(v in EXPECTED_MASK_VALUES for v in np.round(mask_vals, 2)):
                    warnings.append(f"'masks' contains values other than 0 or 1: unique values {mask_vals}")

            if 'videos' in d:
                videos = d['videos']
                if videos.ndim != 1:
                    errors.append(f"'videos' is not 1D: shape {videos.shape}")
                elif n is not None and videos.shape[0] != n:
                    errors.append(f"'videos' len {videos.shape[0]} != signals batch size {n}")
                if videos.dtype.type is np.str_ or videos.dtype.type is np.object_:
                    # This is okay for video filenames
                    pass
                else:
                    errors.append(f"'videos' dtype unexpected: {videos.dtype}")

            if 'win_idxs' in d:
                win_idxs = d['win_idxs']
                if win_idxs.ndim != 1:
                    errors.append(f"'win_idxs' is not 1D: shape {win_idxs.shape}")
                elif n is not None and win_idxs.shape[0] != n:
                    errors.append(f"'win_idxs' len {win_idxs.shape[0]} != signals batch size {n}")

            # 3. Label checks
            if 'label' in d:
                label = d['label']
                if isinstance(label, np.ndarray): label = label.item()
                if isinstance(label, str):
                    warnings.append(f"label is string: {label!r}")
                    fix_label_needed = True
                    label_map = {'real': 0, 'fake': 1}
                    label_fixed = label_map.get(label.lower())
                    if label_fixed is None:
                        errors.append(f"Unknown string label: {label!r}")
                    else:
                        warnings.append(f"Can fix string label '{label}' -> {label_fixed} (with --fix-labels)")
                elif label not in (0, 1):
                    errors.append(f"Unexpected label value: {label!r}")
            else:
                errors.append("Missing 'label'")

            # 4. (Optional) Check for extreme win_idxs
            if 'win_idxs' in d:
                win_max = np.max(win_idxs)
                if win_max > 9999:
                    warnings.append(f"'win_idxs' max unusually high: {win_max}")

            # 5. (Optional) Print a small summary if everything looks good
            if not errors:
                print(f"  signals: {signals.shape}, min={np.min(signals):.3f}, max={np.max(signals):.3f}, dtype={signals.dtype}")
                print(f"  masks:   {masks.shape}, unique={np.unique(masks)}, dtype={masks.dtype}")
                print(f"  label:   {label!r}")
                print(f"  videos:  {videos.shape}, dtype={videos.dtype}")
                print(f"  win_idxs:{win_idxs.shape}")

            # 6. Optionally fix string label
            if fix_label_needed and fix_labels and not dry_run:
                print(f"  Fixing label in: {path}")
                data_to_save = dict(d.items())
                data_to_save['label'] = label_fixed
                np.savez_compressed(path, **data_to_save)
                warnings.append(f"Label fixed and file overwritten!")

    except Exception as e:
        errors.append(f"Exception loading file: {e}")

    return errors, warnings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('npz_dir', help="Directory with .npz batch files")
    parser.add_argument('--fix-labels', action='store_true', help="Fix string labels in-place")
    parser.add_argument('--dry-run', action='store_true', default=False, help="Don't overwrite, just show what would be fixed")
    args = parser.parse_args()

    npz_files = [os.path.join(args.npz_dir, f) for f in os.listdir(args.npz_dir) if f.endswith('.npz')]
    print(f"Checking {len(npz_files)} files in {args.npz_dir}")
    n_bad, n_warn = 0, 0
    for fpath in sorted(npz_files):
        print(f"\n{os.path.basename(fpath)}")
        errors, warnings = check_npz_file(fpath, fix_labels=args.fix_labels, dry_run=args.dry_run)
        if errors:
            print("  ERRORS:")
            for e in errors:
                print("   -", e)
            n_bad += 1
        if warnings:
            print("  WARNINGS:")
            for w in warnings:
                print("   -", w)
            n_warn += 1
    print(f"\nDone. {n_bad} files with ERRORS, {n_warn} files with WARNINGS.")

if __name__ == '__main__':
    main()
