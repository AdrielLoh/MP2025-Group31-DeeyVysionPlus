import os
import subprocess
import sys
from multiprocessing import Pool, cpu_count

def compress_mp4(args):
    input_file, crf = args
    dirpath, filename = os.path.split(input_file)
    output_file = os.path.join(dirpath, f"compressed_{filename}")

    # FFmpeg command: CRF for quality/size, scale to 1080p (but don't upscale smaller)
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-c:v", "libx264", "-preset", "medium",
        "-crf", str(crf),
        "-vf", "scale='if(gt(iw/ih,16/9),-2,1920)':'if(gt(iw/ih,16/9),1080,-2)'",  # 1080p max, keep aspect ratio, no upscaling
        "-c:a", "aac", "-b:a", "160k",
        output_file
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        # Only replace if the new file is smaller and exists
        if os.path.exists(output_file) and os.path.getsize(output_file) < os.path.getsize(input_file):
            os.remove(input_file)
            os.rename(output_file, input_file)
            print(f"[OK] Compressed and replaced: {input_file}")
        else:
            print(f"[SKIP] Compression ineffective for {input_file} (kept original)")
            if os.path.exists(output_file):
                os.remove(output_file)
    except Exception as e:
        print(f"[ERROR] Compression failed for {input_file}: {e}")
        if os.path.exists(output_file):
            os.remove(output_file)

def find_mp4s(root_folder):
    mp4_files = []
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".mp4"):
                mp4_files.append(os.path.join(dirpath, filename))
    return mp4_files

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python compress_mp4s_crf.py <root_folder> [crf]")
        print("  - <root_folder>: Path to the folder with videos")
        print("  - [crf]: Optional, 18 (higher quality, larger) to 28 (smaller, lower quality). Default 23.")
        sys.exit(1)

    root = sys.argv[1]
    crf = int(sys.argv[2]) if len(sys.argv) >= 3 else 23

    files = find_mp4s(root)
    if not files:
        print("No mp4 files found.")
        sys.exit(0)

    print(f"Found {len(files)} mp4 files to process with CRF={crf}.")
    n_workers = min(cpu_count(), 6)
    print(f"Using {n_workers} parallel workers.")

    with Pool(processes=n_workers) as pool:
        pool.map(compress_mp4, [(f, crf) for f in files])

    print("Done.")
