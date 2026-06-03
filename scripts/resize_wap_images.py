import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from PIL import Image
from tqdm import tqdm


DEFAULT_SRC_DIR = "./data/raw/World-Aware-Planning/images"
DEFAULT_DST_DIR = "./data/raw/World-Aware-Planning/images_224"
DEFAULT_SIZE = 224
DEFAULT_WORKERS = 16


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_dir", default=DEFAULT_SRC_DIR)
    parser.add_argument("--dst_dir", default=DEFAULT_DST_DIR)
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE)
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def resize_one(args):
    path, src_dir, dst_dir, size, overwrite = args
    rel_path = path.relative_to(src_dir)
    out_path = dst_dir / rel_path
    if out_path.exists() and not overwrite:
        return "skipped"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(path) as image:
        image = image.convert("RGB")
        if image.size != (size, size):
            image = image.resize((size, size), Image.BICUBIC)
        image.save(out_path)
    return "written"


def main():
    args = parse_args()
    src_dir = Path(args.src_dir)
    dst_dir = Path(args.dst_dir)
    if not src_dir.exists():
        raise FileNotFoundError(f"Source image directory not found: {src_dir}")

    paths = sorted(src_dir.rglob("*.png"))
    if not paths:
        raise RuntimeError(f"No PNG files found in {src_dir}")

    jobs = [
        (path, src_dir, dst_dir, args.size, args.overwrite)
        for path in paths
    ]
    counts = {"written": 0, "skipped": 0}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for result in tqdm(executor.map(resize_one, jobs), total=len(jobs)):
            counts[result] = counts.get(result, 0) + 1

    print(f"Source: {src_dir}")
    print(f"Output: {dst_dir}")
    print(f"Size: {args.size}x{args.size}")
    print(f"Written: {counts.get('written', 0)}")
    print(f"Skipped: {counts.get('skipped', 0)}")


if __name__ == "__main__":
    main()
