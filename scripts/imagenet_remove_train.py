from pathlib import Path
import argparse
import os

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input path.")
    return parser.parse_args()

def main():
    args = get_args()
    in_path = Path(args.in_path)
    assert in_path.is_dir(), f"{in_path} is not a directory."

    for dir in in_path.iterdir():
        if not dir.is_dir():
            continue
        for i, img in enumerate(dir.iterdir()):
            os.remove(img)
            if i == 9:
                break


if __name__ == "__main__":
    main()
