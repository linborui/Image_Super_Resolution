import sys
import os
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i',"--in_path", type=str, required=True)
    parser.add_argument("-o", "--out_path", type=str, default=None)
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    parent_path = Path(args.in_path)

    if os.path.exists(args.out_path):
        os.remove(args.out_path)
    
    with open(args.out_path, 'a') as f:
        for dir in os.listdir(args.in_path):
            curr_path = parent_path / dir
            print(f'current path: {curr_path}')
            for i, file in enumerate(os.listdir(os.path.join(args.in_path, dir))):
                path = curr_path / file
                f.write(f'{path}\n')
                if i == 9:
                    break


if __name__ == "__main__":
    main()