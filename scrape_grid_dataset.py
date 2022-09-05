from asyncio import subprocess
import os
from argparse import ArgumentParser
from pathlib import Path
import subprocess

import requests
from tqdm import tqdm

def fetch_file(individual:int, part:int, output_dir=None):
    assert individual in set(range(1,35)) - {21}
    assert part in {1, 2}
    output_dir = 'grid_dataset' if output_dir is None else output_dir
    file_path = Path(output_dir) / f"s{individual}.mpg_6000.part{part}.tar"
    print(f"Fetching part {part} for individual {individual}; writing to {file_path}")
    if os.path.exists(str(file_path)):
        print(f"File already exists, skipping")
        return
    os.makedirs(output_dir, exist_ok=True)
    url = f"https://spandh.dcs.shef.ac.uk/gridcorpus/s{individual}/video/s{individual}.mpg_6000.part{part}.tar"
    response = requests.get(url, stream=True)
    # progress bar code borrowed from Stack Overflow :) 
    
    total_size_in_bytes= int(response.headers.get('content-length', 0))
    block_size = 1024 #1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)
    with open(file_path, 'wb') as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


def download(output_dir:str):
    individuals = sorted(set(range(1,35)) - {21})  # all individuals except 21, who doesn't have video
    for idx in individuals:
        fetch_file(individual=idx, part=1, output_dir=output_dir)
        fetch_file(individual=idx, part=2, output_dir=output_dir)

def decompress(input_dir:str, output_dir:str):
    assert os.path.exists(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    cmd = f"ls {input_dir}" + '/*.tar | xargs -i tar xf {}' + f"-C {output_dir}/"
    print(f"Extracting videos using this command: {cmd}")
    subprocess.call(cmd, shell=True)
    
def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--output-dir', default='grid_dataset', help='The folder where the tar files will be stored')
    return parser.parse_args()

if __name__ == '__main__':
    # download()
    # Run this from project root
    args = parse_args()
    download(args.output_dir)
    decompress(input_dir=args.output_dir, output_dir=str(Path(args.output_dir) / 'extracted'))