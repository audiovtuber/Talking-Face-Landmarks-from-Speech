"""
The purpose of this script is to download all videos of the GRID dataset and decompress them. This should only be done
if the files are not available via git-lfs
"""
import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

import requests
from tqdm import tqdm


def fetch_file(
    individual: int,
    part: int,
    output_dir: Union[Path, str] = "grid_dataset",
    use_gcs: bool = True,
):
    """Downloads one part of an individual tarball from the GRID dataset. Note that
    each individual has two parts and 33 total individuals (individual 21 is missing),
    totalling 66 files

    Parameters
    ----------
    individual : int
        ID of the individual whose video file is being retrieved. Should be a number in between 1 and 34 (inclusive) except for 21
    part : int
        Specifies whether to download the file "part1" or "part2"
    output_dir : Union[Path, str]
        Specifies where to save the file. Defaults to "grid_dataset"
    use_gcs : bool, optional
        If specified, downloads files from Google Cloud Storage; otherwise, downloads from the original university's website. Defaults to True
    """
    assert individual in set(range(1, 35)) - {21}
    assert part in {1, 2}
    output_dir = "grid_dataset" if output_dir is None else output_dir
    file_path = Path(output_dir) / f"s{individual}.mpg_6000.part{part}.tar"
    print(f"Fetching part {part} for individual {individual}; writing to {file_path}")
    if os.path.exists(str(file_path)):
        print("File already exists, skipping")
        return
    os.makedirs(output_dir, exist_ok=True)
    # Note from Slash_Fury: I made a mirror of the dataset in GCP. This is publically accessible
    url = f"https://storage.googleapis.com/audio-vtuber/s{individual}.mpg_6000.part{part}.tar"
    if not use_gcs:
        # fallback URL; slower, but still fastish!
        url = f"https://spandh.dcs.shef.ac.uk/gridcorpus/s{individual}/video/s{individual}.mpg_6000.part{part}.tar"
    response = requests.get(url, stream=True)

    # progress bar code borrowed from Stack Overflow :)
    total_size_in_bytes = int(response.headers.get("content-length", 0))
    block_size = 1024  # 1 Kibibyte
    progress_bar = tqdm(total=total_size_in_bytes, unit="iB", unit_scale=True)
    with open(file_path, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)
    progress_bar.close()
    if total_size_in_bytes != 0 and progress_bar.n != total_size_in_bytes:
        print("ERROR, something went wrong")


def download(output_dir: str, use_gcs: bool = True):
    """
    Helper function to iterate over all 66 files to be downloaded. See `fetch_file` for details
    """
    # all individuals except 21, who doesn't have video
    individuals = sorted(set(range(1, 35)) - {21})
    for idx in individuals:
        fetch_file(individual=idx, part=1, output_dir=output_dir, use_gcs=use_gcs)
        fetch_file(individual=idx, part=2, output_dir=output_dir, use_gcs=use_gcs)


def decompress(input_dir: str, output_dir: str):
    """
    Decompress all tarballs at once. Could simply do this in a terminal as well
    """
    assert os.path.exists(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    cmd = f"ls {input_dir}" + "/*.tar | xargs -i tar xf {} " + f"-C {output_dir}/"
    print(f"Extracting videos using this command: {cmd}")
    subprocess.call(cmd, shell=True)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--use-gcs", default=True, type=bool, help="Download from Google Storage"
    )
    parser.add_argument(
        "--output-dir",
        default="grid_dataset",
        help="The folder where the tar files will be stored",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Run this from project root
    args = parse_args()
    download(args.output_dir)
    decompress(
        input_dir=args.output_dir, output_dir=str(Path(args.output_dir) / "extracted")
    )
