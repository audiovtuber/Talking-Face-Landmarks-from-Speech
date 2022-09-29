"""
Most of these tests were written not to test functionality, but speed. To take advantage of them,
use `pytest --durations=5` from the project root
"""
import glob
import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path

import dlib
import imageio


test_video_path = "grid_dataset/extracted/s1/video/mpg_6000/bbaf2n.mpg"
target_test_features_frames = "grid_dataset/features/grid_dataset-extracted-s1-video-mpg_6000-bbaf2n.mpg-frames.npy"
target_test_features_melfeatures = "grid_dataset/features/grid_dataset-extracted-s1-video-mpg_6000-bbaf2n.mpg-melfeatures.npy"


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input-dir",
        default="grid_dataset/extracted",
        help="Path to grid dataset with all tarfiles extracted",
    )
    parser.add_argument(
        "--processes",
        type=int,
        required=False,
        help="Number of processes to use. Defaults to available threads minus 1",
    )
    return parser.parse_args()


def prepare_test_extraction():
    if os.path.exists(target_test_features_frames):
        os.remove(target_test_features_frames)
    if os.path.exists(target_test_features_melfeatures):
        os.remove(target_test_features_melfeatures)
    return


def test_dataset_exists():
    assert os.path.exists(test_video_path)


def test_load_dlib():
    # Runtimes (in seconds): 0.75
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    return


def test_ffmpeg():
    # Runtimes (in seconds): 0.09, 0.07
    cmd = f"ffmpeg -hide_banner -loglevel error -y -i {test_video_path} -vn -acodec pcm_s16le -ac 1 -ar 44100 lol.wav"
    subprocess.call(cmd, shell=True)
    os.remove("lol.wav")


def test_imageio_get_data():
    # Runtimes (in seconds): 0.24
    video = imageio.get_reader(test_video_path, "ffmpeg")
    for frame_index in range(video.count_frames()):
        video.get_data(frame_index)


def test_detect():
    # Runtimes (in seconds): 14.38, 15.16
    # Runtime, detection only: 14.46, 15.07
    # NOTE: Runtime with no upsampling: 4.37, 4.13
    # Based on runtimes of other tests, the majority of time is spent in detection and prediction
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    video = imageio.get_reader(test_video_path, "ffmpeg")
    for frame_index in range(video.count_frames()):
        image = video.get_data(frame_index)
        # TODO: IMPORTANT: second argument to detector() is an upsampling parameter. Default of `1` takes 4x longer than `0`!
        detections = detector(image, 0)[0]
        shape = predictor(image, detections)


"""
def test_extract_features():
    # calling ffmpeg directly in bash takes 0.053s
    # Runtimes (in seconds) - 16.13, 15.12, 15.10, 15.23, 15.00
    # After refactor of shape loop: 15.08, 15.38
    # After refactor of point_seq: 15.05, 15.12
    prepare_test_extraction()
    extract_features(test_video_path)
    assert os.path.exists(target_test_features_frames)
    assert os.path.exists(target_test_features_melfeatures)
"""

if __name__ == "__main__":
    args = parse_args()
    video_paths = glob.glob(str(Path(args.input_dir) / "**/*.mpg"), recursive=True)
    print(f"Found {len(video_paths)} videos")
