"""
A replacement for featureExtractor.py with support for parallelized processing
"""
import glob
from pathlib import Path
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count
from copy import deepcopy
import subprocess
import os

import librosa
import dlib
import numpy as np
import imageio
from tqdm import tqdm

import utils

def extract_features(path:str, overwrite:bool=False):
    """
    Processes a single video and saves two numpy files as output:
    1. `{video_id}-frames.npy` - shape (75, 68, 2), which represents the 68 face landmarks for
    all 75 frames of a video
    2. `{video_id}-melfeatures.npy` - shape (75, 128), the spectrogram of the audio

    TODO: A very dirty function and I don't like the return values either
    """
    output_dir = 'grid_dataset/features' # TODO: parameterize (use star_map?)
    video_id = path.replace('/', '-')
    if os.path.exists(f"{output_dir}/{video_id}-frames.npy") \
        and os.path.exists(f"{output_dir}/{video_id}-melfeatures.npy") \
        and not overwrite:
        # print(f"Features already exist for video {path}. Skipping!")
        return path

    # TODO: most of this stuff should be one-time setup for each process; as such, is lots of overhead
    ms = np.load('mean_shape.npy') # Mean face shape, you can use any kind of face instead of mean face.
    fnorm = utils.faceNormalizer()
    ms = fnorm.alignEyePoints(np.reshape(ms, (1, 68, 2)))[0,:,:]
    detector = dlib.get_frontal_face_detector()
    # TODO: refactor hardcoded path
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    wsize = 0.04
    hsize = 0.04
    # These two vectors are for filling the empty cells with zeros for delta and double delta features
    zeroVecD = np.zeros((1, 64))
    zeroVecDD = np.zeros((2, 64))

    try:
        vid = imageio.get_reader(path,  'ffmpeg')
    except OSError as e:
        print(f"Failed to open video: {path} \n{e}")
        return 'FAILED TO OPEN VIDEO'

    point_seq = np.zeros(shape=(vid.count_frames(), 68, 2))
    for frm_cnt in range(0, vid.count_frames()):
        try:
            img = vid.get_data(frm_cnt)
        except:
            return 'FRAME EXCEPTION'

        # NOTE: second argument controls upscaling. Originally was `1`, changed to `0` for 4x speedup
        dets = detector(img, 0)
        if len(dets) != 1:
            return f"FOUND {len(dets)} FACES! (only want 1)"

        shape = predictor(img, dets[0])
        point_seq[frm_cnt] = np.array([[part.x, part.y] for part in shape.parts()])
    cmd = 'ffmpeg -hide_banner -loglevel error -y -i ' + path + ' -vn -acodec pcm_s16le -ac 1 -ar 44100 ' + f"{video_id}.wav"
    subprocess.call(cmd, shell=True) 

    try:
        y, sr = librosa.load(f"{video_id}.wav", sr=44100)
        os.remove(f"{video_id}.wav")
    except FileNotFoundError as e:
        print(f"Failed to load (or remove) extracted wav; video likely corrupt\n{e}")
        return 'FAILED TO DELETE WAV'

    frames = np.array(point_seq)
    fnorm = utils.faceNormalizer()
    aligned_frames = fnorm.alignEyePoints(frames)
    transferredFrames = fnorm.transferExpression(aligned_frames, ms)
    frames = fnorm.unitNorm(transferredFrames)

    # TODO: refactor to allow arbitrary number of frames and melFeatures
    if frames.shape[0] != 75:
        return path

    melFrames = np.transpose(utils.melSpectra(y, sr, wsize, hsize))  # shape after transpose is roughly (audio_length_in_ms / 40, 64)
    melDelta = np.insert(np.diff(melFrames, n=1, axis=0), 0, zeroVecD, axis=0)  # inserts zeros into the first row of the diff of melFrames (the diff operation wraps and we don't want that)
    melDDelta = np.insert(np.diff(melFrames, n=2, axis=0), 0, zeroVecDD, axis=0)  # same idea as last line. n=2 is equal to np.diff(np.diff(x, n=1), n=1), so the first two rows should be zeroes
    melFeatures = np.concatenate((melDelta, melDDelta), axis=1)

    if melFeatures.shape[0] != 75:
        return path
        
    # TODO: change hardcoded output dir
    with open(f"{output_dir}/{video_id}-melfeatures.npy", 'wb') as f:
        np.save(f, melFeatures)
    # TODO: change hardcoded output dir
    with open(f"{output_dir}/{video_id}-frames.npy", 'wb') as f:
        np.save(f, frames)
    return 'success'


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--input-dir', default='grid_dataset/extracted', help='Path to grid dataset with all tarfiles extracted')
    # parser.add_argument('--output-dir', default='grid_features', help='The folder where the feature files will be stored')
    parser.add_argument('--processes', type=int, required=False, help='Number of processes to use. Defaults to available threads minus 1')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    num_processes = max(cpu_count() - 1, 1)
    if args.processes is not None:
        num_processes = args.processes

    print(f"Processing videos using {num_processes} processes")
    os.makedirs("grid_dataset/features", exist_ok=True)
    video_paths = glob.glob(str(Path(args.input_dir) / '**/*.mpg'), recursive=True)
    print(f"Found {len(video_paths)} videos")

    if num_processes > 1:
        pool = Pool(processes=num_processes)
        results = []
        for result in tqdm(pool.imap_unordered(extract_features, video_paths), total=len(video_paths)):
            results.append(result)
    else:
        print("Only using one process. This takes forever and is only useful for debugging")
        for path in tqdm(video_paths):
            extract_features(path)
    

"""
errors:
/workspace/Talking-Face-Landmarks-from-Speech/grid_dataset/extracted/s7/video/mpg_6000/lbad1s.mpg
"""