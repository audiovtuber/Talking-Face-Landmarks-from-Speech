"""
Intended to be a replacement for featureExtractor.py with support for parallelized processing
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

def extract_features(path:str):
    output_dir = 'build_dataset_output' # TODO: parameterize (use star_map?)
    video_id = path.replace('/', '-')
    if os.path.exists(f"{output_dir}/{video_id}-frames.npy") and \
        os.path.exists(f"{output_dir}/{video_id}-melfeatures.npy"):
        print(f"Features already exist for video {path}. Skipping!")
        return path

    # TODO: most of this stuff should be one-time setup for each process; as such, is lots of overhead
    ms = np.load('mean_shape.npy') # Mean face shape, you can use any kind of face instead of mean face.
    fnorm = utils.faceNormalizer()
    ms = fnorm.alignEyePoints(np.reshape(ms, (1, 68, 2)))[0,:,:]
    # TODO: dlib singletons? probably not good for multiprocessing
    detector = dlib.get_frontal_face_detector()
    # TODO: hardcoded path
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    wsize = 0.04
    hsize = 0.04
    # These two vectors are for filling the empty cells with zeros for delta and double delta features
    zeroVecD = np.zeros((1, 64))
    zeroVecDD = np.zeros((2, 64))

    vid = imageio.get_reader(path,  'ffmpeg')
    point_seq = []
    for frm_cnt in range(0, vid.count_frames()):
        points = np.zeros((68, 2), dtype=np.float32)

        try:
            img = vid.get_data(frm_cnt)
        except:
            print('FRAME EXCEPTION!!')
            continue

        dets = detector(img, 1)
        if len(dets) != 1:
            print('FACE DETECTION FAILED!!')
            continue

        for k, d in enumerate(dets):
            shape = predictor(img, d)

            for i in range(68):
                points[i, 0] = shape.part(i).x
                points[i, 1] = shape.part(i).y

        # TODO: refactor the append/deepcopy? if all videos are 75 frames, then this could be optimized
        point_seq.append(deepcopy(points))
    cmd = 'ffmpeg -y -i ' + path + ' -vn -acodec pcm_s16le -ac 1 -ar 44100 ' + f"{video_id}.wav"
    subprocess.call(cmd, shell=True) 

    y, sr = librosa.load(f"{video_id}.wav", sr=44100)
    os.remove(f"{video_id}.wav")

    frames = np.array(point_seq)
    fnorm = utils.faceNormalizer()
    aligned_frames = fnorm.alignEyePoints(frames)
    transferredFrames = fnorm.transferExpression(aligned_frames, ms)
    frames = fnorm.unitNorm(transferredFrames)

    if frames.shape[0] != 75:
        return path

    melFrames = np.transpose(utils.melSpectra(y, sr, wsize, hsize))
    melDelta = np.insert(np.diff(melFrames, n=1, axis=0), 0, zeroVecD, axis=0)
    melDDelta = np.insert(np.diff(melFrames, n=2, axis=0), 0, zeroVecDD, axis=0)
    melFeatures = np.concatenate((melDelta, melDDelta), axis=1)

    if melFeatures.shape[0] != 75:
        return path
        
    # TODO: change hardcoded output dir
    with open(f"{output_dir}/{video_id}-melfeatures.npy", 'wb') as f:
        np.save(f, melFeatures)
    # TODO: change hardcoded output dir
    with open(f"{output_dir}/{video_id}-frames.npy", 'wb') as f:
        np.save(f, frames)
    return path

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
    os.makedirs("build_dataset_output", exist_ok=True)
    video_paths = glob.glob(str(Path(args.input_dir) / '**/*.mpg'), recursive=True)
    print(f"Found {len(video_paths)} videos")

    if num_processes > 1:
        pool = Pool(processes=num_processes)
        results = []
        max_files = 100
        for result in tqdm(pool.imap_unordered(extract_features, video_paths[:max_files]), total=len(video_paths[:max_files])):
            results.append(result)
    else:
        print("Only using one process. This takes forever and is only useful for debugging")
        for path in tqdm(video_paths):
            extract_features(path)
    

