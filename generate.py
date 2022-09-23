"""
Intended as a replacement for `generate.py`
"""
import os
import argparse
import subprocess

import torch
import librosa
import numpy as np
import matplotlib as mpl
mpl.use('Agg')
from tqdm import tqdm

import utils
from train import TalkingFaceLSTM


class TalkingFacePredictor():
    """
    A trained audio vtuber model for inference. It behaves similarly to a regular torch module, but helps manage the LSTM's 
    statefulness between predictions. Intended usage is as follows for each sequence of predictions:
    1. Call predictor.reset_state() (resets hidden state from previous sequences)
    2. Call predictor() or predictor.predict() for each frame in a sequence
    """
    def __init__(self, checkpoint:str):
        self.model = TalkingFaceLSTM.load_from_checkpoint(checkpoint)
        self.model.eval()
        self.hidden_state = None
        self.cell = None

    def reset_state(self,):
        self.hidden_state = None
        self.cell = None

    def predict(self, inputs:np.ndarray):
        with torch.no_grad():
            # cur_features = np.zeros((1, num_frames, features.shape[2]))
            # if i+1 > 75:
            #     lower = i+1-75
            # cur_features[:,-i-1:,:] = features[:,lower:i+1,:]
            if self.hidden_state is None:
                # pred, (self.hidden_state, self.cell) = model(torch.tensor(cur_features).float())
                pred, (self.hidden_state, self.cell) = self.model(torch.tensor(inputs).float())
            else:
                pred, (self.hidden_state, self.cell) = self.model(torch.tensor(inputs).float(), self.hidden_state, self.cell)
            
            # generated = np.append(generated, np.reshape(pred[0,-1,:], (1, num_features_Y)), axis=0)
            return pred.reshape((-1, 68, 2)).numpy()
    def __call__(self, inputs):
        return self.predict(inputs)
            


def addContext(melSpc, ctxWin):
    ctx = melSpc[:,:]
    filler = melSpc[0, :]
    for i in range(ctxWin):
        melSpc = np.insert(melSpc, 0, filler, axis=0)[:ctx.shape[0], :]
        ctx = np.append(ctx, melSpc, axis=1)
    return ctx

def extract_audio_features(path:str, hsize:float=0.04, wsize:float=0.04, sample_rate:int=44100):
    # Used for padding zeros to first and second temporal differences
    zeroVecD = np.zeros((1, 64), dtype='f16')
    zeroVecDD = np.zeros((2, 64), dtype='f16')

    # Load speech and extract features
    sound, sr = librosa.load(path, sr=sample_rate)
    melFrames = np.transpose(utils.melSpectra(sound, sr, wsize, hsize))
    melDelta = np.insert(np.diff(melFrames, n=1, axis=0), 0, zeroVecD, axis=0)
    melDDelta = np.insert(np.diff(melFrames, n=2, axis=0), 0, zeroVecDD, axis=0)

    features = np.concatenate((melDelta, melDDelta), axis=1)
    # features = addContext(features, ctxWin)  # TODO: revisit this!
    features = np.reshape(features, (1, features.shape[0], features.shape[1]))
    return features, sound


def extract_audio_features_from_video(path:str):
    cmd = f"ffmpeg -i {path} temp_output.wav"  # TODO: not multithread safe. refactor to be in-memory or use tempfiles
    subprocess.call(cmd, shell=True)
    features, sound = extract_audio_features('temp_output.wav')
    os.remove('temp_output.wav')
    return features, sound

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-i", "--in-file", type=str, help="input speech file")
    parser.add_argument("-m", "--model", type=str, help="DNN model to use")
    parser.add_argument("-d", "--delay", default=0, type=int, help="Delay in terms of number of frames, where each frame is 40 ms")
    parser.add_argument("-c", "--ctx", default=0, type=int, help="context window size")
    parser.add_argument("-o", "--out-fold", type=str, help="output folder")
    return parser.parse_args()


def old(args):
    output_path = args.out_fold
    trainDelay = args.delay
    ctxWin = args.ctx
    test_file = args.in_file

    num_features_Y = 136
    num_frames = 75

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = TalkingFaceLSTM.load_from_checkpoint(args.model)
    model.eval()


    features, sound = extract_audio_features(path=test_file)
    upper_limit = features.shape[1]
    lower = 0
    generated = np.zeros((0, num_features_Y))

    # Generates face landmarks one-by-one
    # This part can be modified to predict the whole sequence at one, but may introduce discontinuities
    hidden, cell = None, None
    with torch.no_grad():
        for i in tqdm(range(upper_limit)):
            cur_features = np.zeros((1, num_frames, features.shape[2]))
            if i+1 > 75:
                lower = i+1-75
            cur_features[:,-i-1:,:] = features[:,lower:i+1,:]

            if hidden is None:
                pred, (hidden, cell) = model(torch.tensor(cur_features).float())
            else:
                pred, (hidden, cell) = model(torch.tensor(cur_features).float(), hidden, cell)
            generated = np.append(generated, np.reshape(pred[0,-1,:], (1, num_features_Y)), axis=0)

        """
        # TODO: revisit this; not sure it works as intended!
        # Shift the array to remove the delay
        generated = generated[trainDelay:, :]
        tmp = generated[-1:, :]
        for _ in range(trainDelay):
            generated = np.append(generated, tmp, axis=0)
        """

        if len(generated.shape) < 3:
            print(generated.shape)
            generated = np.reshape(generated, (generated.shape[0], int(generated.shape[1]/2), 2))

        fnorm = utils.faceNormalizer()
        generated = fnorm.alignEyePointsV2(600*generated) / 600.0 
        utils.write_video_wpts_wsound(generated, sound, 44100, output_path, 'PD_pts', [0, 1], [0, 1])


if __name__ == '__main__':
    args = parse_args()
    # old(args)
    model = TalkingFacePredictor(checkpoint=args.model)
    features, sound = extract_audio_features(path=args.in_file)

    model.reset_state()  # not necessary here, but good practice
    out = model(features)
    fnorm = utils.faceNormalizer()
    out = fnorm.alignEyePointsV2(600*out) / 600.0 
    utils.write_video_wpts_wsound(out, sound, 44100, args.out_fold, 'PD_pts', [0, 1], [0, 1])



    


