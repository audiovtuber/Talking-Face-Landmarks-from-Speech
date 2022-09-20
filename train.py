# Written by S. Emre Eskimez, in 2017 - University of Rochester
# Usage: python train.py -i path-to-hdf5-train-file/ -u number-of-hidden-units -d number-of-delay-frames -c number-of-context-frames -o output-folder-to-save-model-file
import argparse
import random
import os, shutil, glob

import tensorflow as tf
import numpy as np
from keras import backend as K
from keras.layers import Input, LSTM
from keras.models import Model
from tqdm import tqdm
from keras.optimizers import  Adam
from keras.callbacks import TensorBoard
import pandas as pd
import numpy as np

#-----------------------------------------#
#           Reproducible results          #
#-----------------------------------------#
sess = tf.Session()
K.set_session(sess)
os.environ['PYTHONHASHSEED'] = '128'
np.random.seed(128)
random.seed(128)
tf.set_random_seed(128)
#-----------------------------------------#

parser = argparse.ArgumentParser(description=__doc__)
# parser.add_argument("-i", "--in-file", type=str, help="Input file containing train data")
parser.add_argument("-u", "--hid-unit", type=int, help="hidden units")
parser.add_argument("-d", "--delay", type=int, help="Delay in terms of number of frames")
parser.add_argument("-c", "--ctx", type=int, help="context window size")
parser.add_argument("-o", "--out-fold", type=str, help="output folder")
args = parser.parse_args()

output_path = args.out_fold+'_'+str(args.hid_unit)+'/'

if not os.path.exists(output_path):
    os.makedirs(output_path)
else:
    shutil.rmtree(output_path)
    os.mkdir(output_path)

ctxWin = args.ctx
num_features_X = 128 * (ctxWin+1)# input feature size
num_features_Y = 136 # output feature size --> (68, 2)
num_frames = 75 # time-steps
batchsize = 128
h_dim = args.hid_unit
lr = 1e-3


drpRate = 0.2 # Dropout rate 
recDrpRate = 0.2 # Recurrent Dropout rate 

frameDelay = args.delay # Time delay

numEpochs = 200

# TODO: refactor hardcoded path
lmark_paths = sorted(glob.glob('grid_dataset/features/*-frames.npy')) # the "sorted" call is important; it ensures the filepaths are in the same order
mel_paths = sorted(glob.glob('grid_dataset/features/*-melfeatures.npy'))
data = {'melfeatures': mel_paths, 'frames': lmark_paths}
df = pd.DataFrame(data)
# 'flmark' contains the normalized face landmarks and shape must be (numberOfSamples, time-steps, 136)
# 'MelFeatures' contains the features, namely the delta and double delta mel-spectrogram. Shape = (numberOfSamples, time-steps, 128)

numIt = int(len(df)//batchsize) + 1
metrics = ['MSE', 'MAE']

def addContext(melSpc, ctxWin):
    ctx = melSpc[:,:]
    filler = melSpc[0, :]
    for i in range(ctxWin):
        melSpc = np.insert(melSpc, 0, filler, axis=0)[:ctx.shape[0], :]
        ctx = np.append(ctx, melSpc, axis=1)
    return ctx

def writeParams():
    # Write parameters of the network and training configuration
    with open(os.path.join(output_path, "model_info.txt"), "w") as text_file:
        text_file.write("{:30} {}\n".format('', output_path))
        text_file.write("------------------------------------------------------------------\n")
        text_file.write("{:30} {}\n".format('batchsize:', batchsize))
        text_file.write("{:30} {}\n".format('num_frames:', num_frames))
        text_file.write("{:30} {}\n".format('num_features_X:', num_features_X))
        text_file.write("{:30} {}\n".format('num_features_Y:', num_features_Y))
        text_file.write("{:30} {}\n".format('drpRate:', drpRate))
        text_file.write("{:30} {}\n".format('recDrpRate:', recDrpRate))
        text_file.write("{:30} {}\n".format('learning-rate:', lr))
        text_file.write("{:30} {}\n".format('h_dim:', h_dim))
        # text_file.write("{:30} {}\n".format('train filename:', args.in_file))
        text_file.write("{:30} {}\n".format('loss:', metrics[0]))
        text_file.write("{:30} {}\n".format('metrics:', metrics[1:]))
        text_file.write("{:30} {}\n".format('num_it:', numIt))
        text_file.write("{:30} {}\n".format('frameDelay:', frameDelay))
        text_file.write("------------------------------------------------------------------\n")
        model.summary(print_fn=lambda x: text_file.write(x + '\n'))

def write_log(callback, names, logs, batch_no):
    for name, value in zip(names, logs):
        summary = tf.Summary()
        summary_value = summary.value.add()
        summary_value.simple_value = value
        summary_value.tag = name
        callback.writer.add_summary(summary, batch_no)
        callback.writer.flush()

def dataGenerator():
    X_batch = np.zeros((batchsize, num_frames, num_features_X))
    Y_batch = np.zeros((batchsize, num_frames, num_features_Y))

    idxList = list(range(len(df)))

    batch_cnt = 0    
    while True:
        random.shuffle(idxList)
        for i in idxList:
            cur_lmark = np.load(open(df.iloc[i]['frames'], 'rb')).reshape((75, -1))  # later operations expect shape (75, 136) instead of (75, 68, 2)
            cur_mel = np.load(open(df.iloc[i]['melfeatures'], 'rb'))

            if frameDelay > 0:
                filler = np.tile(cur_lmark[0:1, :], [frameDelay, 1])
                cur_lmark = np.insert(cur_lmark, 0, filler, axis=0)[:num_frames]
             
            X_batch[batch_cnt, :, :] = addContext(cur_mel, ctxWin)
            Y_batch[batch_cnt, :, :] = cur_lmark
            
            batch_cnt+=1

            if batch_cnt == batchsize:
                batch_cnt = 0
                yield X_batch, Y_batch

def build_model():
    net_in = Input(shape=(num_frames, num_features_X))
    h = LSTM(h_dim, 
            activation='sigmoid', 
            dropout=drpRate, 
            recurrent_dropout=recDrpRate,
            return_sequences=True)(net_in)
    h = LSTM(h_dim, 
            activation='sigmoid',  
            dropout=drpRate, 
            recurrent_dropout=recDrpRate,
            return_sequences=True)(h)
    h = LSTM(h_dim, 
            activation='sigmoid', 
            dropout=drpRate, 
            recurrent_dropout=recDrpRate,
            return_sequences=True)(h)
    h = LSTM(num_features_Y, 
            activation='sigmoid', 
            dropout=drpRate, 
            recurrent_dropout=recDrpRate,
            return_sequences=True)(h)
    model = Model(inputs=net_in, outputs=h)
    model.summary()

    opt = Adam(lr=lr)

    model.compile(opt, metrics[0], 
                metrics= metrics[1:])
    return model

gen = dataGenerator()
model = build_model()

writeParams()

callback = TensorBoard(output_path)
callback.set_model(model)

k = 0
for epoch in tqdm(range(numEpochs)):
    for i in tqdm(range(numIt)):
        X_test, Y_test = next(gen)

        logs = model.train_on_batch(X_test, Y_test)
        if np.isnan(logs[0]):
            print ('NAN LOSS!')
            exit()

        write_log(callback, metrics, logs, k)
        k+=1

    model.save(output_path+'talkingFaceModel.h5')