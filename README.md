This repo is a fork of [Talking Face Landmarks From Speech](https://github.com/eeskimez/Talking-Face-Landmarks-from-Speech)

# Docker Setup
1. Clone this repo
2. `cd` to the project root
3. Build the container (if you want) `docker build -t slashfury/fsdl-talking-face .`
  * Alternatively, pull the container from dockerhub via `docker pull slashfury/fsdl-talking-face`
4. Start the container with `docker run --gpus all -it --name tf-dev -v $PWD:/workspace/Talking-Face-Landmarks-from-Speech --ipc=host slashfury/fsdl-talking-face` (this will mount the project inside the container and open a bash terminal)
5. Inside the container, run the following commands and then log into Weights & Biases
``` bash
mv shape_predictor_68_face_landmarks.dat Talking-Face-Landmarks-from-Speech
cd Talking-Face-Landmarks-from-Speech
git config --global --add safe.directory /workspace/Talking-Face-Landmarks-from-Speech
wandb login
```

## A100 Instructions
If running on a machine with an A100 GPU, you need to use a specific version of pytorch. Run this command:
``` bash
pip install torch==1.12.1+cu116  -f https://download.pytorch.org/whl/torch_stable.html
```

# Preparing Data

There are two options for preparing the dataset: build it yourself, or simply download a tarfile and extract it.

## The Easy way

Download [this tarfile](https://storage.googleapis.com/audio-vtuber/grid_features.tar) and extract it to `project_root/grid_dataset/features`. Done!
## The Hard Way

Before you can train a model, you'll need to download and process the [GRID dataset](https://spandh.dcs.shef.ac.uk//gridcorpus/). This can be done via the [`scrape_grid_dataset.py`](scrape_grid_dataset.py) and [`build_dataset.py`](build_dataset.py) scripts. 

> **Note**: This can take a **long time** and should be done using a machine with many CPU cores and a fast internet connection.

Inside the project root in the container, run `python scrape_grid_dataset.py`, which will do these things:

1. Downloads all of the videos as tarballs to `{project_root}/grid_dataset`
2. Extracts the tarballs to `{project_root}/grid_dataset/extracted`

By default, the script has the flag `--use-gcs`, which downloads from a faster mirror

## Building the Training Dataset

After you've [downloaded the GRID dataset](#downloading-the-grid-dataset), run `python build_dataset.py`, which will do these things:

1. Use dlib and ffmpeg to extract face landmarks from every frame of every video
2. Use librosa to extract mel-spectrogram features
3. Store the results as files in `grid_dataset/features`

You're now ready to train a model!

-  [ ] TODO: Host the processed `*.npy` files somewhere. 

# Training

Inside the project root in the container, run [`train.py`](train.py) after you've [prepared the dataset](#preparing-data). The only required argument is `--save-path`, but feel free to experiment with other flags!

``` bash
# Example Training Run
python train.py --layers 4 --save-path my_experiment/ --epochs 100 --batch-size 256
```

# Predicting

Inside the project root in the container, run [`generate.py`](generate.py) after you've [trained a model](#training)

``` bash
python generate.py -i test_samples/test1.flac -m my_experiment/trained_model.ckpt -d 0 -c 0 -o output_dir
```


# Local Setup
> Note: Setup was tested in an Ubuntu 20.04 image in WSL.

1. Install [conda](https://docs.conda.io/en/latest/miniconda.html) like normal using linux script
2. Install dlib prerequisites:
``` bash
sudo apt update
sudo apt install cmake build-essential gdb
```
3. Clone this repo and setup your conda environment:
``` bash
conda env update -f environment.yml
```
>**Note**: This may not yet be stable
4. Download and decompress dlib's face landmark model; this is used to generate ground truth from video
``` bash
# in project root
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
```

Finally, activate the conda environment using `conda activate talking-face`. At this point, you should be able to run the [code example](#code-example) below using your own FLAC audio file

---

> **Note**: The original README.md contents are below
# Generating Talking Face Landmarks

The code for [the paper](https://link.springer.com/chapter/10.1007/978-3-319-93764-9_35) "Generating Talking Face Landmarks from Speech."

You can find the project page [here](http://www2.ece.rochester.edu/projects/air/projects/talkingface.html).

An improved version of this project can be found [here](http://www2.ece.rochester.edu/projects/air/projects/3Dtalkingface.html).

## Installation

> **(Note from FSDL team)**: This is outdated and incorrect. Follow install instructions above instead
#### The project depends on the following Python packages:

* Keras --- 2.2.4
* Tensorflow --- 1.9.0
* Librosa --- 0.6.0
* opencv-python --- 3.3.0.10
* dlib --- 19.7.0
* tqdm 
* subprocess

#### It also depends on the following packages:
* ffmpeg --- 3.4.1
* OpenCV --- 3.3.0

The code has been tested on Ubuntu 16.04 and OS X Sierra and High Sierra. 

## Code Example

The generation code has the following arguments:

* -i --- Input speech file
    * See [this](http://librosa.github.io/librosa/generated/librosa.core.load.html#librosa.core.load) link for supported audio formats.
* -m --- Input talking face landmarks model 
* -d --- Delay in terms of frames, where one frame is 40 ms
* -c --- Number of context frames
* -o --- Output path

You can run the following code to test the system:

```
python generate.py -i test_samples/test1.flac -m models/D40_C3.h5 -d 1 -c 3 -o results/D40_C3_test1
```
## Feature Extraction

You can run featureExtractor.py to extract features from videos directly. The arguments are as follows:

* -vp --- Input folder containing video files (if your video file types are different from .mpg or .mp4, please modify the script accordingly)
* -sp --- Path to shape_predictor_68_face_landmarks.dat. You can download this file [here](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat).
* -o --- Output file name

Usage: 

```
python featureExtractor.py -vp path-to-video-files/ -sp path-to-shape-predictor-68-face-landmarks-dat -o output-file-name.hdf5
```

## Training

The training code has the following arguments:

* -i --- Input hdf5 file containing training data
* -u --- Number of hidden units
* -d --- Delay in terms of frames, where one frame is 40 ms
* -c --- Number of context frames
* -o --- Output folder path to save the model

Usage:

```
python train.py -i path-to-hdf5-train-file/ -u number-of-hidden-units -d number-of-delay-frames -c number-of-context-frames -o output-folder-to-save-model-file
```

## Citation

```
@inproceedings{eskimez2018generating,
  title={Generating talking face landmarks from speech},
  author={Eskimez, Sefik Emre and Maddox, Ross K and Xu, Chenliang and Duan, Zhiyao},
  booktitle={International Conference on Latent Variable Analysis and Signal Separation},
  pages={372--381},
  year={2018},
  organization={Springer}
}
```