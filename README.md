This repo is a fork of [Talking Face Landmarks From Speech](https://github.com/eeskimez/Talking-Face-Landmarks-from-Speech)

# Docker Setup
1. Clone this repo
2. `cd` to the project root
3. Build the container (if you want) `docker build -t slashfury/fsdl-talking-face .`
> *Alternatively, pull the container from dockerhub via `docker pull slashfury/fsdl-talking-face`*
4. Start the container with `docker run --gpus all -it --name tf-dev -v $PWD:/workspace/Talking-Face-Landmarks-from-Speech --ipc=host slashfury/fsdl-talking-face` (this will mount the project inside the container and open a bash terminal)
5. Inside the container, run the following command
``` bash
mv shape_predictor_68_face_landmarks.dat Talking-Face-Landmarks-from-Speech
```
> *You only need to do step 5 once ever; by moving the file, it's now in your project root and will be mounted every time you start the container*

## A100 Instructions
If running on a machine with an A100 GPU, you need to use a specific version of pytorch. Run this command:
``` bash
pip install torch==1.12.1+cu116  -f https://download.pytorch.org/whl/torch_stable.html
```

# Preparing Data

There are two options for preparing the dataset: build it yourself, or simply download a tarfile and extract it.

## The Easy way

Download [this tarfile](https://storage.googleapis.com/audio-vtuber/grid_features.tar) and extract it to `project_root/grid_dataset/features`. Done! You're now ready for [Training](#training)
## The Hard Way

Before you can train a model, you'll need to download and process the [GRID dataset](https://spandh.dcs.shef.ac.uk//gridcorpus/). This can be done via the [`scrape_grid_dataset.py`](src/scrape_grid_dataset.py) and [`build_dataset.py`](src/build_dataset.py) scripts.

> **Note**: This can take a **long time** and should be done using a machine with many CPU cores and a fast internet connection.

Inside the project root in the container, run `python src/scrape_grid_dataset.py`, which will do these things:

1. Downloads all of the videos as tarballs to `{project_root}/grid_dataset`
2. Extracts the tarballs to `{project_root}/grid_dataset/extracted`

By default, the script has the flag `--use-gcs`, which downloads from a faster mirror

### Building the Training Dataset

After you've [downloaded the GRID dataset](#downloading-the-grid-dataset), run `python src/build_dataset.py`, which will do these things:

1. Use dlib and ffmpeg to extract face landmarks from every frame of every video
2. Use librosa to extract mel-spectrogram features
3. Store the results as files in `grid_dataset/features`

You're now ready to train a model!

# Training

Inside the project root in the container, run [`train.py`](src/talking_face/train.py) after you've [prepared the dataset](#preparing-data). The only required argument is `--save-path`, but feel free to experiment with other flags!

> The first time you run training after starting a new docker container, Weights & Biases will prompt you to login. Follow the instructions in terminal

``` bash
# Example Training Run
python -m talking_face.train --layers 4 --save-path experiments/my_experiment --epochs 100 --batch-size 256
```

# Predicting

Inside the project root in the container, there are two scripts for prediction: [`overlay_face.py`](src/talking_face/overlay_face.py) and [`generate.py`](src/talkiing_face/generate.py) after you've [trained a model](#training). The difference is that `generate.py` will generate a video using only a matplotlib plot, whereas `overlay_face.py` will output the original video plus ground truth face landmarks (left) side-by-side with the audio-only predictions from the model (right). I recommend using [`overlay_face.py`](src/talking_face/overlay_face.py)

## `overlay_face` example

``` bash
python -m talking_face.overlay_face --input-video test_samples/my_video.mp4 --model experiments/my_experiment/trained_model.ckpt --output-dir results/my_video
```

## `generate` example
> *I recommend using [overlay_face](#overlayface-example) instead*

``` bash
python -m talking_face.generate -i test_samples/test1.flac -m experiments/my_experiment/trained_model.ckpt -d 0 -c 0 -o output_dir
```


# Local Setup
A local setup is useful especially for code completion, linting, and autoformatting
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
5. If using vscode, install the extensions `autoDocstring` and `Remote Development` extensions. Configure `autoDocstring` to generate numpy-style docstrings

Finally, activate the conda environment using `conda activate talking-face`. At this point, you should be able to run the [code example](#code-example) below using your own FLAC audio file

---

> **Note**: The original README.md contents are below
# References

[The paper](https://link.springer.com/chapter/10.1007/978-3-319-93764-9_35) "Generating Talking Face Landmarks from Speech."

You can find the original project page [here](http://www2.ece.rochester.edu/projects/air/projects/talkingface.html).

An improved version of this project can be found [here](http://www2.ece.rochester.edu/projects/air/projects/3Dtalkingface.html).
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
