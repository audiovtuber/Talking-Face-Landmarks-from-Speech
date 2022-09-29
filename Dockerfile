# FROM nvcr.io/nvidia/tensorflow:22.08-tf1-py3
# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
FROM nvcr.io/nvidia/pytorch:22.08-py3
# USER root
# Setting noninteractive because some the installation of some apt packages prompt for user input and this skips it
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y cmake build-essential gdb libopencv-dev python3-opencv ffmpeg bzip2 wget && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt

# pip must be >=21.3 so that `pip install -e .` will use the pyproject.toml configuration; fails otherwise!
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

RUN wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
    && bzip2 -d shape_predictor_68_face_landmarks.dat.bz2

# The remaining commands assume you run the container with the project mounted in `/workspace/Talking-Face-Landmarks-from-Speech`

# Weights & Biases causes everything to explode if you don't configure git to add repo as a safe directory, so we do that here for convenience
RUN git config --global --add safe.directory /workspace/Talking-Face-Landmarks-from-Speech

# Running ldconfig to fix this error when doing `import torch`:
#   "OSError: /opt/hpcx/ompi/lib/libmpi.so.40: undefined symbol: opal_hwloc201_hwloc_get_type_depth"
CMD cd Talking-Face-Landmarks-from-Speech && \
    pip install -e . && \
    ldconfig && \
    bash
