# FROM nvcr.io/nvidia/tensorflow:22.08-tf1-py3
# FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel
FROM nvcr.io/nvidia/pytorch:22.08-py3
# USER root
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y cmake build-essential gdb libopencv-dev python3-opencv ffmpeg bzip2 wget && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt
RUN wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
    && bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
CMD ["bash"]
