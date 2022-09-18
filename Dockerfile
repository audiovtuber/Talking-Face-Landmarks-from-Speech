FROM nvcr.io/nvidia/tensorflow:22.08-tf1-py3

# USER root
RUN apt update && apt install -y cmake build-essential gdb libopencv-dev python3-opencv ffmpeg
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
RUN wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 \
    && bzip2 -d shape_predictor_68_face_landmarks.dat.bz2
CMD ["bash"]  