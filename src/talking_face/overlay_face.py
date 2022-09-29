import os
from argparse import ArgumentParser
from pathlib import Path
import subprocess

import dlib
import imageio.v3 as iio
import imageio.v2 as iiov2
import numpy as np
import imageio
from tqdm import tqdm

from talking_face.generate import extract_audio_features_from_video, TalkingFacePredictor


def overlay_landmarks(image, shape):
    overlayed_image = np.copy(image)
    try:
        for part in shape.parts():
            overlayed_image[
                part.y - 2 : part.y + 2, part.x - 2 : part.x + 2
            ] = np.array([255, 255, 255])
    except AttributeError:
        for x, y in shape:
            # likely a numpy array has been passed in instead of a dlib full_object_detection object
            overlayed_image[y - 2 : y + 2, x - 2 : x + 2] = np.array([255, 255, 255])
        # overlayed_image[shape] = np.array([0, 0, 255])

    return overlayed_image


def process_video(
    video_path: str,
    output_dir: str,
    model: TalkingFacePredictor = None,
    save_images: bool = False,
):
    print(f"Processing {video_path}")
    os.makedirs(output_dir, exist_ok=True)

    features, sound = extract_audio_features_from_video(video_path)
    detector = dlib.get_frontal_face_detector()
    # TODO: refactor hardcoded path
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    try:
        video = imageio.get_reader(video_path, "ffmpeg")
        video_shape = video.get_meta_data()["size"]
    except OSError as e:
        print(f"Failed to open video: {video_path} \n{e}")
        return

    if model is not None:
        model = TalkingFacePredictor(model)
        model.reset_state()
        predictions = model(features)

        # normalize shape based on image width and height and offset X to be on the right-half of video
        predictions[:, :, 0] = predictions[:, :, 0] * video_shape[0] + video_shape[0]
        predictions[:, :, 1] = predictions[:, :, 1] * video_shape[1]
        predictions = predictions.astype("int")

        # TODO: why do I need to pad frames into the audio? refactor this hack
        predictions = np.roll(
            np.resize(
                predictions,
                (video.count_frames(), predictions.shape[1], predictions.shape[2]),
            ),
            3,
            axis=0,
        )

        assert predictions.shape[0] == video.count_frames()

    temp_video_path = Path(output_dir) / "silent_my_video.mp4"
    target_video_path = Path(output_dir) / "my_video.mp4"

    w = iiov2.get_writer(temp_video_path, format="FFMPEG", fps=25)
    for frame_idx in tqdm(range(video.count_frames())):
        try:
            image = video.get_data(frame_idx)  # (H, W, C), uint8

            # dlib type `rectangles`, indexable. Each rectangle is ordered left(), top(), right(), bottom(), so x0, y0, x1, y1
            dets = detector(image, 0)
            if len(dets) != 1:
                return f"FOUND {len(dets)} FACES! (only want 1)"
            # shape.parts() has 68 points, each which has (x, y) attributes, all uint8 giving absolute position in image
            shape = predictor(image, dets[0])

            overlayed_image = overlay_landmarks(image, shape)
            if model is not None:
                overlayed_image = np.append(
                    overlayed_image, np.zeros_like(image), axis=1
                )
                overlayed_image = overlay_landmarks(
                    overlayed_image, predictions[frame_idx]
                )

            w.append_data(overlayed_image)

            if save_images:
                with open(
                    Path(output_dir)
                    / f"{video_path.replace('/', '-')}_frame-{frame_idx:03d}.png",
                    "wb",
                ) as f:
                    iio.imwrite(f, overlayed_image, plugin="pillow", extension=".png")
        except Exception as e:
            print(e)
            return "FRAME EXCEPTION"
    w.close()

    # Add the audio stream back in
    cmd = f"ffmpeg -i {temp_video_path} -i {video_path} -map 0:v -map 1:a -c:v copy -shortest {target_video_path}"
    subprocess.call(cmd, shell=True)
    os.remove(temp_video_path)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument(
        "--input-video",
        help="Video on which to overlay face landmarks",
    )
    parser.add_argument(
        "--model",
        required=False,
        help="Path to a trained audio-vtuber model. If provided, the resulting video will be augmented with its predictions",
    )
    parser.add_argument(
        "--save-images",
        required=False,
        action="store_true",
        help="Saves individual video frames as images",
    )
    parser.add_argument("--output-dir", help="Where to save the resulting video")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_video(
        video_path=args.input_video,
        output_dir=args.output_dir,
        model=args.model,
        save_images=args.save_images,
    )
