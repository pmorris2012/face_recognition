import cv2
import click
from pathlib import Path
import os
import re
import face_recognition.api as face_recognition
import itertools
import sys
import PIL.Image
import numpy as np

video_exts = [
    '.mkv', 
    '.flv', 
    '.avi', 
    '.mp4',
    '.m4v',
    '.mpg',
    '.mpeg'
]

image_exts = [
    '.png', 
    '.jpg', 
    '.jpeg'
]


def scale_frame(frame):
    height, width = frame.shape[:2]
    w, h = width, height
    ratio = -1

    # Resize in case of to bigger dimension
    # In first instance manage the HIGH-Dimension photos
    if width > 3600 or height > 3600:
        if width > height:
            ratio = width / 800
        else:
            ratio = height / 800

    elif 1200 <= width <= 1600 or 1200 <= height <= 1600:
        ratio = 1 / 2
    elif 1600 <= width <= 3600 or 1600 <= height <= 3600:
        ratio = 1 / 3

    if 0 < ratio < 1:
        # Scale image in case of width > 1600
        w = width * ratio
        h = height * ratio
    elif ratio > 1:
        # Scale image in case of width > 3600
        w = width / ratio
        h = height / ratio
    if w != width:
        # Check if scaling was applied
        w, h = round(w), round(h)
        frame = cv2.resize(frame, dsize=(w,h), interpolation=cv2.INTER_CUBIC)

    return frame


def move_path(path, from_path, to_path):
    relative = path.relative_to(from_path)
    return Path(to_path, relative)


def save_keypoints_video(video_path, input_folder, output_folder, tolerance, batch_size):
    video = cv2.VideoCapture(str(video_path))

    out_path = move_path(video_path, input_folder, output_folder)
    out_path = out_path.with_suffix('')
    known_faces = []
    frame_count = 0
    frames = []
    while video.isOpened():
        ret, frame = video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = scale_frame(frame)
            frames.append(frame)

        if (len(frames) == batch_size) or not ret:
            batch_of_face_locations = face_recognition.batch_face_locations(frames, number_of_times_to_upsample=upsample, batch_size=batch_size)

            # Now let's list all the faces we found in all batch_size frames
            for i, face_locations in enumerate(batch_of_face_locations):
                frame_count += 1
                encodings, landmarks = face_recognition.face_encodings_and_landmarks(frames[i], known_face_locations=locations)

                num_faces = landmarks.shape[0]
                if num_faces == 0:
                    continue

                for face_idx, (encoding, landmark_array) in enumerate(zip(encodings, landmarks)):
                    distances = face_recognition.face_distance(known_faces, encoding)
                    if len(distances) == 0 or distances.min() > tolerance:
                        known_idx = len(known_faces)
                        known_faces.append(encoding)
                    else:
                        known_idx = int(np.argmin(distances))
                        known_faces[known_idx] = np.mean([known_faces[known_idx], encoding], axis=0)

                    file_path = Path(out_path, F"{known_idx}", F"{frame_count}.npy")
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    np.save(file_path, landmark_array)

            # Clear the frames array to start the next batch
            frames = []

    print(F"finished video {out_path}")


def process_files(paths, *args, **kwargs):
    for path in paths:
        save_keypoints_video(path, *args, **kwargs)


def find_files(base_path):
    image_paths, video_paths = [], []
    for path in Path(base_path).glob('**/*'):
        if path.is_file():
            if path.suffix in image_exts:
                image_paths.append(path)
            elif path.suffix in video_exts:
                video_paths.append(path)

    return image_paths, video_paths


@click.command()
@click.argument('input_folder')
@click.argument('output_folder')
@click.option('--tolerance', default=0.6, help='Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.')
@click.option('--batch_size', default=64, help='For face locations. Set lower if out of memory error occurs.')
def main(input_folder, output_folder, tolerance, batch_size):
    image_paths, video_paths = find_files(input_folder)
    print(F"found {len(image_paths)} images and {len(video_paths)} videos.")
    
    process_files(video_paths, input_folder, output_folder, tolerance, batch_size)


if __name__ == "__main__":
    main()

