# -*- coding: utf-8 -*-
from __future__ import print_function
import click
from pathlib import Path
import os
import re
import face_recognition.api as face_recognition
import multiprocessing
import itertools
import sys
import PIL.Image
import numpy as np
import cv2

#distances = face_recognition.face_distance(known_face_encodings, unknown_encoding)
#result = list(distances <= tolerance)

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


def move_path(path, from_path, to_path):
    relative = path.relative_to(from_path)
    return Path(to_path, relative)


def save_keypoints_image(image_path, input_folder, output_folder, model):
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image, model=model)
    landmarks = face_recognition.face_landmarks(image, face_locations=locations, return_dict=False)
    
    num_faces = landmarks.shape[0]
    if num_faces == 0:
        return

    out_path = move_path(image_path, input_folder, output_folder)
    out_path = out_path.with_suffix('')
    out_path.mkdir(parents=True, exist_ok=True)

    for face_idx in range(num_faces):
        file_path = Path(out_path, F"{face_idx}.npy")
        landmark_array = landmarks[face_idx]
        np.save(file_path, landmark_array)

    print(F"finished image {out_path}")


def save_keypoints_video(video_path, input_folder, output_folder, model, tolerance):
    video = cv2.VideoCapture(str(video_path))

    out_path = move_path(video_path, input_folder, output_folder)
    out_path = out_path.with_suffix('')
    known_faces = []
    frame_count = 0
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(frame, model=model)
        encodings, landmarks = face_recognition.face_encodings_and_landmarks(frame, known_face_locations=locations)

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

        frame_count += 1

    print(F"finished video {out_path}")


def process_files(paths, input_folder, output_folder, model, processes, process_fn, tolerance=0.6):
    if processes == -1:
        processes = None

    function_parameters = [
        paths,
        itertools.repeat(input_folder),
        itertools.repeat(output_folder),
        itertools.repeat(model)
    ]
    if process_fn == save_keypoints_video:
        function_parameters.append(itertools.repeat(tolerance))

    if processes == 1:
        for parameters in zip(*function_parameters):
            process_fn(*parameters)
    else:
        context = multiprocessing
        if "forkserver" in multiprocessing.get_all_start_methods():
            context = multiprocessing.get_context("forkserver")
        pool = context.Pool(processes=processes)
        pool.starmap(process_fn, zip(*function_parameters))


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
@click.option('--model', default='hog', help='face detector model, either cnn or hog. (cnn is accurate, hog is fast)')
@click.option('--cpus', default=-1, help='number of CPU cores to use in parallel (can speed up processing lots of images). -1 means "use all in system"')
@click.option('--tolerance', default=0.6, help='Tolerance for face comparisons. Default is 0.6. Lower this if you get multiple matches for the same person.')
def main(input_folder, output_folder, model, cpus, tolerance):
    image_paths, video_paths = find_files(input_folder)
    print(F"found {len(image_paths)} images and {len(video_paths)} videos.")
    
    process_files(image_paths, input_folder, output_folder, model, cpus, save_keypoints_image)
    process_files(video_paths, input_folder, output_folder, model, cpus, save_keypoints_video, tolerance=tolerance)



if __name__ == "__main__":
    main()
