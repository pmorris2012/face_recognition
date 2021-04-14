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


def save_keypoints_image(image_path, input_folder, output_folder):
    image = face_recognition.load_image_file(image_path)
    landmarks = face_recognition.face_landmarks(image, return_dict=False)
    
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


def process_images(image_paths, input_folder, output_folder, processes):
    if processes == -1:
        processes = None

    function_parameters = zip(
        image_paths,
        itertools.repeat(input_folder),
        itertools.repeat(output_folder)
    )

    if processes == 1:
        for parameters in function_parameters:
            save_keypoints_image(*parameters)
    else:
        context = multiprocessing
        if "forkserver" in multiprocessing.get_all_start_methods():
            context = multiprocessing.get_context("forkserver")
        pool = context.Pool(processes=processes)
        pool.starmap(save_keypoints_image, function_parameters)

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
@click.option('--cpus', default=-1, help='number of CPU cores to use in parallel (can speed up processing lots of images). -1 means "use all in system"')
def main(input_folder, output_folder, cpus):
    image_paths, video_paths = find_files(input_folder)
    print(F"found {len(image_paths)} images and {len(video_paths)} videos.")
    
    process_images(image_paths, input_folder, output_folder, cpus)



if __name__ == "__main__":
    main()
