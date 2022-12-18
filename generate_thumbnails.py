import math
import os
from typing import Union

import cv2
import numpy


def run(filename: Union[str, os.PathLike]):
    """Extract frames from the video and creates thumbnails for one of each"""
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            "%s does not exist" % filename
        )

    # Extract frames from video
    print("Extracting frames from video")
    frames = video_to_frames(filename=filename)

    # Generate and save thumbs
    print("Generate and save thumbs")
    for i in range(len(frames)):
        thumb = image_to_thumbs(img=frames[i])
        os.makedirs('frames/%d' % i)
        for k, v in thumb.items():
            cv2.imwrite('frames/%d/%s.png' % (i, k), v)


def video_to_frames(filename: Union[str, os.PathLike]):
    """Extract frames from video"""
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    video_duration = frame_count / fps
    frames_required = int(video_duration / 10)
    frame_split = []
    print("Video duration: %s" % video_duration)
    print("Frames required: %s" % frames_required)

    i = 0
    frames_required = frames_required - 1
    while i <= 1:
        i += 1 / frames_required  # Places at which the thumbnail has to be captured
        frame_split.append(math.ceil(i * 100) / 100)
    frame_split = [i for i in frame_split if i < 1]  # Values go beyond 1 but we don't need that

    frames = []
    if cap.isOpened() and frame_count > 0:
        # Get frame ID of the thumbnail locations
        frame_ids = [round(frame_count * split_at) for split_at in frame_split]
        frame_ids.insert(0, 0)
        frame_ids.append(frame_count - 1)

        count = 0
        success, image = cap.read()
        while success:
            if count in frame_ids:
                frames.append(image)
            success, image = cap.read()
            count += 1
    return frames


def image_to_thumbs(img: numpy.ndarray):
    """Create thumbs from image"""
    # TODO: Figure out the right size required and store it in thumbnails within source dir
    height, width, channels = img.shape
    thumbs = {"original": img}
    sizes = [640, 320, 160]
    for size in sizes:
        if width >= size:
            r = (size + 0.0) / width
            max_size = (size, int(height * r))
            thumbs[str(size)] = cv2.resize(img, max_size, interpolation=cv2.INTER_AREA)
    return thumbs
