import math
import os
from threading import Thread
from typing import Union, Iterable

import cv2
import numpy


def save_img(filename: Union[str, os.PathLike], thumbnail: numpy.ndarray):
    """Writes the image to a file."""
    cv2.imwrite(filename, thumbnail)


def run(filename: Union[str, os.PathLike]):
    """Extract frames from the video and creates thumbnails for one of each"""
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            "%s does not exist" % filename
        )

    print("Generate and save thumbs")
    thumb_dir = os.path.join("source", "thumbnails")
    if not os.path.isdir(thumb_dir):
        os.makedirs(thumb_dir)
    for index, frame in enumerate(video_to_frames(filename=filename)):  # Extract frames from video
        for thumb in image_to_thumbs(img=frame):  # Generate and save thumbs
            fname = os.path.join(thumb_dir, "thumbnail_" + str(index + 1) + ".jpg")
            print(f'Storing {fname}')
            # TODO: Implement concurrency instead of repeated threads
            Thread(target=save_img, kwargs={'filename': fname, 'thumbnail': thumb}).start()


def video_to_frames(filename: Union[str, os.PathLike]) -> Iterable[numpy.ndarray]:
    """Extract frames from video"""
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    video_duration = frame_count / fps
    frames_required = int(video_duration / 10)
    frame_split = []
    print("Video duration: ~%s %s" % (int(video_duration), "seconds"))
    print("Frames required: %s" % frames_required)

    i = 0
    frames_required = frames_required - 1
    while i <= 1:
        i += 1 / frames_required  # Places at which the thumbnail has to be captured
        frame_split.append(math.ceil(i * 100) / 100)
    frame_split = [i for i in frame_split if i < 1]  # Values go beyond 1 but we don't need that

    if cap.isOpened() and frame_count > 0:
        # Get frame ID of the thumbnail locations
        frame_ids = [round(frame_count * split_at - 0.25) for split_at in frame_split]
        frame_ids.insert(0, 0)
        frame_ids.append(frame_count - 1)

        count = 0
        success, image = cap.read()
        while success:
            if count in frame_ids:
                yield image
            success, image = cap.read()
            count += 1


def image_to_thumbs(img: numpy.ndarray) -> Iterable[numpy.ndarray]:
    """Create thumbs from image"""
    height, width, channels = img.shape
    size = 160
    if width >= size:
        r = (size + 0.0) / width
        max_size = (size, int(height * r))
        yield cv2.resize(img, max_size, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    run(filename="source/video.mp4")
