import math
import os
from threading import Thread
from typing import Union, Iterable

import cv2
import numpy

thumb_dir = os.path.join("source", "thumbnails")
if not os.path.isdir(thumb_dir):
    os.makedirs(thumb_dir)


def save_img(iter_val: int, frame: numpy.ndarray):
    """Writes the image to a file."""
    filename = os.path.join(thumb_dir, "thumbnail_" + str(iter_val + 1) + ".jpg")
    print(f'Storing {filename}')
    thumbnail = image_to_thumbs(img=frame)
    cv2.imwrite(filename, thumbnail)


def run(filename: Union[str, os.PathLike], frame_interval: int = 10):
    """Extract frames from the video and creates thumbnails for one of each"""
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            "%s does not exist" % filename
        )

    print("Extracting frames from the video")
    for index, frame in enumerate(video_to_frames(filename=filename, interval=frame_interval)):
        Thread(target=save_img, kwargs={'iter_val': index, 'frame': frame}).start()


def video_to_frames(filename: Union[str, os.PathLike], interval: int) -> Iterable[numpy.ndarray]:
    """Extract frames from video"""
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    video_duration = frame_count / fps
    frames_required = int(video_duration / interval)
    print("Video duration: ~%s %s" % (int(video_duration), "seconds"))
    print("Frames required: %s" % frames_required)

    frame_split = []
    i = 0
    frames_required = frames_required - 1  # Since the last frame is inserted later manually
    while i <= 1:
        i += 1 / frames_required  # Places at which the thumbnail has to be captured
        frame_split.append(math.ceil(i * 100) / 100)
    frame_split = [i for i in frame_split if i < 1]  # Values go beyond 1 but we don't need that

    if cap.isOpened() and frame_count > 0:
        # Get frame ID of the thumbnail locations
        frame_ids = [round(frame_count * split_at) for split_at in frame_split]
        frame_ids.insert(0, 0)  # Insert frame at 0th second at 0th position in the list
        frame_ids.append(frame_count - 1)  # Insert the frame at last second at the last position in the list

        count = 0
        success, image = cap.read()
        while success:
            if count in frame_ids:
                yield image
            success, image = cap.read()
            count += 1


def image_to_thumbs(img: numpy.ndarray, size: int = 160) -> numpy.ndarray:
    """Create thumbnails from image by resizing the image."""
    height, width, channels = img.shape
    if width >= size:
        r = (size + 0.0) / width
        max_size = (size, int(height * r))
        return cv2.resize(img, max_size, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    run(filename="source/video.mp4")
