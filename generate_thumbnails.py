import logging
import math
import os
from multiprocessing.pool import ThreadPool
from typing import Union, Iterable, NoReturn

import cv2
import numpy

FORMATTER = logging.Formatter(
    fmt='%(asctime)s - %(levelname)s - [%(module)s:%(lineno)d] - %(funcName)s - %(message)s',
    datefmt='%b-%d-%Y %H:%M:%S'
)
LOGGER = logging.getLogger(__name__)
HANDLER = logging.StreamHandler()
HANDLER.setFormatter(fmt=FORMATTER)
LOGGER.addHandler(hdlr=HANDLER)
LOGGER.setLevel(level=logging.DEBUG)

POOL = ThreadPool(processes=int(os.environ.get('PROCESSES', os.cpu_count() / 2)))  # Number of processes to spin up


def save_img(iter_val: int, frame: numpy.ndarray) -> NoReturn:
    """Writes the image to a file.

    Args:
        iter_val: Iterative number to form the filename.
        frame: Image frame that has to be stored for each file.
    """
    filename = os.path.join("source", "thumbnails", "thumbnail_" + str(iter_val + 1) + ".jpg")
    LOGGER.info("Generating thumbnail %s" % filename)
    thumbnail = image_to_thumbs(img=frame)
    cv2.imwrite(filename, thumbnail)


def run(filename: Union[str, os.PathLike], frame_interval: int = 10) -> NoReturn:
    """Extract frames from the video and create thumbnails for one of each.

    Args:
        filename: Name of the source file.
        frame_interval: Interval at which frames have to be captured for the video.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            "%s does not exist" % filename
        )

    if not os.path.isdir(os.path.join("source", "thumbnails")):
        os.makedirs(os.path.join("source", "thumbnails"))

    LOGGER.info("Extracting frames from the video")
    for index, frame in enumerate(video_to_frames(filename=filename, interval=frame_interval)):
        POOL.apply_async(func=save_img, kwds={'iter_val': index, 'frame': frame})
    POOL.close()
    POOL.join()


def video_to_frames(filename: Union[str, os.PathLike], interval: int) -> Iterable[numpy.ndarray]:
    """Extract frames from video.

    Args:
        filename: Source filename
        interval: Interval at which frames have to be captured for the video.

    Yields:
        numpy.ndarray:
        Frame of the image as a numpy array.
    """
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1

    video_duration = frame_count / fps
    frames_required = int(video_duration / interval)
    LOGGER.info("Video duration: ~%s %s" % (int(video_duration), "seconds"))
    LOGGER.info("Frames required: %s" % frames_required)

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
        success, image = cap.read()  # Read the initial frame
        while success:
            if count in frame_ids:
                yield image  # Yield the frame when the count matches the frame ID required for thumbnails
            success, image = cap.read()  # Keep reading frames until the end of the video
            count += 1


def image_to_thumbs(img: numpy.ndarray, size: int = 160) -> numpy.ndarray:
    """Create thumbnails from image by resizing the image.

    Args:
        img: Image frame received as numpy array.
        size: Size of the thumbnail.

    Returns:
        numpy.ndarray:
        Frame of the thumbnail as a numpy array.
    """
    height, width, channels = img.shape
    if width >= size:
        r = (size + 0.0) / width
        max_size = (size, int(height * r))
        return cv2.resize(img, max_size, interpolation=cv2.INTER_AREA)


if __name__ == '__main__':
    run(filename="source/video.mp4")
