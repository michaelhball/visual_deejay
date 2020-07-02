import cv2
import math
import matplotlib.pyplot as plt
import os
import pytesseract
import subprocess
import sys

from PIL import Image


def angle_between_two_lines(v1, v2):
    """

    :param v1:
    :param v2:
    :return:
    """

    x1, y1 = v1
    x2, y2 = v2
    inner_product = x1 * x2 + y1 * y2
    len1 = math.hypot(x1, y1)
    len2 = math.hypot(x2, y2)
    return math.acos(inner_product / (len1 * len2))


def convert_time_rem_to_seconds(time_rem):
    """

    :param time_rem:
    :return:
    """

    try:
        time_rem = time_rem if not time_rem.startswith("-") else time_rem[1:]
        minutes, seconds = time_rem.split(":")
        return 60 * int(minutes) + float(seconds)
    except:
        # print("Exception converting 'time_gone' string to seconds => returning None")
        return None


def convert_time_gone_to_seconds(time_gone):
    """

    :param time_gone:
    :return:
    """

    try:
        minutes, seconds = time_gone.split(":")
        return 60 * int(minutes) + float(seconds)
    except:
        # print("Exception converting 'time_gone' string to seconds => returning None")
        return None


def display_video_frame(video_capture, n):
    """

    :param video_capture:
    :param n:
    :return:
    """

    video_capture.set(1, n)
    ret, frame = video_capture.read()
    if isinstance(ret, bool) and not ret:
        return False
    plt.imshow(frame)
    plt.show()
    return True


def extract_text_from_image(img):
    """

    :param img:
    :return:
    """

    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    text = pytesseract.image_to_string(Image.fromarray(grayscale))
    return text


def get_time_conversion(video_features, audio_features):
    """ Finds the needed function to convert from video feature timestamps --> timestamps of the actual audio
        recording.

    :param video_features: (list(dict)) features for each time-step in the screen recording
    :param audio_features: (list(dict)) features for each time-step in the screen recording
    :return: A function that converts time-stamps from video space --> audio space.
    """

    # TODO: audio-file time-start is just whenever the waveform first has sound (should be silent before PLAY pressed)
    # TODO: could find first moment in video using the first time something changes i.e. the frame where time
    #  remaining changes for the first track

    pass


def get_video_properties(filename):
    """

    :param filename:
    :return:
    """

    duration = get_video_duration(filename)
    fps = get_video_frame_rate(filename)
    if fps == -1:
        return False

    return {
        "duration": duration,
        "fps": fps,
        "num_frames": math.floor(duration * fps)
    }


def get_video_duration(filename):
    """
        https://stackoverflow.com/questions/3844430/how-to-get-the-duration-of-a-video-in-python

    :param filename:
    :return:
    """

    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                             "default=noprint_wrappers=1:nokey=1", filename], stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT)
    return float(result.stdout)


def get_video_frame_rate(filename):
    """
        https://askubuntu.com/questions/110264/how-to-find-frames-per-second-of-any-video-file

    :param filename:
    :return:
    """

    if not os.path.exists(filename):
        sys.stderr.write("ERROR: filename %r was not found!" % (filename,))
        return -1
    out = subprocess.check_output(
        ["ffprobe", filename, "-v", "0", "-select_streams", "v", "-print_format", "flat", "-show_entries",
         "stream=r_frame_rate"])
    rate = out.decode("utf-8").split('=')[1].strip()[1:-1].split('/')
    if len(rate) == 1:
        return float(rate[0])
    if len(rate) == 2:
        return float(rate[0]) / float(rate[1])
    return -1


def plot_one(img, cmap='viridis'):
    """

    :param img:
    :param cmap:
    :return:
    """

    plt.imshow(img, cmap=cmap)
    plt.show()


def plot_two(img1, img2, cmap='viridis'):
    """

    :param img1:
    :param img2:
    :param cmap:
    :return:
    """

    fig = plt.figure()
    _ = fig.add_subplot(1,2,1).imshow(img1, cmap=cmap)
    _ = fig.add_subplot(1,2,2).imshow(img2, cmap=cmap)
    plt.show()
