import cv2
import math
import numpy as np
import pickle

from pathlib import Path
from tqdm import tqdm

from .utils import angle_between_two_lines, extract_text_from_image, get_video_properties, plot_one


def extract_knob_angle(knob_img):
    """

    :param knob_img:
    :return:
    """

    # convert to grayscale + thresholding to only show the white ticker marker
    knob_img = cv2.cvtColor(knob_img, cv2.COLOR_BGR2GRAY)
    ret, knob_img_thresh = cv2.threshold(knob_img, 127, 255, cv2.THRESH_BINARY)
    height, width = knob_img_thresh.shape

    # find all coordinates within the white ticket
    ticker = np.nonzero(knob_img_thresh)

    # find topmost y coord (and the rightmost x coord with said y coord)
    top_coord_y = np.min(ticker[0])
    for j in range(width - 1, 0, -1):
        if knob_img_thresh[top_coord_y, j] != 0:
            top_coord_x = j
            break
    top_point = top_coord_x, top_coord_y  # top == top-right

    # find bottommost y coord (& the leftmost x coord with said y coord)
    bottom_coord_y = np.max(ticker[0])
    for j in range(width):
        if knob_img_thresh[bottom_coord_y, j] != 0:
            bottom_coord_x = j
            break
    bottom_point = bottom_coord_x, bottom_coord_y  # bottom == bottom-left

    # find leftmost x coord (& the topmost y coord with said x coord)
    left_point = np.min(ticker[1]), ticker[0][np.argmin(ticker[1])]  # left is top left

    # find rightmost x coord (& the bottommost y coord with said x coord)
    right_coord_x = np.max(ticker[1])
    for i in range(height - 1, 0, -1):
        if knob_img_thresh[i, right_coord_x] != 0:
            right_coord_y = i
            break
    right_point = right_coord_x, right_coord_y

    # find out whether rectangle knob ticker is more horizontal or vertical, and compute lines between 4 corners
    if (np.linalg.norm(np.array(left_point) - np.array(bottom_point)) < np.linalg.norm(
            np.array(left_point) - np.array(top_point))):
        v1 = right_point[0] - left_point[0], right_point[1] - left_point[1]
        v2 = top_point[0] - bottom_point[0], top_point[1] - bottom_point[1]
    else:
        v1 = right_point[0] - left_point[0], right_point[1] - left_point[1]
        v2 = bottom_point[0] - top_point[0], bottom_point[1] - top_point[1]

    # compute mean (i.e. the straight line going up the lengthwise center of the rectangle ticker)
    v = np.mean([v1, v2], axis=0)

    # normalise mean vector according to the skew of pixels in the image (so the angles resemble common-sense)
    v[0] *= width / (width + height)
    v[1] *= height / (width + height)

    # return angle between solved vector & the upward-pointing vertical line
    return math.degrees(angle_between_two_lines((0, 1), v))


def extract_channel_features(channel_img):
    """

    :param channel_img:
    :return:
    """

    # extract fader value
    fader_img = channel_img[314:392, :, :]
    fader_img = cv2.cvtColor(fader_img, cv2.COLOR_BGR2GRAY)
    ret, fader_img_thresh = cv2.threshold(fader_img, 127, 255, cv2.THRESH_BINARY)
    fader_val = (fader_img_thresh.shape[0] - 1 - np.max(np.nonzero(fader_img_thresh)[0])) / (fader_img_thresh.shape[0] - 1)
    if fader_val <= 0:
        return {"fader": 0}

    # extract trim value
    trim_img = channel_img[8:50, :, :]
    trim_angle = extract_knob_angle(trim_img)

    # extract highs value
    highs_img = channel_img[74:116, :, :]
    highs_angle = extract_knob_angle(highs_img)

    # extract mids value
    mids_img = channel_img[140:182, :, :]
    mids_angle = extract_knob_angle(mids_img)

    # extract lows value
    lows_img = channel_img[206:248, :, :]
    lows_angle = extract_knob_angle(lows_img)

    return {"fader": fader_val, "trim": trim_angle, "high": highs_angle, "mid": mids_angle, "low": lows_angle}


def extract_features_from_frame(frame):
    """

    :param frame:
    :return:
    """

    # TODO: convert all image croppings to relative rather than absolute in case they don't work with resolution of vid
    #   ultimately might want to find a better way... for now I can rely on having everything just so on the screen??

    features = {"left": {}, "right": {}}

    # extract time gone/remaining for each track
    left_track_time_rem_img = frame[615:665, 1075:1210, :]
    features["left"]["time_rem"] = extract_text_from_image(left_track_time_rem_img)
    right_track_time_rem_img = frame[615:665, 2860:2995, :]
    features["right"]["time_rem"] = extract_text_from_image(right_track_time_rem_img)
    left_track_time_gone_img = frame[615:665, 1215:1330, :]
    features["left"]["time_gone"] = extract_text_from_image(left_track_time_gone_img)
    right_track_time_gone_img = frame[615:665, 3000:3120, :]
    features["right"]["time_gone"] = extract_text_from_image(right_track_time_gone_img)

    # extract title & artist of each  track
    left_track_title_img = frame[580:625, 110:1000, :]
    features["left"]["title"] = extract_text_from_image(left_track_title_img)
    left_track_artist_img = frame[625:660, 110:430, :]
    features["left"]["artist"] = extract_text_from_image(left_track_artist_img)
    right_track_title_img = frame[580:625, 1900:2780, :]
    features["right"]["title"] = extract_text_from_image(right_track_title_img)
    right_track_artist_img = frame[625:660, 1900:2220, :]
    features["right"]["artist"] = extract_text_from_image(right_track_artist_img)

    left_channel_eqs_img = frame[600:1010, 1595:1665, :]
    left_channel_features = extract_channel_features(left_channel_eqs_img)
    if isinstance(left_channel_features, bool) and not left_channel_features:
        return False
    features["left"]["eqs"] = left_channel_features

    right_channel_eqs_img = frame[600:1010, 1700:1770, :]
    right_channel_features = extract_channel_features(right_channel_eqs_img)
    if isinstance(right_channel_features, bool) and not right_channel_features:
        return False
    features["right"]["eqs"] = right_channel_features

    return features


def extract_features_from_video(video_file, params):
    """ Extracts features for each channel from a complete video

    :param video_file: (path) path to video file we want to extract features from
    :param params: (dict) control params for the feature extraction
    :return: Features extracted if successful, else False.
    """

    interval = params.get("interval")

    # get video fps, duration, & => total number of frames to process
    video_properties = get_video_properties(video_file)
    if isinstance(video_properties, bool) and not video_properties:
        return False
    duration, fps = video_properties.get("duration"), video_properties.get("fps")
    total_num_frames = video_properties.get("num_frames")
    frame_time = 1.0 / fps

    print(video_properties)

    # iterate through video frame by frame...
    video_capture = cv2.VideoCapture(video_file)
    features = []

    # frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    # fps = int(video_capture.get(cv2.CAP_PROP_FPS))
    # num_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    # duration = num_frames / fps
    # print(duration, fps, num_frames, frame_height, frame_width)
    # return

    pbar = tqdm(total=total_num_frames)
    for idx in range(total_num_frames):
        pbar.update(1)

        # get the next frame
        ret, frame = video_capture.read()
        if isinstance(ret, bool) and not ret:
            return False

        # extract features for both channels incl. curr time stamps (only for desired fps)
        if idx % interval == 0:
            frame_features = extract_features_from_frame(frame)
            if isinstance(frame_features, bool) and not frame_features:
                return False
            frame_features["time"] = idx * frame_time
            features.append(frame_features)

        idx += 1
        if idx > 10000:
            break

    pbar.close()
    video_capture.release()
    cv2.destroyAllWindows()

    pickle.dump(features, Path("./features_example_big.pkl").open('wb'), pickle.HIGHEST_PROTOCOL)
    return True


def extract_track_features_from_video_features(video_features):
    """

    :param video_features:
    :return: Dictionary containing time-series features for each track if successful, False otherwise.
    """

    curr_left_track_name, curr_right_track_name = None, None
    curr_left_track_artist, curr_right_track_artist = None, None
    features = {}

    try:
        for i, frame_features in enumerate(video_features):
            frame_features["left"]["time"] = frame_features["time"]
            frame_features["right"]["time"] = frame_features["time"]

            # work out whether we've transitioned tracks & append data to the right place
            left_track_name, right_track_name = frame_features["left"]["title"], frame_features["right"]["title"]
            left_track_artist, right_track_artist = frame_features["left"]["artist"], frame_features["right"]["artist"]
            if left_track_name != curr_left_track_name or left_track_artist != curr_left_track_artist:
                curr_left_track_artist, curr_left_track_name = left_track_artist, left_track_name
                if curr_left_track_artist not in features:
                    features[left_track_artist] = {left_track_name: []}
                else:
                    features[left_track_artist][left_track_name] = []
            if right_track_name != curr_right_track_name or right_track_artist != curr_right_track_artist:
                curr_right_track_artist, curr_right_track_namae = right_track_artist, right_track_name
                if curr_right_track_artist not in features:
                    features[right_track_artist] = {right_track_name: []}
                else:
                    features[right_track_artist][right_track_name] = []

            # append to overall feature collection
            features[left_track_artist][left_track_name].append(frame_features["left"])
            features[right_track_artist][right_track_name].append(frame_features["right"])

        return features
    except Exception as e:
        print(f"Exception extracting per-track features from video features: \n{e}")
        return False
