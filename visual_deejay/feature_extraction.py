import cv2
import math
import numpy as np

from .utils import angle_between_two_lines, plot_one


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
    # extract fader value
    fader_img = channel_img[300:, :, :]
    print(fader_img.shape)
    plot_one(fader_img)

    # extract trim value
    trim_img = channel_img[8:50, :, :]
    plot_one(trim_img)
    trim_val = extract_knob_angle(trim_img)
    return

    # extract highs value
    highs_img = channel_img[74:116, :, :]
    print(highs_img.shape)
    plot_one(highs_img)
    highs_angle = extract_knob_angle(highs_img)

    # extract mids value
    mids_img = channel_img[140:182, :, :]
    print(mids_img.shape)
    plot_one(mids_img)
    mids_angle = extract_knob_angle(mids_img)

    # extract lows value
    lows_img = channel_img[206:248, :, :]
    print(lows_img.shape)
    plot_one(lows_img)
    lows_angle = extract_knob_angle(lows_img)

    return {}


def extract_features_from_frame(frame):
    """

    :param frame:
    :return:
    """

    # TODO: convert all image croppings to relative rather than absolute in case they don't work with resolution of vid
    left_channel_eqs_img = frame[600:1010, 1590:1665, :]
    right_channel_eqs_img = frame[600:1010, 1695:1770, :]
    print(f"channel eqs images shapes -- left: {left_channel_eqs_img.shape}, right: {right_channel_eqs_img}")

    # left_channel_features = extract_channel_features(left_channel_eqs_img)
    # if isinstance(left_channel_features, bool) and not left_channel_features:
    #     return False

    right_channel_features = extract_channel_features(right_channel_eqs_img)
    if isinstance(right_channel_features, bool) and not right_channel_features:
        return False

    return True


def extract_features_from_video():
    pass
