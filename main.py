import cv2

from visual_deejay.feature_extraction import extract_features_from_frame, extract_features_from_video


if __name__ == '__main__':
    # file_path = "/Users/michaelball/Desktop/projects/visual_deejay/res/example_rekordbox_screenshot.png"
    # frame = cv2.imread(file_path)
    # extract_features_from_frame(frame)

    extract_features_from_video()
