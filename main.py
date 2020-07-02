import cv2
import numpy as np
import pickle

from pathlib import Path

from visual_deejay.tracklist import create_tracklist
from visual_deejay.utils import display_video_frame
from visual_deejay.video_feature_extraction import (clean_video_features, extract_features_from_video,
                                                    extract_track_features_from_video_features)


if __name__ == '__main__':
    # file_path = "/Users/michaelball/Desktop/projects/visual_deejay/res/example_rekordbox_screenshot.png"
    # frame = cv2.imread(file_path)
    # extract_features_from_frame(frame)

    # TODO: make feature extraction more robust to different image resolutions/positions i.e. in theory I want a model
    #  that can extract the stuff automatically

    # TODO: work out how to sync screen recording video to audio recording... not easy because we audio start !=
    #  song 1 play. --> NB: on a side note, could try & find actual moment of mix start by seeing when the 'play'
    #  button switches from off to on.

    # TODO: write the stuff that computes useful things based on the complete time a song is playing, e.g.
    #  the fraction multiplier that trim applies to the mix should be computed as a fraction of the maximum trim value
    #  that is used by that song (there is no absolute maximum, e.g. because different tracks have diff volumes).

    # TODO: convert tracklist time-stamps from video space --> audio space

    # TODO: test whether fps, num_frames etc. coming from my function are correct (cv2 functions give diff response).

    video_file = "/Users/michaelball/Desktop/big_breakfast.mov"
    video_features = pickle.load(Path('./features_example_big.pkl').open('rb'))
    video_features = clean_video_features(video_features)
    tracklist = create_tracklist(video_features)
    print(tracklist)

    # params = {"interval": 10}  # => 6 fps
    # extract_features_from_video(video_file, params)

    # video_capture = cv2.VideoCapture(video_file)
    # per_track_features = extract_track_features_from_video_features(video_features)
    # for artist, tracks in per_track_features.items():
    #     for track, track_features in tracks.items():
    #         if artist != "Random Movement":
    #             continue
    #         trim = [f['eqs']['fader'] for f in track_features]
    #         print(trim)
    #         print(max(trim), track_features[np.argmax(trim)]["og_idx"])
    #         video_capture.set(1, track_features[np.argmax(trim)]["og_idx"])
    #         # ret, frame = video_capture.read()
    #         # plt.imshow(frame)
    #         # plt.show()
    #         print(min(trim), track_features[np.argmin(trim)]["og_idx"])
    #         video_capture.set(1, track_features[np.argmin(trim)]["og_idx"])
    #         # ret, frame = video_capture.read()
    #         # plt.imshow(frame)
    #         # plt.show()
    #         # TODO: work out physically what these angles mean => how to convert them all to a 0-1 scale that is track-specific
    #         #  & do the same for the other knobs although they shouldn't see a lot of change
