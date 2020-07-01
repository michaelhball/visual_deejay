import cv2

from visual_deejay.feature_extraction import extract_features_from_video


if __name__ == '__main__':
    # file_path = "/Users/michaelball/Desktop/projects/visual_deejay/res/example_rekordbox_screenshot.png"
    # frame = cv2.imread(file_path)
    # extract_features_from_frame(frame)

    # TODO: automatically generate time-stamped track-list for a mix (e.g. when a track becomes non-zero)
    # TODO: make feature extraction more robust to different image resolutions/positions i.e. in theory I want a model
    #  that can extract the stuff automatically

    # TODO: write code that takes in the complete feature time series and
    #  1) automatically creates a tracklist with timestamps
    #  2) extracts the features for each song separately incl. removing all leading zeros for each & converting time
    #     remaining to actual time-stamp in the song (i.e. from 0 to len(song)).

    # TODO: work out how to sync screen recording video to audio recording... not easy because we audio start !=
    #  song 1 play. --> NB: on a side note, could try & find actual moment of mix start by seeing when the 'play'
    #  button switches from off to on.

    # TODO: write the stuff that computes useful things based on the complete time a song is playing, e.g.
    #  the fraction multiplier that trim applies to the mix should be computed as a fraction of the maximum trim value
    #  that is used by that song (there is no absolute maximum, e.g. because different tracks have diff volumes).

    video_file = "/Users/michaelball/Desktop/big_breakfast.mov"
    params = {"interval": 10}  # => 6 fps
    extract_features_from_video(video_file, params)
