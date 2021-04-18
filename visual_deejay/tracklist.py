"""
Functions for automatically generating a tracklist based off screen recording of DJ software.
"""

__all__ = ["create_tracklist", "find_first_track_start"]


def find_first_track_start(video_features):
    """ Find the exact frame at which the first track starts playing and its channel.

    :param video_features: (list(dict)) list of features extracted from screen recording video.
    :return: Tuple containing the index & the channel as a string if successful, False otherwise.
    """

    try:
        FADER_THRESH = 0.5
        for i, frame_features in enumerate(video_features):
            left_track, right_track, time = frame_features["left"], frame_features["right"], frame_features["time"]
            if left_track["eqs"]["fader"] > FADER_THRESH and video_features[i+1]["left"]["time_gone"] > 0:
                return i, "left"
            elif right_track["eqs"]["fader"] > FADER_THRESH and video_features[i+1]["right"]["time_gone"] > 0:
                return i, "right"
        return False
    except:
        print("Exception finding first track start")
        return False


def create_tracklist(video_features):
    """ Creates a time-stamped tracklist from video features (in video time-space)

    :param video_features: (list(dict)) list of features for each time-stamp in screen-recording video
    :return: Tracklist if successful (artist, track, & starting timestamp), False otherwise.
    """

    # setup
    tracklist = []
    FADER_THRESH = 0.5
    curr_track_artist, curr_track_name = {"left": None, "right": None}, {"left": None, "right": None}

    # find index where the first track starts playing
    first_track_start = find_first_track_start(video_features)
    if isinstance(first_track_start, bool) and not first_track_start:
        return False
    first_track_start_idx, first_track_side = first_track_start

    try:
        for i, frame_features in enumerate(video_features):

            if i < first_track_start_idx:
                continue

            elif i == first_track_start_idx:
                first_track_artist = frame_features[first_track_side]["artist"]
                first_track_name = frame_features[first_track_side]["title"]
                tracklist.append({"time": frame_features["time"], "artist": first_track_artist, "name": first_track_name})
                curr_track_artist[first_track_side] = first_track_artist
                curr_track_name[first_track_side] = first_track_name

            else:
                left_track, right_track, time = frame_features["left"], frame_features["right"], frame_features["time"]
                left_track_name, left_track_artist = left_track["title"], left_track["artist"]
                right_track_name, right_track_artist = right_track["title"], right_track["artist"]

                # make sure we have the new artists & tracks in our dictionary
                if left_track_name != curr_track_name["left"] or left_track_artist != curr_track_artist["left"]:
                    if left_track["eqs"]["fader"] > FADER_THRESH:
                        tracklist.append({"time": time, "artist": left_track_artist, "name": left_track_name})
                        curr_track_artist["left"], curr_track_name["left"] = left_track_artist, left_track_name

                if right_track_name != curr_track_name["right"] or right_track_artist != curr_track_artist["right"]:
                    if right_track["eqs"]["fader"] > FADER_THRESH:
                        tracklist.append({"time": time, "artist": right_track_artist, "name": right_track_name})
                        curr_track_artist["right"], curr_track_name["right"] = right_track_artist, right_track_name

        return tracklist
    except Exception as e:
        print(e)
        return False
