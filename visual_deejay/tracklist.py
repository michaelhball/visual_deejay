def create_tracklist(video_features):
    """ Createst a time-stamped tracklist from video features (in video time-space)

    :param video_features:
    :return: Tracklist if successful (artist, track, & starting timestamp), False otherwise.
    """

    # TODO: create tracklist using video time-stamps then can convert --> audio file time-stamps
    tracklist = []
    FADER_THRESH = 0.5
    curr_left_track_name, curr_right_track_name = None, None
    curr_left_track_artist, curr_right_track_artist = None, None

    try:
        for i, frame_features in enumerate(video_features):
            left_track, right_track, time = frame_features["left"], frame_features["left"], frame_features["time"]

            # work out whether we've transitioned tracks & add to tracklist but curr variables
            left_track_name, right_track_name = left_track["title"], right_track["title"]
            left_track_artist, right_track_artist = left_track["artist"], right_track["artist"]

            if left_track_name != curr_left_track_name or left_track_artist != curr_left_track_artist:
                if left_track["fader"] > FADER_THRESH:
                    tracklist.append({"time": time, "artist": left_track_artist, "name": left_track_name})
                    curr_left_track_artist, curr_left_track_name = left_track_artist, left_track_name
            elif right_track_name != curr_right_track_name or right_track_artist != curr_right_track_artist:
                if right_track["fader"] > FADER_THRESH:
                    tracklist.append({"time": time, "artist": right_track_artist, "name": right_track_name})
                    curr_right_track_artist, curr_right_track_name = right_track_artist, right_track_name

        return tracklist
    except Exception as e:
        print(f"Exception creating tracklist: {e}")
        return False



