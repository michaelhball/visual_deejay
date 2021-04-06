import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pickle
import scipy.stats

__all__ = ['get_metadata', 'load_file', 'strip']


def get_metadata(y, sampling_rate=44100):
    """ Extract track metadata properties.

    :param y: track signal
    :param sampling_rate: sampling rate
    :return: Dictionary containing all metadata properties
    """

    return {
        "duration": librosa.get_duration(y, sampling_rate),
    }


def load_file(file_path, sampling_rate=44100, duration=None):
    """ Load audio file from disc

    :param file_path: (str) path to audio file
    :param sampling_rate: (int)
    :param duration: (float) optional duration (in s) if we want only a snippet of the track.
    :return:
    """

    try:
        file_type = file_path.split(".")[-1]
        if file_type == "flac":
            y, sr = librosa.load(file_path, sr=sampling_rate, duration=duration)
            return y, sr
        else:
            print(f".{file_type} filetype is not currently supported")
            return False
    except:
        print("Exception loading audio file")
        return False


def strip(y, frame_length, hop_length=512):
    """ Removing leading silence from an audio track

    :param y: (np.ndarray) audio signal
    :param frame_length: (int)
    :param hop_length: (int)
    :return: Audio signal with leading silence removed
    """

    # compute RMSE.
    rms = librosa.feature.rms(y, frame_length=frame_length, hop_length=hop_length, center=True)

    # identify the first frame index where RMSE exceeds a threshold.
    thresh = 0.01
    frame_index = 0
    while rms[0][frame_index] < thresh:
        frame_index += 1

    # convert units of frames to samples.
    start_sample_index = librosa.frames_to_samples(frame_index, hop_length=hop_length)

    # return the trimmed signal.
    return y[start_sample_index:]


def melspectrogram(y, sampling_rate=44100, n_ftt=4096, hop_length=256):
    """

    :param y:
    :param sampling_rate:
    :param n_ftt:
    :param hop_length:
    :return:
    """

    S = librosa.feature.melspectrogram(y, sr=sampling_rate, n_fft=n_ftt, hop_length=hop_length)
    return S


def amplitude_to_db(Y):
    """

    :param Y:
    :return:
    """

    S = librosa.amplitude_to_db(abs(Y))
    return S


def convert_power_to_db(S):
    """

    :param S:
    :return:
    """

    logS = librosa.power_to_db(abs(S))
    return logS


def display_spectrogram(S, hop_length=512, sr=44100, y_axis='linear'):
    """

    :param S:
    :param hop_length: (int)
    :param sr: (int)
    :param y_axis: (str) 'linear' | 'mel' (~ to log(1+f)) | 'log' | ''
    :return:
    """

    plt.figure(figsize=(15, 5))
    librosa.display.specshow(S, sr=sr, hop_length=hop_length, x_axis='time', y_axis=y_axis)
    plt.colorbar(format='%+2.0f dB')
    plt.show()


def get_frames(y, frame_length=1024, hop_length=512):
    """ NB: this is often not required because Librosa will auto-segment inside the feature extraction functions.

    :param y:
    :param frame_length:
    :param hop_length:
    :return:
    """

    try:
        return librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
    except:
        print("Exception getting frames")
        return False


def extract_tempo(y, sampling_rate=44100, tempo_type="static", prior_type=None):
    """

    :param y:
    :param prior:
    :param sampling_rate:
    :param tempo_type:
    :param prior_type:
    :return:
    """

    onset_env = librosa.onset.onset_strength(y, sr=sampling_rate)
    if tempo_type == "static":
        aggregate = np.mean
        prior = scipy.stats.uniform(30, 300) if prior_type == "uniform" else None
    else:
        aggregate = None
        prior = scipy.stats.lognorm(loc=np.log(120), scale=120, s=1) if prior_type == "lognorm" else None
    tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sampling_rate, aggregate=aggregate, prior=prior)
    return tempo


def extract_beat_frames(y, sr=44100, hop_length=512):
    """ Can convert these to time using frames_to_time so we get the beat timestamps for adding effects to the image.

    :param y:
    :param sr:
    :param hop_length:
    :return: Returns tempo estimate (bpm) & an array of frame numbers corresponding to detected beat events.
    """

    return librosa.beat.beat_track(y, sr=sr, hop_length=hop_length)


def frames_to_time(frames, sampling_rate=44100, hop_length=512):  # v useful for plotting
    """

    :param frames:
    :param sampling_rate:
    :param hop_length:
    :return:
    """

    return librosa.frames_to_time(frames, sr=sampling_rate, hop_length=hop_length)


def mfcc(y, sr=44100, hop_length=512, n_mfcc=20):
    """

    :param y: (np.ndarray) the audio signal (1D array where y[t] = amplitude of waveform at sample t)
    :param sr: (int) number of samples per second of audio ()
    :param hop_length: (int) # of samples between successive frames (=> determines T)
    :param n_mfcc: (int) number of mfcc's we want to return
    :return: mfcc of size (n_mpcc, T), where T is the track duration in frames
    """

    return librosa.feature.mfcc(y, sr=sr, hop_length=hop_length, n_mfcc=n_mfcc)


def stft(y, n_fft=2048, hop_length=512):
    """ Short-Time Fourier Transform

    :param y:
    :param n_fft: (int) frame size (i.e. the size of the Fourier Transform)
    :param hop_length:
    :return:
    """

    return librosa.stft(y, n_fft=n_fft, hop_length=hop_length)


def zcr(y):
    """ ZCR (Zero-Crossing Rate): Number of times the signal crosses the x-axis.

    :param y:
    :return:
    """

    return librosa.feature.zero_crossing_rate(y + 0.0001)  # constant added to avoid high ZCR for silences ~0


def compute_feature_delta(data):
    """ Computes smoothed first-order differences among columns of input (i.e. most commonly among frames) => same
        shape as input.

    :param data:
    :return:
    """

    return librosa.feature.delta(data)


def rmse(y, frame_length=1024, hop_length=512):
    """ Energy of the signal is defined as total magnitude => corresponds to how loud it is.

    :param y:
    :param frame_length:
    :param hop_length:
    :return:
    """

    try:
        return librosa.feature.rms(y, frame_length=frame_length, hop_length=hop_length, center=True)[0]
    except:
        print("Exception computing RMSE")
        return False


def hpss(y):
    """

    :param y:
    :return:
    """

    try:
        y_harm, y_perc = librosa.effects.hpss(y)
        return y_harm, y_perc
    except:
        print("Exception performing harmonic & percussive source separation")
        return False


def pipeline(track_file_path, frame_length=1024, hop_length=512, n_fft=2048, sampling_rate=44100):
    """

    :param track_file_path: (path) path to audio file we want to analyse
    :param hop_length: (int) number of samples between successive frames
    :param n_fft: ()
    :param sampling_rate: (int)
    :return:
    """

    # load audio file as waveform features
    loaded = load_file(track_file_path, sampling_rate=sampling_rate)
    if isinstance(loaded, bool) and not loaded:
        return False
    y, sr = loaded

    # song metadata
    metadata = get_metadata(y, sr)
    if isinstance(metadata, bool) and not metadata:
        return False

    print(metadata)

    # extract tempo, both  static & dynamic
    static_tempo = extract_tempo(y, sampling_rate, "static", None)  # use uniform only for electronic music
    if isinstance(static_tempo, bool) and not static_tempo:
        return False
    dynamic_tempo = extract_tempo(y, sampling_rate, "dynamic", "lognorm")
    if isinstance(dynamic_tempo, bool) and not dynamic_tempo:
        return False
    print(static_tempo)

    # extract beat times
    beats = extract_beat_frames(y, sr=sampling_rate, hop_length=hop_length)
    if isinstance(beats, bool) and not beats:
        return False
    another_dynamic_tempo, beat_frames = beats

    # Short-Time Fourier Transform
    Y = stft(y, n_fft=n_fft, hop_length=hop_length)  # 1025 frequency bins x num_frames
    if isinstance(Y, bool) and not Y:
        return False
    print(Y.shape)

    # Compute ZCR
    Z = zcr(y)
    if isinstance(Z, bool) and not Z:
        return False
    print(Z.shape)          

    # Mel-Frequency Cepstral Coefficient
    M = mfcc(y, sr=sampling_rate, hop_length=hop_length, n_mfcc=13)
    if isinstance(M, bool) and not M:
        return False
    print(M.shape)

    from pathlib import Path
    pickle.dump(M, Path('laurence_M.pkl').open('wb'))

    # feature manipulation: delta
    M_delta = compute_feature_delta(M)
    if isinstance(M_delta, bool) and not M_delta:
        return False
    print(M_delta.shape)

    print(len(y))

    # # Harmonic-Percussive Source Separation
    # separated = hpss(y)
    # if isinstance(separated, bool) and not separated:
    #     return False
    # y_harmonic, y_percussive = separated


if __name__ == '__main__':
    file_path = "./res/laurence_guy__dissociation_in_the_car_park_at_sainbury.flac"
    pipeline(file_path)

    # from pathlib import Path
    # M = pickle.load(Path("laurence_M.pkl").open('rb'))
    # print(M.shape)
    # M = M[:, 25000:26000]
    #
    # sr = 44100
    # hop_length = 512
    #
    # display_spectrogram(M, hop_length=hop_length, sr=sr, y_axis='log')
    # M_scaled = sklearn.preprocessing.scale(M, axis=1)  # scale MFCC so that each coef. dim. has 0 mean & unit var
    # display_spectrogram(M_scaled, hop_length=hop_length, sr=sr, y_axis='linear')
