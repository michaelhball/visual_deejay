import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats


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


def display_spectrogram(S, hop_length=512, sampling_rate=44100, y_axis='linear'):
    """

    :param S:
    :param hop_length: (int)
    :param sampling_rate: (int)
    :param y_axis: (str) 'linear' | 'mel' (~ to log(1+f)) | 'log' | ''
    :return:
    """

    plt.figure(figsize=(15, 5))
    librosa.display.specshow(S, sr=sampling_rate, hop_length=hop_length, x_axis='time', y_axis=y_axis)
    plt.colorbar(format='%+2.0f dB')


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


def frames_to_time(frames, sampling_rate=44100, hop_length=512):  # v useful for plotting
    """

    :param frames:
    :param sampling_rate:
    :param hop_length:
    :return:
    """

    return librosa.frames_to_time(frames, sr=sampling_rate, hop_length=hop_length)


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


def get_metadata(y, sampling_rate=44100):
    """

    :param y:
    :param sampling_rate:
    :return:
    """

    return {
        "duration": librosa.get_duration(y, sampling_rate),
    }


def load_file(file_path, sampling_rate=44100):
    """

    :param file_path:
    :param sampling_rate:
    :return:
    """

    try:
        file_type = file_path.split(".")[-1]
        if file_type == "flac":
            y, sr = librosa.load(file_path, sr=sampling_rate)
            return y, sr
        else:
            print(f".{file_type} filetype is not currently supported")
            return False
    except:
        print("Exception loading audio file")
        return False


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

    :param track_file_path:
    :param hop_length:
    :param n_fft:
    :param sampling_rate:
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

    # extract tempo, both  static & dynamic
    static_tempo = extract_tempo(y, sampling_rate, "static", "uniform")
    if isinstance(static_tempo, bool) and not static_tempo:
        return False
    dynamic_tempo = extract_tempo(y, sampling_rate, "dynamic", "lognorm")
    if isinstance(dynamic_tempo, bool) and not dynamic_tempo:
        return False

    # Short-Time Fourier Transform
    Y = stft(y, n_fft=n_fft, hop_length=hop_length)  # 1025 frequency bins x num_frames
    if isinstance(Y, bool) and not Y:
        return False

    # Harmonic-Percussive Source Separation
    separated = hpss(y)
    if isinstance(separated, bool) and not separated:
        return False
    y_harmonic, y_percussive = separated


if __name__ == '__main__':
    file_path = "../res/joe_satriani__slow_down_blues.flac"
    pipeline(file_path)
