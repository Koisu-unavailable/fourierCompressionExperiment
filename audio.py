import numpy as np



def get_amplitude_at_time(time, samplerate, data: np.ndarray):
    if len(data.shape) == 1:
        # mono
        # Assuming data is a 1D array for mono audio
        # if you put in more than 2 channels, kys
        return data[int(time * samplerate)]
    if data.shape[1] == 2:
        # stereo
        return data[:, 0][int(time * samplerate)], data[:, 1][int(time * samplerate)]
    else:
        raise Exception("We only support mono and stereo audio files")

