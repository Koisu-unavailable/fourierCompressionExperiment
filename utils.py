import numpy as np

def is_mono(data: np.ndarray) -> bool:
    """
    Check if the audio data is mono.
    :param data: Audio data as a numpy array.
    :return: True if mono, False otherwise.
    """
    return len(data.shape) == 1 or (len(data.shape) == 2 and data.shape[1] == 1)