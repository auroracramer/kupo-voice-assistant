import numpy as np


def audio_byte_to_numpy_float(data):
    int16_norm = -np.iinfo('int16').min
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / int16_norm


def numpy_float_to_audio_byte(data):
    int16_norm = -np.iinfo('int16').min
    # Expecting only mono
    return (data * int16_norm).astype(np.int16).flatten().tobytes()