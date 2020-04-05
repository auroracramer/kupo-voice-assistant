import sys
import time
from collections import deque

import numpy as np

from kupo.base import AudioModule
from kupo.consts import CHUNK


class MusicVizModule(AudioModule):
    def __init__(self, buffer_len=100, onset_timeout=0.2, *args, **kwargs):
        super(MusicVizModule, self).__init__(name="music-viz", *args, **kwargs)
        self.buffer_len = buffer_len
        self.prev_frame = None
        self.flux_buffer = None
        self.reset()
        self.onset_timeout = onset_timeout
        self.onset_timeout_ts = None

    def reset(self):
        self.prev_frame = np.zeros((CHUNK // 2, ))
        self.flux_buffer = deque(maxlen=self.buffer_len)

    def process_audio(self, inp_buf):
        fft = np.abs(np.fft.rfft(inp_buf))
        energy = fft[0]
        spec = fft[1:]
        flux = np.abs(spec - self.prev_frame).mean()
        self.flux_buffer.append(flux)

        flux_threshold = np.percentile(self.flux_buffer, 80)

        curr_ts = time.time()
        if self.onset_timeout_ts is not None and (curr_ts - self.onset_timeout_ts) >= self.onset_timeout:
            self.onset_timeout_ts = None

        is_onset = flux > flux_threshold
        if len(self.flux_buffer) == self.buffer_len and is_onset and self.onset_timeout_ts is None:
            self.onset_timeout_ts = curr_ts
            print("BEAT ", end="")
            sys.stdout.flush()