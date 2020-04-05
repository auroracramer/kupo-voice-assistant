import string
import struct
import time
from collections import deque

import numpy as np
import pvporcupine
import webrtcvad

from kupo.base import AudioModule, AsyncModule
from kupo.utils import numpy_float_to_audio_byte
from kupo.consts import SAMPLE_RATE
import kupo.deepspeech_utils as ds_transcriber


class QueryModule(AsyncModule):
    def process_query(self, words):
        raise NotImplementedError()

    def process_msg(self, msg):
        # Could do type check here
        self.process_query(msg)


class SubseqMatchQueryModule(QueryModule):
    def __init__(self, target_words, *args, **kwargs):
        super(SubseqMatchQueryModule, self).__init__(*args, **kwargs)
        if type(target_words) == str:
            self.target_words = target_words.split()
        else:
            self.target_words = target_words

    def is_match(self, words):
        if len(words) < len(self.target_words):
            return False

        for start_idx in range(len(words) - len(self.target_words) + 1):
            end_idx = start_idx + len(self.target_words)
            if self.target_words == words[start_idx:end_idx]:
                return True

        return False


class ConstResponseCommandQueryModule(SubseqMatchQueryModule):
    def __init__(self, target_words, response, *args, **kwargs):
        super(ConstResponseCommandQueryModule, self).__init__(target_words, *args, **kwargs)
        self.response = response

    def process_query(self, words):
        if self.is_match(words):
            self.manager.send(("tts", self.response))


class AssistantAudioHandler(AudioModule):
    def __init__(self, manager=None, name="assistant", cmd_timeout=10.0, vad_timeout=1.0,
                 vad_buffer_len=32, vad_aggressiveness=2):
        super(AssistantAudioHandler, self).__init__(name, manager=manager)
        self.keyword_handler = pvporcupine.create(library_path=PORCUPINE_LIB_PATH,
                                                  model_file_path=PORCUPINE_MODEL_PATH,
                                                  keywords=["picovoice"])
        self.voice_active = False
        self.activate_ts = None
        self.vad_timeout_ts = None
        self._activate = False
        self._deactivate = False
        self.cmd_timeout = cmd_timeout
        self.vad_timeout = vad_timeout
        self.vad = webrtcvad.Vad(int(vad_aggressiveness))
        self.vad_buffer = None
        self.vad_buffer_len = vad_buffer_len

        # Set up query modules
        self.query_modules = [
            ConstResponseCommandQueryModule("this is a test", "hear you loud and clear", name="test-command-1"),
            ConstResponseCommandQueryModule("guess what", "chicken butt", name="test-command-2")
        ]

        for m in self.query_modules:
            m.register_manager(self.manager)

        # Resolve all the paths of model files
        output_graph, lm, trie = ds_transcriber.resolve_models(DEEPSPEECH_MODEL_PATH)
        # Load output_graph, alpahbet, lm and trie
        self.speech_model = ds_transcriber.load_model(output_graph, lm, trie)[0]
        self.stream_ctx = None

    def register_manager(self, manager):
        super(AssistantAudioHandler, self).register_manager(manager)
        try:
            for module in self.query_modules:
                module.register_manager(manager)
        except AttributeError:
            pass

    def process_audio(self, inp_buf):
        pcm = numpy_float_to_audio_byte(inp_buf)
        pcm = struct.unpack_from("h" * self.keyword_handler.frame_length, pcm)
        keyword_index = self.keyword_handler.process(pcm)
        if (type(keyword_index) == int and keyword_index >= 0) or keyword_index:
            # JTC: For now, just have the same behavior for all keywords
            self.voice_active = True
            self._activate = True
            self.activate_ts = time.time()
            self.vad_buffer = deque(maxlen=self.vad_buffer_len)

            print("Keyword detected.")

        if self.voice_active:
            # Time out check
            ts = time.time()
            if (ts - self.activate_ts) >= self.cmd_timeout:
                self._deactivate = True
                self.vad_timeout_ts = None

            # Update VAD
            # Hack: take first 30 ms and last 30 ms, take maximum value
            samples_per_frame = int(SAMPLE_RATE * 0.03)
            inp_bytes_1 = numpy_float_to_audio_byte(inp_buf[:samples_per_frame])
            inp_bytes_2 = numpy_float_to_audio_byte(inp_buf[-samples_per_frame:])
            is_speech = self.vad.is_speech(inp_bytes_1, SAMPLE_RATE) or self.vad.is_speech(inp_bytes_2, SAMPLE_RATE)
            self.vad_buffer.append(is_speech)
            vad_active = len(self.vad_buffer) < self.vad_buffer_len or np.mean(self.vad_buffer) > 0.9

            # Handle VAD timeout
            if self.vad_timeout_ts is None and not vad_active:
                # If VAD timeout hasn't started, and VAD is not currently active -> start VAD timeout
                self.vad_timeout_ts = time.time()
            elif self.vad_timeout_ts is not None and vad_active:
                # If VAD timeout has started, but VAD is active -> stop VAD timeout
                self.vad_timeout_ts = None
            elif self.vad_timeout_ts is not None and not vad_active and (ts - self.vad_timeout_ts) >= self.vad_timeout:
                # If VAD timeout has started, VAD is not active, and timeout has run out -> deactivate
                self.vad_timeout_ts = None
                self._deactivate = True
                self.vad_timeout_ts = None

            if self._activate:
                # Handle query initialization
                self.stream_ctx = self.speech_model.createStream()
                self._activate = False
            elif self._deactivate:
                # Handle query ending
                self.voice_active = False
                self._deactivate = False
                self.activate_ts = None
                speech = self.speech_model.finishStream(self.stream_ctx)
                print("Transcription: {}".format(speech))
                self.process_speech(speech)
            else:
                # Consume audio
                # 512 Block size
                assert (inp_buf.shape[0]) % 512 == 0

                # Block into chunks of 16 samples
                for start_idx in range(0, inp_buf.shape[0], 512):
                    end_idx = start_idx + 512
                    # Model takes in int16
                    chunk = np.frombuffer(numpy_float_to_audio_byte(inp_buf[start_idx:end_idx]), np.int16)
                    self.speech_model.feedAudioContent(self.stream_ctx, chunk)

    @staticmethod
    def preprocess_speech(speech):
        tokens = speech.split()
        words = []
        for token in tokens:
            word = token.translate(str.maketrans('', '', string.punctuation)).lower()
            words.append(word)
        return words

    def process_speech(self, speech):
        words = self.preprocess_speech(speech)
        for module in self.query_modules:
            # Only allow for first module to activate
            success = module.process_query(words)
            if success:
                break

    def close(self):
        self.keyword_handler.delete()


PORCUPINE_LIB_PATH = "/home/jsondotload/projects/personal/assistant/porcupine/lib/linux/x86_64/libpv_porcupine.so"
PORCUPINE_MODEL_PATH = "/home/jsondotload/projects/personal/assistant/porcupine/lib/common/porcupine_params.pv"
DEEPSPEECH_MODEL_PATH = "/home/jsondotload/projects/personal/assistant/deepspeech/deepspeech-0.6.1-models"