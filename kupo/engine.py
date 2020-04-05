import pyaudio

from kupo.ambient import MusicVizModule
from kupo.base import ModuleManager
from kupo.consts import FORMAT, CHANNELS, SAMPLE_RATE, CHUNK, STOP_FLAG
from kupo.context import TTSContextModule
from kupo.utils import audio_byte_to_numpy_float
from kupo.voice import AssistantAudioHandler


class Kupo(object):
    def __init__(self):
        self.manager = ModuleManager()

        # Set up context modules
        self.context_modules = [
            TTSContextModule(),
        ]
        for module in self.context_modules:
            self.manager.register_module(module)

        # Set up audio modules
        self.audio_modules = [
            AssistantAudioHandler(),
            MusicVizModule(),
        ]
        for module in self.audio_modules:
            self.manager.register_module(module)

    def start(self):
        audio_handler = pyaudio.PyAudio()
        # Requires default audio devices to be set!
        input_stream = audio_handler.open(format=FORMAT,
                                          channels=CHANNELS,
                                          rate=SAMPLE_RATE,
                                          input=True,
                                          frames_per_buffer=CHUNK)

        self.manager.start()
        try:
            while True:
                mic_data = input_stream.read(CHUNK, exception_on_overflow=False)
                inp_buf = audio_byte_to_numpy_float(mic_data)

                for module in self.audio_modules:
                    module.send(inp_buf)

        except (KeyboardInterrupt, SystemExit):
            print("Process terminated. Exiting.")
        finally:
            self.manager.send(STOP_FLAG)
            self.manager.process.join()
            input_stream.stop_stream()
            input_stream.close()
            audio_handler.terminate()