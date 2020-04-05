import pyttsx3
from kupo.base import AsyncModule


class NeoPixelContextModule(AsyncModule):
    def __init__(self):
        super(NeoPixelContextModule, self).__init__("neopixel")


class TTSContextModule(AsyncModule):
    def __init__(self):
        super(TTSContextModule, self).__init__("tts")
        self.tts_engine = pyttsx3.init()

    def say(self, msg):
        self.tts_engine.say(msg)
        self.tts_engine.runAndWait()

    def process_msg(self, msg):
        self.say(msg)