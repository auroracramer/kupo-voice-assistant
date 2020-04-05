# Kupo Voice Assistant

An open-source modular voice assistant and general ambient audio processor. Speech recognition (via [DeepSpeech](https://github.com/mozilla/DeepSpeech)) and keyword detection (via [porcupine](https://github.com/Picovoice/porcupine)) are all done on device to avoid network communication of audio or speech transcriptions. Ambient audio processing can also be done, eventually for visualization using [AdaFruit NeoPixel](https://www.adafruit.com/category/168). This is meant to run on a Raspberry Pi.

##### Current Features
* Keyword detection and speech recognition
* Simple phrase based queries
* TTS responses
* Spectral flux based onset detection

##### Upcoming Features
* NeoPixel integration for ambient/music visualization
* Beat visualization using onset detection
* Other types of visualization modes
* Other types of queries
* Integration with streaming services like Spotify
* Integration with [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) for context-aware queries (and for fun visualizations)

This is a work in progress!

