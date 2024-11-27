import cv2
import time
import queue
import typing
import logging
import threading
import importlib
import numpy as np

from . import Image
from mltu.annotations.audio import Audio
from mltu.annotations.detections import Detections

""" Implemented Transformers:
- ExpandDims - Expand dimension of data
- LabelIndexer - Convert label to index by vocab
- LabelPadding - Pad label to max_word_length
- SpectrogramPadding - Pad spectrogram to max_spectrogram_length
- AudioToSpectrogram - Convert Audio to Spectrogram
"""

class Transformer:
    def __init__(self, log_level: int = logging.INFO) -> None:
        self._log_level = log_level

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

    def __call__(self, data: typing.Any, label: typing.Any, *args, **kwargs):
        raise NotImplementedError

class ExpandDims(Transformer):
    def __init__(self, axis: int=-1):
        self.axis = axis

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return np.expand_dims(data, self.axis), label

class LabelIndexer(Transformer):
    """Convert label to index by vocab
    
    Attributes:
        vocab (typing.List[str]): List of characters in vocab
    """
    def __init__(
        self, 
        vocab: typing.List[str]
        ) -> None:
        self.vocab = vocab

    def __call__(self, data: np.ndarray, label: np.ndarray):
        return data, np.array([self.vocab.index(l) for l in label if l in self.vocab])

class LabelPadding(Transformer):
    """Pad label to max_word_length
    
    Attributes:
        padding_value (int): Value to pad
        max_word_length (int): Maximum length of label
        use_on_batch (bool): Whether to use on batch. Default: False
    """
    def __init__(
        self, 
        padding_value: int,
        max_word_length: int = None, 
        use_on_batch: bool = False
        ) -> None:
        self.max_word_length = max_word_length
        self.padding_value = padding_value
        self.use_on_batch = use_on_batch

        if not use_on_batch and max_word_length is None:
            raise ValueError("max_word_length must be specified if use_on_batch is False")

    def __call__(self, data: np.ndarray, label: np.ndarray):
        if self.use_on_batch:
            max_len = max([len(a) for a in label])
            padded_labels = []
            for l in label:
                padded_label = np.pad(l, (0, max_len - len(l)), "constant", constant_values=self.padding_value)
                padded_labels.append(padded_label)

            padded_labels = np.array(padded_labels)
            return data, padded_labels

        label = label[:self.max_word_length]
        return data, np.pad(label, (0, self.max_word_length - len(label)), "constant", constant_values=self.padding_value)

class SpectrogramPadding(Transformer):
    """Pad spectrogram to max_spectrogram_length
    
    Attributes:
        padding_value (int): Value to pad
        max_spectrogram_length (int): Maximum length of spectrogram. Must be specified if use_on_batch is False. Default: None
        use_on_batch (bool): Whether to use on batch. Default: False
    """
    def __init__(
        self, 
        padding_value: int,
        max_spectrogram_length: int = None, 
        use_on_batch: bool = False
        ) -> None:
        self.max_spectrogram_length = max_spectrogram_length
        self.padding_value = padding_value
        self.use_on_batch = use_on_batch

        if not use_on_batch and max_spectrogram_length is None:
            raise ValueError("max_spectrogram_length must be specified if use_on_batch is False")

    def __call__(self, spectrogram: np.ndarray, label: np.ndarray):
        if self.use_on_batch:
            max_len = max([len(a) for a in spectrogram])
            padded_spectrograms = []
            for spec in spectrogram:
                padded_spectrogram = np.pad(spec, ((0, max_len - spec.shape[0]), (0,0)), mode="constant", constant_values=self.padding_value)
                padded_spectrograms.append(padded_spectrogram)

            padded_spectrograms = np.array(padded_spectrograms)
            label = np.array(label)

            return padded_spectrograms, label

        padded_spectrogram = np.pad(spectrogram, ((0, self.max_spectrogram_length - spectrogram.shape[0]),(0,0)), mode="constant", constant_values=self.padding_value)

        return padded_spectrogram, label

class AudioPadding(Transformer):
    def __init__(self, max_audio_length: int, padding_value: int = 0, use_on_batch: bool = False, limit: bool = False):
        super(AudioPadding, self).__init__()
        self.max_audio_length = max_audio_length
        self.padding_value = padding_value
        self.use_on_batch = use_on_batch
        self.limit = limit

    def __call__(self, audio: Audio, label: typing.Any):
        # batched padding
        if self.use_on_batch:
            max_len = max([len(a) for a in audio])
            padded_audios = []
            for a in audio:
                # limit audio if it exceed max_audio_length
                padded_audio = np.pad(a, (0, max_len - a.shape[0]), mode="constant", constant_values=self.padding_value)
                padded_audios.append(padded_audio)

            padded_audios = np.array(padded_audios)
            # limit audio if it exceed max_audio_length
            if self.limit:
                padded_audios = padded_audios[:, :self.max_audio_length]

            return padded_audios, label

        audio_numpy = audio.numpy()
        # limit audio if it exceed max_audio_length
        if self.limit:
            audio_numpy = audio_numpy[:self.max_audio_length]
        padded_audio = np.pad(audio_numpy, (0, self.max_audio_length - audio_numpy.shape[0]), mode="constant", constant_values=self.padding_value)

        audio.audio = padded_audio

        return audio, label

class AudioToSpectrogram(Transformer):
    """Read wav file with librosa and return audio and label
    
    Attributes:
        frame_length (int): Length of the frames in samples.
        frame_step (int): Step size between frames in samples.
        fft_length (int): Number of FFT components.
        log_level (int): Logging level (default: logging.INFO)
    """
    def __init__(
            self,
            frame_length: int = 256,
            frame_step: int = 160,
            fft_length: int = 384,
            log_level: int = logging.INFO,
        ) -> None:
        super(AudioToSpectrogram, self).__init__(log_level=log_level)
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.fft_length = fft_length

        # import librosa using importlib
        try:
            self.librosa = importlib.import_module('librosa')
            print("librosa version:", self.librosa.__version__)
        except ImportError:
            raise ImportError("librosa is required to augment Audio. Please install it with `pip install librosa`.")

    def __call__(self, audio: Audio, label: typing.Any):
        """Compute the spectrogram of a WAV file

        Args:
            audio (Audio): Audio object
            label (Any): Label of audio

        Returns:
            np.ndarray: Spectrogram of audio
            label (Any): Label of audio
        """

        # Compute the Short Time Fourier Transform (STFT) of the audio data and store it in the variable 'spectrogram'
        # The STFT is computed with a hop length of 'frame_step' samples, a window length of 'frame_length' samples, and 'fft_length' FFT components.
        # The resulting spectrogram is also transposed for convenience
        spectrogram = self.librosa.stft(audio.numpy(), hop_length=self.frame_step, win_length=self.frame_length, n_fft=self.fft_length).T

        # Take the absolute value of the spectrogram to obtain the magnitude spectrum
        spectrogram = np.abs(spectrogram)

        # Take the square root of the magnitude spectrum to obtain the log spectrogram
        spectrogram = np.power(spectrogram, 0.5)

        # Normalize the spectrogram by subtracting the mean and dividing by the standard deviation.
        # A small value of 1e-10 is added to the denominator to prevent division by zero.
        spectrogram = (spectrogram - np.mean(spectrogram)) / (np.std(spectrogram) + 1e-10)

        return spectrogram, label


    """Show image for visual inspection
    """
    def __init__(
        self, 
        verbose: bool = True,
        log_level: int = logging.INFO,
        name: str = "Image"
        ) -> None:
        """
        Args:
            verbose (bool): Whether to log label
            log_level (int): Logging level (default: logging.INFO)
            name (str): Name of window to show image
        """
        super(ImageShowCV2, self).__init__(log_level=log_level)
        self.verbose = verbose
        self.name = name
        self.thread_started = False

    def init_thread(self):
        if not self.thread_started:
            self.thread_started = True
            self.image_queue = queue.Queue()

            # Start a new thread to display the images, so that the main loop could run in multiple threads
            self.thread = threading.Thread(target=self._display_images)
            self.thread.start()

    def _display_images(self) -> None:
        """ Display images in a continuous loop """
        while True:
            image, label = self.image_queue.get()
            if isinstance(label, Image):
                cv2.imshow(self.name + "Label", label.numpy())
            cv2.imshow(self.name, image.numpy())
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def __call__(self, image: Image, label: typing.Any) -> typing.Tuple[Image, typing.Any]:
        """ Show image for visual inspection

        Args:
            data (np.ndarray): Image data
            label (np.ndarray): Label data
        
        Returns:
            data (np.ndarray): Image data
            label (np.ndarray): Label data (unchanged)
        """
        # Start cv2 image display thread
        self.init_thread()

        if self.verbose:
            if isinstance(label, (str, int, float)):
                self.logger.info(f"Label: {label}")

        if isinstance(label, Detections):
            for detection in label:
                img = detection.applyToFrame(np.asarray(image.numpy()))
                image.update(img)

        # Add image to display queue
        # Sleep if queue is not empty
        while not self.image_queue.empty():
            time.sleep(0.5)

        # Add image to display queue
        self.image_queue.put((image, label))

        return image, label