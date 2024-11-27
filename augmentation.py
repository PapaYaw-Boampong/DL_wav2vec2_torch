import typing
import numpy as np
import importlib
import logging

from . import Image
from mltu.annotations.audio import Audio

def randomness_decorator(func):
    """ Decorator for randomness """
    def wrapper(self, data: typing.Union[Image, Audio], annotation: typing.Any) -> typing.Tuple[typing.Union[Image, Audio], typing.Any]:
        """ Decorator for randomness and type checking

        Args:
            data (typing.Union[Image, Audio]): Image or Audio object to be adjusted
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            data (typing.Union[Image, Audio]): Adjusted image or audio
            annotation (typing.Any): Adjusted annotation
        """
        # check if image is Image object
        if not isinstance(data, (Image, Audio)):
            self.logger.error(f"data must be Image or Audio object, not {type(data)}, skipping augmentor")
            # TODO instead of error convert image into Image object
            # TODO instead of error convert audio into Audio object
            return data, annotation

        if np.random.rand() > self._random_chance:
            return data, annotation

        # return result of function
        return func(self, data, annotation)

    return wrapper

class Augmentor:
    """ Object that should be inherited by all augmentors

    Args:
        random_chance (float, optional): Chance of applying the augmentor. Where 0.0 is never and 1.0 is always. Defaults to 0.5.
        log_level (int, optional): Log level for the augmentor. Defaults to logging.INFO.
    """
    def __init__(self, random_chance: float=0.5, log_level: int = logging.INFO, augment_annotation: bool = False) -> None:
        self._random_chance = random_chance
        self._log_level = log_level
        self._augment_annotation = augment_annotation

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        assert 0 <= self._random_chance <= 1.0, "random chance must be between 0.0 and 1.0"

    def augment(self, data: typing.Union[Image, Audio]):
        """ Augment data """
        raise NotImplementedError

    @randomness_decorator
    def __call__(self, data: typing.Union[Image, Audio], annotation: typing.Any) -> typing.Tuple[typing.Union[Image, Audio], typing.Any]:
        """ Randomly add noise to audio

        Args:
            data (typing.Union[Image, Audio]): Image or Audio object to be adjusted
            annotation (typing.Any): Annotation to be adjusted

        Returns:
            data (typing.Union[Image, Audio]): Adjusted image or audio
            annotation (typing.Any): Adjusted annotation if necessary
        """
        data = self.augment(data)

        if self._augment_annotation and isinstance(annotation, np.ndarray):
            annotation = self.augment(annotation)

        return data, annotation

class RandomAudioNoise(Augmentor):
    """ Randomly add noise to audio

    Attributes:
        random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
        log_level (int): Log level for the augmentor. Defaults to logging.INFO.
        augment_annotation (bool): Whether to augment the annotation. Defaults to False.
        max_noise_ratio (float): Maximum noise ratio to be added to audio. Defaults to 0.1.
    """

    def __init__(
            self,
            random_chance: float = 0.5,
            log_level: int = logging.INFO,
            augment_annotation: bool = False,
            max_noise_ratio: float = 0.1,
    ) -> None:
        super(RandomAudioNoise, self).__init__(random_chance, log_level, augment_annotation)
        self.max_noise_ratio = max_noise_ratio

    def augment(self, audio: Audio) -> Audio:
        noise = np.random.uniform(-1, 1, len(audio))
        noise_ratio = np.random.uniform(0, self.max_noise_ratio)
        audio_noisy = audio + noise_ratio * noise

        return audio_noisy

class RandomAudioPitchShift(Augmentor):
    """ Randomly add noise to audio

    Attributes:
        random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
        log_level (int): Log level for the augmentor. Defaults to logging.INFO.
        augment_annotation (bool): Whether to augment the annotation. Defaults to False.
        max_n_steps (int): Maximum number of steps to shift audio. Defaults to 5.
    """

    def __init__(
            self,
            random_chance: float = 0.5,
            log_level: int = logging.INFO,
            augment_annotation: bool = False,
            max_n_steps: int = 5,
    ) -> None:
        super(RandomAudioPitchShift, self).__init__(random_chance, log_level, augment_annotation)
        self.max_n_steps = max_n_steps

        # import librosa using importlib
        try:
            self.librosa = importlib.import_module('librosa')
            print("librosa version:", self.librosa.__version__)
        except ImportError:
            raise ImportError("librosa is required to augment Audio. Please install it with `pip install librosa`.")

    def augment(self, audio: Audio) -> Audio:
        random_n_steps = np.random.randint(-self.max_n_steps, self.max_n_steps)
        # changing default res_type "kaiser_best" to "linear" for speed and memory efficiency
        shift_audio = self.librosa.effects.pitch_shift(
            audio.numpy(), sr=audio.sample_rate, n_steps=random_n_steps, res_type="linear"
        )
        audio.audio = shift_audio

        return audio


class RandomAudioTimeStretch(Augmentor):
    """ Randomly add noise to audio

    Attributes:
        random_chance (float): Float between 0.0 and 1.0 setting bounds for random probability. Defaults to 0.5.
        log_level (int): Log level for the augmentor. Defaults to logging.INFO.
        augment_annotation (bool): Whether to augment the annotation. Defaults to False.
        min_rate (float): Minimum rate to stretch audio. Defaults to 0.8.
        max_rate (float): Maximum rate to stretch audio. Defaults to 1.2.
    """

    def __init__(
            self,
            random_chance: float = 0.5,
            log_level: int = logging.INFO,
            augment_annotation: bool = False,
            min_rate: float = 0.8,
            max_rate: float = 1.2
    ) -> None:
        super(RandomAudioTimeStretch, self).__init__(random_chance, log_level, augment_annotation)
        self.min_rate = min_rate
        self.max_rate = max_rate

        try:
            librosa.__version__
        except ImportError:
            raise ImportError("librosa is required to augment Audio. Please install it with `pip install librosa`.")

    def augment(self, audio: Audio) -> Audio:
        random_rate = np.random.uniform(self.min_rate, self.max_rate)
        stretch_audio = librosa.effects.time_stretch(audio.numpy(), rate=random_rate)
        audio.audio = stretch_audio

        return audio