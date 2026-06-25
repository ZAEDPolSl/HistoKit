import random
import numpy as np
from abc import ABC, abstractmethod

class Transform(ABC):
    """
    Base class for image augmentations.

    Every augmentation that inherits from this class should implement
    the `apply` method, which contains the actual image augmentation logic.

    The augmentation is applied with probability `prob`. If the augmentation
    is not applied, the original image is returned.

    Parameters
    ----------
    prob : float, (default=0.5)
        Probability of applying the augmentation. The value should be in the
        range [0.0, 1.0].
    """
    def __init__(self, prob=0.5):
        """
        Initialize the augmentation.

        Parameters
        ----------
        prob : float, default=0.5
            Probability of applying the augmentation.
        """
        self.prob = prob

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply the augmentation to an image.

        The augmentation is applied only when a randomly generated value
        is lower than `prob`.

        Parameters
        ----------
        img : np.ndarray
            Input image represented as a NumPy array.

        Returns
        -------
        np.ndarray
            Augmented image, or the original image if the augmentation
            was not applied.
        """
        if random.random() < self.prob:
            return self.apply(img)
        return img

    @abstractmethod
    def apply(self, img: np.ndarray) -> np.ndarray:
        """
        Perform the augmentation.

        This abstract method must be implemented by all subclasses of
        `Transform`.

        Parameters
        ----------
        img : np.ndarray
            Input image represented as a NumPy array.

        Returns
        -------
        np.ndarray
            Augmented image.
        """
        pass

class OneOf:
    def __init__(self, transforms, prob=0.5):
        """
        Randomly select and apply one augmentation from a list.

        `OneOf` groups multiple transformations and applies at most one of them
        during a single call. The whole operation is executed with probability
        `prob`.

        Parameters
        ----------
        transforms : list
            List of augmentations to choose from. Each element should be callable,
            for example an instance of a class inheriting from `Transform`.
        prob : float, default=0.5
            Probability of applying one of the augmentations.
        """
        self.transforms = transforms
        self.prob = prob

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply one randomly selected augmentation to an image.

        If a randomly generated value is lower than `prob`, one augmentation
        is selected from `transforms` and applied to the image. Otherwise,
        the original image is returned unchanged.

        Parameters
        ----------
        img : np.ndarray
            Input image represented as a NumPy array.

        Returns
        -------
        np.ndarray
            Image after applying one randomly selected transformation, or the
            original image if no transformation was applied.
        """
        if random.random() < self.prob:
            transform = random.choice(self.transforms)
            return transform(img)

        return img


class Compose:
    """
    Combine multiple augmentations into a sequential pipeline.

    Augmentations are applied one after another in the order in which they
    are provided in `transforms`. The output of one augmentation becomes
    the input of the next augmentation.

    Parameters
    ----------
    transforms : list
        List of augmentations to apply sequentially. Each element should be
        callable and accept a NumPy array as input.
    """
    def __init__(self, transforms):
        """
        Initialize the augmentation pipeline.

        Parameters
        ----------
        transforms : list
            List of augmentations to apply in sequence.
        """
        self.transforms = transforms

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """
        Apply all augmentations sequentially to an image.

        Parameters
        ----------
        img : np.ndarray
            Input image represented as a NumPy array.

        Returns
        -------
        np.ndarray
            Image after applying all augmentations in sequence.
        """
        for transform in self.transforms:
            img = transform(img)

        return img