"""Utility functions for working with images."""

import numpy as np
from PIL import Image


class ProcessedImage:

    """A processed image object."""

    def __init__(self, original, resized, array):
        """Initialize the image object"""

        self.original = original
        self.resized = resized
        self.array = array


def read_image(image_path):
    """Read an image from a file."""
    return Image.open(image_path)


def resize_image(image, width, height):
    """Resize an image to the given dimensions."""
    return image.resize((width, height), Image.BILINEAR)


def img_to_array(image):
    """Convert an image to a NumPy array."""
    return np.array(image)
