import numpy as np
from PIL import Image

from core.utils.images import read_image, resize_image, img_to_array


def test_image_read():
    image_path = "../../test_img.jpeg"
    image = read_image(image_path)
    assert image is not None
    assert isinstance(image, Image.Image)

def test_image_resize():
    image_path = "../../test_img.jpeg"
    image = read_image(image_path)
    resized_image = resize_image(image, 100, 100)
    assert resized_image is not None
    assert isinstance(resized_image, Image.Image)
    assert resized_image.size == (100, 100)

def test_image_img_to_array():
    image_path = "../../test_img.jpeg"
    image = read_image(image_path)
    resized_image = resize_image(image, 100, 100)
    image_array = img_to_array(resized_image)
    assert image_array is not None
    assert isinstance(image_array, np.ndarray)
    assert image_array.shape == (100, 100, 3)
    assert image_array.dtype == np.uint8
