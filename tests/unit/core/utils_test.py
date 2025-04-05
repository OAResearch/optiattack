import os
import time
import numpy as np
import pytest
from PIL import Image

from core.utils.images import read_image, resize_image, img_to_array
from core.utils.incremental_average import IncrementalAverage


def test_image_read():
    current_path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_path, "../../test_img.jpeg")
    image = read_image(image_path)
    assert image is not None
    assert isinstance(image, Image.Image)

def test_image_resize():
    current_path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_path, "../../test_img.jpeg")
    image = read_image(image_path)
    resized_image = resize_image(image, 100, 100)
    assert resized_image is not None
    assert isinstance(resized_image, Image.Image)
    assert resized_image.size == (100, 100)

def test_image_img_to_array():
    current_path = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(current_path, "../../test_img.jpeg")
    image = read_image(image_path)
    resized_image = resize_image(image, 100, 100)
    image_array = img_to_array(resized_image)
    assert image_array is not None
    assert isinstance(image_array, np.ndarray)
    assert image_array.shape == (100, 100, 3)
    assert image_array.dtype == np.uint8


def test_incremental_average():
    avg = IncrementalAverage()

    # Test adding values
    avg.add_value(10)
    avg.add_value(20)
    avg.add_value(30)

    assert avg.mean == 20.0
    assert avg.min == 10.0
    assert avg.max == 30.0
    assert avg.n == 3

    # Test string representation
    assert str(avg) == "Avg=20.00, min=10.00, max=30.00"

def test_timer_functionality():
    avg = IncrementalAverage()

    # Test timer
    avg.start_timer()
    time.sleep(0.1)  # Sleep for 100 ms
    elapsed = avg.add_elapsed_time()

    assert elapsed >= 100  # Elapsed time should be at least 100 ms
    assert avg.is_recording_timer() is False

    # Test adding elapsed time to the average
    assert avg.n == 1
    assert avg.mean == pytest.approx(elapsed, rel=1e-2)  # Allow small floating-point differences

def test_empty_average():
    avg = IncrementalAverage()

    assert avg.n == 0
    assert avg.mean == 0.0
    assert avg.min == 0.0
    assert avg.max == 0.0
    assert str(avg) == "Avg=0.00, min=0.00, max=0.00"

def not_test_add_value_after_timer():
    avg = IncrementalAverage()

    avg.start_timer()
    time.sleep(0.05)
    avg.add_elapsed_time()

    avg.add_value(50)
    assert avg.n == 2
    assert avg.mean == pytest.approx((avg.mean * 1 + 50) / 2, rel=1e-1)