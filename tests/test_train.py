import os
import pytest
import tensorflow as tf
from src.preprocessing.preprocessing import preprocess_split_data

@pytest.fixture
def data_dir(tmp_path):
    # create a temporary directory and generate some test images
    current_dir = os.getcwd()
    dir_path = f"{current_dir}/tests/test_images"
    os.makedirs(dir_path, exist_ok=True)
    os.makedirs(f"{dir_path}/random_class", exist_ok=True)

    for i in range(10):
        img_path = f"{dir_path}/random_class/image_{i}.jpg"
        img = tf.random.uniform((100, 100, 3), maxval=255, dtype=tf.int32)
        tf.keras.preprocessing.image.save_img(str(img_path), img)
    return dir_path

def test_preprocess_split_data(data_dir):
    # define test parameters
    img_height = 50
    img_width = 50
    batch_size = 2
    validation_split = 0.2
    
    # call the function to preprocess and split the data
    train_ds, val_ds = preprocess_split_data(data_dir, img_height, img_width, batch_size, validation_split)
    
    # check that the datasets have the correct properties
    assert isinstance(train_ds, tf.data.Dataset)
    assert isinstance(val_ds, tf.data.Dataset)
    assert len(train_ds) == 4
    assert len(val_ds) == 1
