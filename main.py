import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib


def download_dataset(dataset_url):
    """
    Purpose: download dataset from online data store
    """
    data_dir = tf.keras.utils.get_file(
        'flower_photos', origin=dataset_url, untar=True)
    return pathlib.Path(data_dir)


def create_dataset(data_dir, batch_size, img_width, img_height):
    """
    Purpose: Load data off disk using Kera Utility
    """
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=0.2,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size)

    return train_ds, val_ds
# end def


def visualize_data(train_ds):
    """
    Purpose: Visualize images the training dataset
    """
    plt.figure(figsize=(10, 10))
    class_names = train_ds.class_names
    for images, labels in train_ds.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis('off')
        # end for
    # end for

    plt.show()
# end def


data_dir = download_dataset(
    "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz")
# image_count = len(list(data_dir.glob('*/*.jpg')))
# print("Total image count: ", image_count)

train_ds, val_ds = create_dataset(data_dir, 32, 180, 180)
visualize_data(train_ds)
