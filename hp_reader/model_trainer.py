import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, losses, metrics
from .deal_models import *
from .lr_schedules import *
from pathlib import Path
import os

SHUFFLE_BUFFER = 1000

def deal_model(
    model_function,
    optimizer,
    input_shape,
    max_digits,
):
    """deal_model
    Gets an image of hp-bar and returns current hp

    Arguments
    ---------
    model_function: A function that takes keras.Input and returns LOGIT of each digits
                    The output shape is (batch, digits, 10)

    optimizer: keras.optimizer.Optimizer object

    input_shape: (height, width, 3)

    max_digits: int
    """
    inputs = keras.Input(shape=input_shape)
    outputs = model_function(inputs, max_digits)

    deal_model= keras.Model(inputs=inputs,outputs=outputs)
    deal_model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(
            from_logits=True,
        ),
        metrics=[metrics.SparseCategoricalAccuracy()]
    )
    return deal_model

def image_dataset(dir_lists:list, max_digit:int, image_size, batch_size:int):
    """image_data
    Returns a Dataset
    Requires all image names to be integers

    arguments
    ---------
    dir_lists: list of directories that contain images

    max_digit: maximum digit of label (padded with 0)
               All labels should be less than max_digit
        ex) max_digit=3, label=23 -> [0 2 3]

    image_size: Image will be resized to this size
        (Height, Width)
    """
    image_lists = [
        str(Path(d)/'*.png' for d in dir_lists)
    ]
    dataset = tf.data.Dataset.list_files(image_lists)

    def process_path(image_path):
        image_raw = tf.io.read_file(image_path)
        image = tf.io.decode_image(image_raw,channels=3)
        image = tf.image.resize(image, image_size)
        raw_label = tf.strings.split(
            tf.strings.split(image_path,os.sep)[-1],'.'
        )[0]
        i = tf.strings.to_number(raw_label,out_type=tf.int32)
        tf.debugging.assert_less(i,10**max_digit,
            message='Label is larger than max digit')
        d = tf.range(max_digit,0,-1)
        label = (i-(i//(10**d))*10**d)//(10**(d-1))
        
        return image, label

    dataset = dataset.map(process_path)
    dataset = dataset.shuffle(SHUFFLE_BUFFER,reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)

    return dataset


def run_training(
    model_function,
    optimizer,
    train_dir_lists,
    val_dir_lists,
    img_size,
    max_digits,
):
    