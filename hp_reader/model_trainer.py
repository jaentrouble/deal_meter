import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, losses, metrics, mixed_precision
from pathlib import Path
import os
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
import io
import numpy as np

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
        str(Path(d)/'*.png') for d in dir_lists
    ]
    dataset = tf.data.Dataset.list_files(image_lists)

    def process_path(image_path):
        image_raw = tf.io.read_file(image_path)
        image = tf.io.decode_png(image_raw,channels=3)
        crop_size = tf.shape(image) - tf.constant([0, 150, 0],dtype=tf.int32)
        image = tf.image.random_crop(image,crop_size)
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = tf.image.resize(image, image_size)
        raw_label = tf.strings.split(
            tf.strings.split(image_path,os.sep)[-1],'.'
        )[0]
        i = tf.strings.to_number(raw_label,out_type=tf.int64)
        tf.debugging.assert_less(i,10**max_digit,
            message='Label is larger than max digit')
        d = tf.range(max_digit,0,-1,dtype=tf.int64)
        label = (i-(i//(10**d))*10**d)//(10**(d-1))

        return image, label

    dataset = dataset.map(process_path)
    dataset = dataset.shuffle(SHUFFLE_BUFFER,reshuffle_each_iteration=True)
    dataset = dataset.batch(batch_size)

    return dataset

class ValFigCallback(keras.callbacks.Callback):
    def __init__(self, train_ds, val_ds, logdir):
        super().__init__()
        self.val_ds = val_ds
        self.train_ds = train_ds
        self.filewriter = tf.summary.create_file_writer(logdir+'/val_image')

    def plot_to_image(self, figure):
        """Converts the matplotlib plot specified by 'figure' to a PNG image and
        returns it. The supplied figure is closed and inaccessible after this call."""
        # Save the plot to a PNG in memory.
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        # Closing the figure prevents it from being displayed directly inside
        # the notebook.
        plt.close(figure)
        buf.seek(0)
        # Convert PNG buffer to TF image
        image = tf.image.decode_png(buf.getvalue(), channels=4)
        # Add the batch dimension
        image = tf.expand_dims(image, 0)
        return image

    def val_result_fig(self):
        samples = self.val_ds.take(1).as_numpy_iterator()
        sample = next(samples)
        fig = plt.figure(figsize=(15,15))
        sample_x = sample[0]
        logits = self.model(sample_x, training=False)
        predict = np.argmax(logits,axis=-1)
        self.next_text = str(logits[:4])
        for i in range(4):
            ax = fig.add_subplot(4,1,i+1,title=str(predict[i]))
            ax.imshow(sample_x[i])
        
        return fig

    def train_result_fig(self):
        samples = self.train_ds.take(1).as_numpy_iterator()
        sample = next(samples)
        fig = plt.figure(figsize=(15,15))
        sample_x = sample[0]
        logits = self.model(sample_x, training=False)
        predict = np.argmax(logits,axis=-1)
        for i in range(4):
            ax = fig.add_subplot(4,1,i+1,title=str(predict[i]))
            ax.imshow(sample_x[i])
        
        return fig


    def on_epoch_end(self, epoch, logs=None):
        train_image = self.plot_to_image(self.train_result_fig())
        val_image = self.plot_to_image(self.val_result_fig())
        with self.filewriter.as_default():
            tf.summary.image('train prediction', train_image, step=epoch)
            tf.summary.image('val prediction', val_image, step=epoch)
            tf.summary.text('logits',self.next_text,step=epoch)



def run_training(
    name,
    model_function,
    optimizer,
    epochs,
    batch_size,
    train_dir_list,
    val_dir_list,
    img_size,
    max_digits,
    load_model_path=None,
    profile = False,
):
    """
    img_size: (height, width)
    """
    mixed_precision.set_global_policy('mixed_float16')
    
    input_shape = (img_size[0],img_size[1],3)
    mymodel = deal_model(
        model_function=model_function,
        optimizer=optimizer,
        input_shape=input_shape,
        max_digits=max_digits,
    ) # Will get a compiled model

    if load_model_path:
        mymodel.load_weights(load_model_path)
        print('loaded from: '+load_model_path)

    mymodel.summary()
    log_dir = 'logs/fit/' + name
    if profile:
        tb_callback = keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            profile_batch=(3,5),
        )
    else:
        tb_callback = keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
        )

    savedir = 'savedmodels/' + name + '/{epoch}'
    save_callback = keras.callbacks.ModelCheckpoint(
        savedir,
        monitor='val_sparse_categorical_accuracy',
        save_weights_only=True,
        verbose=1,
        save_best_only=True,
        mode='max',
    )
    tqdm_callback = tfa.callbacks.TQDMProgressBar()
    
    train_ds = image_dataset(
        train_dir_list,
        max_digits,
        img_size,
        batch_size,
    )
    val_ds = image_dataset(
        val_dir_list,
        max_digits,
        img_size,
        batch_size,
    )

    image_callback = ValFigCallback(train_ds, val_ds, log_dir)

    mymodel.fit(
        x=train_ds,
        epochs=epochs,
        callbacks=[
            tb_callback,
            save_callback,
            tqdm_callback,
            image_callback,
        ],
        verbose=0,
        validation_data=val_ds,
    )

if __name__ == '__main__':
    import argparse
    from deal_models import *
    from deal_optimizers import *

    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--name',dest='name')
    parser.add_argument('-e','-epochs', dest='epochs')
    parser.add_argument('--load',dest='load',default=False)
    parser.add_argument('-pf','--profile', dest='profile',
                        action='store_true',default=False)
    args = parser.parse_args()


    train_dirs = [
        'videos/1',
        'videos/2',
        'videos/4',
        'videos/5',
        'videos/6',
        'videos/7',
        'videos/8',
        'videos/9',
        'videos/10',
    ]
    val_dirs = [
        'videos/3'
    ]

    kwargs = {}

    kwargs['name'] = args.name
    kwargs['model_function'] = mobv3_lstm
    kwargs['optimizer'] = exp_4_6
    kwargs['epochs'] = int(args.epochs)
    kwargs['batch_size'] = 128
    kwargs['train_dir_list'] = train_dirs
    kwargs['val_dir_list'] = val_dirs
    kwargs['img_size'] = (128,704)
    kwargs['max_digits'] = 11
    kwargs['load_model_path'] = args.load
    kwargs['profile'] = args.profile
    
    run_training(**kwargs)
