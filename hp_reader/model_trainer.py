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
    load_model_path = False,
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
    if load_model_path:
        deal_model.load_weights(load_model_path)
        print('loaded from: '+load_model_path)

    deal_model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(
            from_logits=True,
        ),
        metrics=[metrics.SparseCategoricalAccuracy()]
    )
    return deal_model

def image_dataset(dir_lists:list, max_digit:int, image_size, batch_size:int,
                  augment=False):
    """image_data
    Returns a Dataset
    Requires all image names to be integers
    Expects width ~1440

    arguments
    ---------
    dir_lists: list of directories that contain images

    max_digit: maximum digit of label (padded with 0)
               All labels should be less than max_digit
        ex) max_digit=3, label=23 -> [0 2 3]

    image_size: Image will be resized to this size
        (Height, Width)
    
    batch_size: int

    augment: Whether to add noise to the image
    """
    image_lists = [
        str(Path(d)/'*.png') for d in dir_lists
    ]
    dataset = tf.data.Dataset.list_files(image_lists)

    def process_path(image_path):
        image_raw = tf.io.read_file(image_path)
        image = tf.io.decode_png(image_raw,channels=3)
        width= tf.cast(tf.shape(image)[1],tf.float32)
        w_st = tf.cast(width*0.27,tf.int32)
        w_ed = tf.cast(width*0.70,tf.int32)
        image = image[:,w_st:w_ed,:]
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = tf.image.resize(image, image_size)
        if augment:
            # random invert color
            if tf.random.uniform([]) < 0.5:
                image = 1.0 - image
            # random shuffle rgb
            if tf.random.uniform([]) < 0.5:
                image = tf.gather(
                    image,
                    tf.random.shuffle([0,1,2]),
                    axis=-1,
                )
            # Random quality
            image = tf.image.random_jpeg_quality(image, 1, 50)

        image = image * 255

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

def random_hp_dataset(base_img_dir:str, max_digit:int, image_size, batch_size:int):
    """image_data
    Returns a Dataset
    Requires all image names to be integers

    arguments
    ---------
    dir_lists: list of directories that contain base images

    max_digit: maximum digit of label (padded with 0)
               All labels should be less than max_digit
        ex) max_digit=3, label=23 -> [0 2 3]

    image_size: Image will be resized to this size
        (Height, Width)
    """
    from PIL import Image, ImageFont, ImageDraw
    import random

    class HpGenerator():
        """An iterable generator that makes fake hp bar images

        Max_HP and Current_HP are both random while
        Current_HP <= Max_HP
        """
        def __init__(self,base_img_dir:str, max_digit:int) -> None:
            self.base_img_list = [
                Image.open(d) for d in Path(base_img_dir).iterdir()
                    if d.match('*.png')
            ] # List of PIL.Image objects, not np.array objects
            self.fonts = [
                ImageFont.truetype('NanumBarunGothic.ttf',size=s)
                    for s in range(28,36)
            ]
            assert max_digit > 0
            self.max_digit = max_digit

        def __iter__(self):
            return self

        def __call__(self, *args):
            return self

        def __next__(self):
            new_img = random.choice(self.base_img_list).copy()
            # Current_hp is more important - choose current_hp first
            # Uniform digit
            target_digit = random.randrange(1,max_digit+1)
            current_hp = random.randrange(10**(target_digit-1),
                                          10**target_digit-1)
            max_hp = random.randrange(current_hp, 10**max_digit)
            hp_text = f'{current_hp}/{max_hp}'
            draw = ImageDraw.Draw(new_img)
            font = random.choice(self.fonts)
            draw.text(
                xy=(new_img.width//2+random.randrange(-50,51),
                    new_img.height//2+random.randrange(0,16)),
                text=hp_text,
                fill=(255,255,255),
                font=font,
                anchor='mm'
            )

            x = np.array(new_img.convert('RGB'))
            y = current_hp

            return x, y
    
    dataset = tf.data.Dataset.from_generator(
        HpGenerator(base_img_dir, max_digit),
        output_signature=(
            tf.TensorSpec(shape=[None,None,3], dtype=tf.uint8),
            tf.TensorSpec(shape=[], dtype=tf.int64)
        )
    )

    def image_aug(image, raw_label):
        width= tf.cast(tf.shape(image)[1],tf.float32)
        w_st = tf.cast(width*0.27,tf.int32)
        w_ed = tf.cast(width*0.70,tf.int32)
        image = image[:,w_st:w_ed,:]
        image = tf.image.convert_image_dtype(image,tf.float32)
        image = tf.image.resize(image, image_size)
        # random invert color
        if tf.random.uniform([]) < 0.5:
            image = 1.0 - image
        # random shuffle rgb
        if tf.random.uniform([]) < 0.5:
            image = tf.gather(
                image,
                tf.random.shuffle([0,1,2]),
                axis=-1,
            )
        # Random quality
        image = tf.image.random_jpeg_quality(image, 1, 50)

        image = image * 255

        i = raw_label
        tf.debugging.assert_less(i,10**max_digit,
            message='Label is larger than max digit')
        d = tf.range(max_digit,0,-1,dtype=tf.int64)
        label = (i-(i//(10**d))*10**d)//(10**(d-1))

        return image, label

    dataset = dataset.map(image_aug)
    dataset = dataset.shuffle(SHUFFLE_BUFFER,reshuffle_each_iteration=False)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    dataset = dataset.repeat()

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
            ax.imshow(sample_x[i]/255)
        
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
            ax.imshow(sample_x[i]/255)
        
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
    lr_function,
    epochs,
    batch_size,
    train_dir,
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
        optimizer='adam',
        input_shape=input_shape,
        max_digits=max_digits,
        load_model_path=load_model_path,
    ) # Will get a compiled model


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
        monitor='sparse_categorical_accuracy',
        save_weights_only=True,
        verbose=1,
        save_best_only=True,
        mode='max',
    )
    tqdm_callback = tfa.callbacks.TQDMProgressBar()

    lr_callback = keras.callbacks.LearningRateScheduler(lr_function, verbose=1)
    
    train_ds = random_hp_dataset(
        train_dir,
        max_digits,
        img_size,
        batch_size,
    )
    val_ds = image_dataset(
        val_dir_list,
        max_digits,
        img_size,
        batch_size,
        augment=False,
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
            lr_callback,
        ],
        steps_per_epoch=1000,
        verbose=0,
        validation_data=val_ds,
    )

if __name__ == '__main__':
    import argparse
    from deal_models import *
    # from deal_optimizers import *
    from deal_lr_callbacks import *

    parser = argparse.ArgumentParser()
    parser.add_argument('-n','--name',dest='name')
    parser.add_argument('-e','-epochs', dest='epochs')
    parser.add_argument('--load',dest='load',default=False)
    parser.add_argument('-pf','--profile', dest='profile',
                        action='store_true',default=False)
    args = parser.parse_args()


    train_dir = 'videos/base'
    val_dirs = [
        'videos/abrel_6_4k/1',
        'videos/abrel_6_4k/2',
        'videos/abrel_6_4k/3',
        'videos/abrel_4_1080/1',
    ]

    kwargs = {}

    kwargs['name'] = args.name
    kwargs['model_function'] = effb7_lstm
    kwargs['lr_function'] = sqrt_decay_1
    kwargs['epochs'] = int(args.epochs)
    kwargs['batch_size'] = 32
    kwargs['train_dir'] = train_dir
    kwargs['val_dir_list'] = val_dirs
    kwargs['img_size'] = (64,640)
    kwargs['max_digits'] = 11
    kwargs['load_model_path'] = args.load
    kwargs['profile'] = args.profile
    
    run_training(**kwargs)
