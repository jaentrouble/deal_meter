import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Model, mixed_precision
import tensorflow_addons as tfa

def deal_model():
    # temporary
    return keras.Model()

def run_training(
    name,
    model_function,
    optimizer,
    epochs,
    batch_size,
    train_dir_list,
    val_dir_list,
    img_size,
    load_model_path=None,
    profile = False,
):
    mixed_precision.set_global_policy('mixed_float16')
    
    #TODO: fill deal_model arguments
    mymodel = deal_model() # Will get a compiled model

    if load_model_path:
        mymodel.load_weights(load_model_path)
        print('loaded from: '+load_model_path)

    mymodel.summary()
    log_dir = 'logs/fit/' + name
    if profile:
        tb_callback = keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_images=True,
            profile_batch=(3,5),
        )
    else:
        tb_callback = keras.callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_images=True,
        )

    savedir = 'savedmodels/' + name + '/{epoch}'
    save_callback = keras.callbacks.ModelCheckpoint(
        savedir,
        save_weights_only=True,
        verbose=1
    )
    tqdm_callback = tfa.callbacks.TQDMProgressBar()