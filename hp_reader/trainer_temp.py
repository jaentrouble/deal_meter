import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, Model, mixed_precision

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
    img_size
):
    mixed_precision.set_global_policy('mixed_float16')
    
    #TODO: fill deal_model arguments
    mymodel = deal_model() # Will get a compiled model

    