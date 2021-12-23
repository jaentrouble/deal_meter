import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, applications

# All functions should take 'input tensor' and 'digits' as arguments

def mobv3_lstm(inputs, digits:int):
    # No Softmax activation
    mobv3_output = applications.MobileNetV3Large(
        input_tensor=inputs,
        weights=None,
        include_top=True,
        classes=1024,
        minimalistic=True,
        classifier_activation=None,
    ).output
    tiled_output = tf.tile(tf.expand_dims(
        mobv3_output,axis=1,
    ),[1,digits,1])
    lstm_output = layers.LSTM(
        units=1024,
        return_sequences=True,
    )(tiled_output)
    logit_output = layers.Dense(
        units=10,
    )(lstm_output)
    return logit_output
    

def effb7_lstm(inputs, digits:int):
    # No Softmax activation
    mobv3_output = applications.EfficientNetB7(
        input_tensor=inputs,
        weights=None,
        include_top=True,
        classes=1024,
        classifier_activation=None,
    ).output
    tiled_output = tf.tile(tf.expand_dims(
        mobv3_output,axis=1,
    ),[1,digits,1])
    lstm_output = layers.LSTM(
        units=1024,
        return_sequences=True,
    )(tiled_output)
    logit_output = layers.Dense(
        units=10,
    )(lstm_output)
    return logit_output
    

