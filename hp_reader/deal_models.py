import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, applications

def mobv3_lstm(input_shape, digits:int):
    mobv3 = applications.MobileNetV3Large(
        input_shape=input_shape,
        weights=None,
        include_top=True,
        classes=1024,
    )
    
