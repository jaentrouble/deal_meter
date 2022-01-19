
if __name__ == '__main__':
    import tensorflow as tf
    from model_trainer import deal_model
    from deal_models import *

    MODEL_FUNCTION = effb7_lstm
    INPUT_SHAPE = (64,640,3)
    MAX_DIGITS = 11
    WEIGHT_PATH = 'savedmodels/stroke_1/best'
    SAVE_PATH = 'savedmodels/final'

    inputs = keras.Input(shape=INPUT_SHAPE)
    outputs = MODEL_FUNCTION(inputs, MAX_DIGITS)
    deal_model = keras.Model(inputs=inputs,outputs=outputs)
    deal_model.load_weights(WEIGHT_PATH)
    deal_model.save(SAVE_PATH)