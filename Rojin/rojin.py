'''

rojin.py


'''




import os
import numpy as np

import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Input, Dense
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D, Reshape, BatchNormalization
from tensorflow.python.keras.layers import Activation, Dropout, Flatten, Cropping2D, Lambda
from tensorflow.python.keras.layers.merge import concatenate
from tensorflow.python.keras.layers import LSTM
from tensorflow.python.keras.layers.wrappers import TimeDistributed as TD
from tensorflow.python.keras.layers import Conv3D, MaxPooling3D, Cropping3D, Conv2DTranspose

import donkeycar as dk


if tf.__version__ == '1.13.1':
    from tensorflow import ConfigProto, Session

    # Override keras session to work around a bug in TF 1.13.1
    # Remove after we upgrade to TF 1.14 / TF 2.x.
    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = Session(config=config)
    keras.backend.set_session(session)


class KerasPilot(object):
    '''
    Base class for Keras models that will provide steering and throttle to guide a car. #Sprich Steuerung und die Geschwindigkeiten
    '''
    def __init__(self):
        self.model = None
        self.optimizer = "adam"

    def load(self, model_path):
        self.model = keras.models.load_model(model_path, compile=False)

    def load_weights(self, model_path, by_name=True):
        self.model.load_weights(model_path, by_name=by_name)

    def shutdown(self):
        pass

    def compile(self):
        pass

    def set_optimizer(self, optimizer_type, rate, decay):
        if optimizer_type == "adam":
            self.model.optimizer = keras.optimizers.Adam(lr=rate, decay=decay)
        elif optimizer_type == "sgd":
            self.model.optimizer = keras.optimizers.SGD(lr=rate, decay=decay)
        elif optimizer_type == "rmsprop":
            self.model.optimizer = keras.optimizers.RMSprop(lr=rate, decay=decay)
        else:
            raise Exception("unknown optimizer type: %s" % optimizer_type)

    def train(self, train_gen, val_gen,
              saved_model_path, epochs=100, steps=100, train_split=0.8,
              verbose=1, min_delta=.0005, patience=5, use_early_stop=True):

        """
        train_gen: generator that yields an array of images an array of

        """

        #checkpoint to save model after each epoch
        save_best = keras.callbacks.ModelCheckpoint(saved_model_path,
                                                    monitor='val_loss',
                                                    verbose=verbose,
                                                    save_best_only=True,
                                                    mode='min')

        #stop training if the validation error stops improving.
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss',
                                                   min_delta=min_delta,
                                                   patience=patience,
                                                   verbose=verbose,
                                                   mode='auto')

        callbacks_list = [save_best]

        if use_early_stop:
            callbacks_list.append(early_stop)

        hist = self.model.fit_generator(
                        train_gen,
                        steps_per_epoch=steps,
                        epochs=epochs,
                        verbose=1,
                        validation_data=val_gen,
                        callbacks=callbacks_list,
                        validation_steps=steps*(1.0 - train_split))
        return hist


def adjust_input_shape(input_shape, roi_crop):
    height = input_shape[0]
    new_height = height - roi_crop[0] - roi_crop[1]
    return (new_height, input_shape[1], input_shape[2])



class RojinModel(KerasPilot):
    def __init__(self, num_outputs=2, input_shape=(120, 160, 3), roi_crop=(0, 0), *args, **kwargs):
        super(RojinModel, self).__init__(*args, **kwargs)
        self.model = RojinLinear(num_outputs, input_shape, roi_crop)
        self.compile()

    # Set adam as optimizer and mean squared error as the error function
    def compile(self):
        self.model.compile(optimizer="adam",
                loss='mse')

    def run(self, img_arr):
        img_arr = img_arr.reshape((1,) + img_arr.shape)
        outputs = self.model.predict(img_arr)
        steering = outputs[0]
        throttle = outputs[1]
        return steering[0][0], throttle[0][0]


def RojinLinear(num_outputs, input_shape, roi_crop):

    input_shape = adjust_input_shape(input_shape, roi_crop)
    img_in = Input(shape=input_shape, name='img_in')
    x = img_in

    # Dropout rate
    keep_prob = 0.9
    rate = 1 - keep_prob


    # Convolutional Layer
    x = Convolution2D(filters=16, kernel_size=5, strides=(2, 2), input_shape = input_shape)(x)
    x = Dropout(rate)(x)
    x = Convolution2D(filters=16, kernel_size=5, strides=(2, 2), activation='relu')(x)
    x = Dropout(rate)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(x)
    x = Convolution2D(filters=32, kernel_size=5, strides=(2, 2), activation='relu')(x)
    x = Dropout(rate)(x)
    x = Convolution2D(filters=32, kernel_size=3, strides=(1, 1), activation='relu')(x)
    x = Dropout(rate)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='valid')(x)
    x = Convolution2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu')(x)
    x = Dropout(rate)(x)

    # Flatten Layers [Flatten to 1D (Fully connected))]
    x = Flatten()(x)

    # Fully Connected Layer
    x = Dense(100, activation='relu')(x)
    x = Dense(50, activation='relu')(x)
    x = Dense(25, activation='relu')(x)
    x = Dense(10, activation='relu')(x)
    x = Dense(5, activation='relu')(x)

    outputs = []


    #categorical output of the angle
    #angle_out = Dense(15, activation='softmax', name='angle_out')(x)        # Connect every input with every output and output 15 hidden units. Use Softmax to give percentage. 15 categories and find best one based off percentage 0.0-1.0

    #continous output of throttle
    #throttle_out = Dense(20, activation='softmax', name='throttle_out')(x)      # Reduce to 1 number, Positive number only


    for i in range(num_outputs):
        # Output layer
        outputs.append(Dense(1, activation='linear', name='n_outputs' + str(i))(x))

    model = Model(inputs=[img_in], outputs=outputs)


    return model
