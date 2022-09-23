

# FCN model
# when tuning start with learning rate->mini_batch_size ->
# momentum-> #hidden_units -> # learning_rate_decay -> #layers
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np
import time

from utils.utils import save_logs
from utils.utils import calculate_metrics

class Classifier_LSTM:

    def __init__(self, output_directory, input_shape, nb_classes, verbose=False,build=True):
        self.output_directory = output_directory

        if build == True:
            self.model = self.build_model(input_shape, nb_classes)
            if (verbose == True):
                self.model.summary()
            self.verbose = verbose
            self.model.save_weights(self.output_directory + 'model_init.hdf5')

        return

    def build_model(self, input_shape, nb_classes):
        ip = keras.layers.Input(input_shape)

        x = keras.layers.Masking()(ip)
        x = keras.layers.LSTM(128)(x)
        x = keras.layers.Dropout(0.8)(x)

        y = keras.layers.Permute((2, 1))(ip)
        y = keras.layers.Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation('relu')(y)
        #y = squeeze_excite_block(y)

        y = keras.layers.Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation('relu')(y)
        #y = squeeze_excite_block(y)

        y = keras.layers.Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(y)
        y = keras.layers.BatchNormalization()(y)
        y = keras.layers.Activation('relu')(y)

        y = keras.layers.GlobalAveragePooling1D()(y)

        x = keras.layers.concatenate([x, y])

        out = keras.layers.Dense(nb_classes, activation='softmax')(x)

        model = keras.models.Model(ip, out)
        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                          metrics=['accuracy'])

        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                               save_best_only=True)

        self.callbacks = [reduce_lr, model_checkpoint]

        return model

     



    def build_model_example_NOT_USE(self, input_shape, nb_classes):
        padding = 'valid'
        input_layer = keras.layers.Input(input_shape)

        if input_shape[0] < 60: # for italypowerondemand dataset
            padding = 'same'

        conv1 = keras.layers.Conv1D(filters=6,kernel_size=7,padding=padding,activation='sigmoid')(input_layer)
        conv1 = keras.layers.AveragePooling1D(pool_size=3)(conv1)

        conv2 = keras.layers.Conv1D(filters=12,kernel_size=7,padding=padding,activation='sigmoid')(conv1)
        conv2 = keras.layers.AveragePooling1D(pool_size=3)(conv2)

        flatten_layer = keras.layers.Flatten()(conv2)

        output_layer = keras.layers.Dense(units=nb_classes,activation='sigmoid')(flatten_layer)

        model = keras.models.Model(inputs=input_layer, outputs=output_layer)

        model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(),
                      metrics=['accuracy'])

        file_path = self.output_directory + 'best_model.hdf5'

        model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss',
                                                           save_best_only=True)

        self.callbacks = [model_checkpoint]

        return model

    def fit(self, x_train, y_train):
        if not tf.test.is_gpu_available:
            print('error')
            exit()
       
        batch_size = 128
        nb_epochs = 1000

        mini_batch_size = int(min(x_train.shape[0] / 10, batch_size))

        start_time = time.time()

        self.model.fit(x_train, y_train, batch_size=mini_batch_size, epochs=nb_epochs,
                              verbose=self.verbose, callbacks=self.callbacks)

        duration = time.time() - start_time

    def predict(self, x_test):
        model_path = self.output_directory + 'best_model.hdf5'
        model = keras.models.load_model(model_path)
        y_pred = model.predict(x_test)
        
        return np.argmax(y_pred, axis = 1)

    def evaluate(self, x_test, y_test):
        loss, accuracy = self.model.evaluate(x_test,
                                         y_test,verbose = True)
        return accuracy


