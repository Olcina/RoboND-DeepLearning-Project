import os
import glob
import sys
import tensorflow as tf
from tqdm import tqdm_notebook as tqdm
import logging
from scipy import misc
import numpy as np

from tensorflow.contrib.keras.python import keras
from tensorflow.contrib.keras.python.keras import layers, models

from tensorflow import image

from utils import scoring_utils
from utils.separable_conv2d import SeparableConv2DKeras, BilinearUpSampling2D
from utils import data_iterator
#from utils import plotting_tools
from utils import model_tools
#testing module
from utils.testing_tools import create_test, dump_test_case, load_test_cases

from aux_function import fcn_model, fcn_model2 , decoder_block, encoder_block,bilinear_upsample,conv2d_batchnorm,separable_conv2d_batchnorm
"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

image_hw = 160
image_shape = (image_hw, image_hw, 3)
inputs = layers.Input(image_shape)
num_classes = 3

# Call fcn_model()
output_layer = fcn_model(inputs, num_classes)
print('output layer size',output_layer.get_shape().as_list())
logging.info('modelo preparado')



"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
def train_model(test, inputs, output_layer):
    #load variables for the test
    learning_rate = test['learning_rate']
    batch_size = test['batch_size']
    steps_per_epoch = test['steps_per_epoch']
    num_epochs = test['num_epochs']
    validation_steps = test['validation_steps']
    workers = test['workers']

    # Define the Keras model and compile it for training
    model = models.Model(inputs=inputs, outputs=output_layer)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='categorical_crossentropy')

    # Data iterators for loading the training and validation data
    train_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                   data_folder=os.path.join('..', 'data', 'train'),
                                                   image_shape=image_shape,
                                                   shift_aug=True)

    val_iter = data_iterator.BatchIteratorSimple(batch_size=batch_size,
                                                 data_folder=os.path.join('..', 'data', 'validation'),
                                                 image_shape=image_shape)

    #logger_cb = plotting_tools.LoggerPlotter()
    #callbacks = [logger_cb]

    model.fit_generator(train_iter,
                        steps_per_epoch = steps_per_epoch, # the number of batches per epoch,
                        epochs = num_epochs, # the number of epochs to train for,
                        validation_data = val_iter, # validation iterator
                        validation_steps = validation_steps, # the number of batches to validate on

                        workers = workers)
    return model

tests_cases = load_test_cases('DFCN_tests.p')
# Save your trained model weights
import time
import pickle
#loop over all the test cases defined and save the training time
training_times = []
for test in tests_cases:
    print(test['name'])
    test_training_time = {}
    test_training_time['name'] = test['name']
    #start the clock
    test_training_time['start'] = time.time()
    #train the model
    model = train_model(test, inputs, output_layer)
    #stop the clock
    test_training_time['finish'] = time.time()
    #append to dict
    training_times.append(test_training_time)
    #save the weights of the model for evaluation
    weight_file_name = test['name']
    model_tools.save_network(model, weight_file_name)


#save the times
with open('train_time.p', 'wb+') as file:
    pickle.dump(training_times, file)
