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
import time
import pickle
from view_test_cases import load_available_models
from aux_function import train_model,
                         fcn_model_best, fcn_model, fcn_model2 ,
                         decoder_block, encoder_block, double_encoder_block
                         bilinear_upsample,conv2d_batchnorm,separable_conv2d_batchnorm

image_hw = 160
image_shape = (image_hw, image_hw, 3)
num_classes = 3
#initial weights
inputs = layers.Input(image_shape)
# Call fcn_model()
output_layer = fcn_model_best(inputs, num_classes)
tests_cases = load_available_models()
print('---------------------')
print('')
kill_weights = input('kill weights after each test?....(y/n)?')
if kill_weights == 'y':
    kill_weights = True
else:
    kill_weights = False
# Save your trained model weights
#loop over all the test cases defined
for test in tests_cases:
    #train the model
    model = train_model(test, inputs, output_layer)
    #kill and initialize the weigts again for different calculation
    #dont kill the weights for the same epoch number
    if kill_weights:
        inputs = None
        inputs = layers.Input(image_shape)
        #reset the model
        output_layer = None
        output_layer = fcn_model_best(inputs, num_classes)
    #save the weights of the model for evaluation
    weight_file_name = test['name']
    model_tools.save_network(model, weight_file_name)
