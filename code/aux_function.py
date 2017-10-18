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
def separable_conv2d_batchnorm(input_layer, filters, strides=1):
    output_layer = SeparableConv2DKeras(filters=filters,kernel_size=3, strides=strides,
                             padding='same', activation='relu')(input_layer)

    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer

def conv2d_batchnorm(input_layer, filters, kernel_size=3, strides=1):
    output_layer = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                      padding='same', activation='relu')(input_layer)

    output_layer = layers.BatchNormalization()(output_layer)
    return output_layer


def bilinear_upsample(input_layer):
    output_layer = BilinearUpSampling2D((2,2))(input_layer)
    return output_layer

def encoder_block(input_layer, filters, strides):

    #  Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    lay1 = separable_conv2d_batchnorm(input_layer, filters, strides)
    #output_layer = conv2d_batchnorm(lay1, filters, kernel_size=1, strides=1)
    return lay1

def double_encoder_block(input_layer, filters, strides):

    #  Create a separable convolution layer using the separable_conv2d_batchnorm() function.
    lay1 = separable_conv2d_batchnorm(input_layer, filters, strides)
    output_layer = conv2d_batchnorm(lay1, filters, kernel_size=1, strides=1)
    return output_layer


def decoder_block(small_ip_layer, large_ip_layer, filters):

    # TODO Upsample the small input layer using the bilinear_upsample() function.
    upsampled_layer = bilinear_upsample(small_ip_layer)
    # TODO Concatenate the upsampled and large input layers using layers.concatenate
    concatenated_layers = layers.concatenate([upsampled_layer, large_ip_layer])
    # TODO Add some number of separable convolution layers
    lay1 = separable_conv2d_batchnorm(concatenated_layers, filters, strides=1)
    output_layer = separable_conv2d_batchnorm(lay1, filters, strides=1)
    return output_layer

def fcn_model(inputs, num_classes):

    # TODO Add Encoder Blocks.
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encod1 = encoder_block(inputs, 32, 2)
    encod2 = encoder_block(encod1, 64, 2)
    encod3 = encoder_block(encod2, 128, 2)
    encod4 = encoder_block(encod3, 256, 2)
    encod5 = encoder_block(encod4, 512, 2)

    input_shape = inputs.get_shape().as_list()

    print('input layer size',input_shape)
    print('encod1 layer size',encod1.get_shape().as_list())
    print('encod2 layer size',encod2.get_shape().as_list())
    print('encod3 layer size',encod3.get_shape().as_list())
    print('encod4 layer size',encod4.get_shape().as_list())
    print('encod5 layer size',encod5.get_shape().as_list())
    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_layer = conv2d_batchnorm(encod5, 512, kernel_size=1, strides=1)

    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decod1 = decoder_block(conv_layer, encod4, 512)
    decod2 = decoder_block(decod1, encod3, 256)
    decod3 = decoder_block(decod2, encod2, 128)
    decod4 = decoder_block(decod3, encod1, 64)
    decod5 = decoder_block(decod4, inputs, 32)


    print('conv_layer layer size',conv_layer.get_shape().as_list())
    print('decoder1  layer size',decod1.get_shape().as_list())
    print('decoder2  layer size',decod2.get_shape().as_list())
    print('decoder3  layer size',decod3.get_shape().as_list())
    print('decoder4  layer size',decod4.get_shape().as_list())
    output_shape = decod5.get_shape().as_list()
    print('decoder5  layer size',output_shape)

    assert (input_shape[1:2] == output_shape[1:2]), "Input/Output shapes are not the shame %r !=  %r" % (input_shape,output_shape)
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(decod5)

def fcn_model2(inputs, num_classes):

    # TODO Add Encoder Blocks.
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encod1 = encoder_block(inputs, 2, 2)
    input_shape = inputs.get_shape().as_list()
    conv_layer = conv2d_batchnorm(encod1, 2, kernel_size=1, strides=1)
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decod1 = decoder_block(conv_layer, inputs, 2)

    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(decod1)

def fcn_model_best(inputs, num_classes):

    # TODO Add Encoder Blocks.
    # Remember that with each encoder layer, the depth of your model (the number of filters) increases.
    encod1 = encoder_block(inputs, 32, 2)
    encod2 = encoder_block(encod1, 64, 2)
    encod3 = encoder_block(encod2, 128, 2)
    encod4 = encoder_block(encod3, 256, 2)
    print('encod1 layer size',encod1.get_shape().as_list())
    print('encod2 layer size',encod2.get_shape().as_list())
    print('encod3 layer size',encod3.get_shape().as_list())
    print('encod4 layer size',encod4.get_shape().as_list())

    # TODO Add 1x1 Convolution layer using conv2d_batchnorm().
    conv_layer = conv2d_batchnorm(encod4, 256, kernel_size=1, strides=1)

    print('conv_layer layer size',conv_layer.get_shape().as_list())
    # TODO: Add the same number of Decoder Blocks as the number of Encoder Blocks
    decod1 = decoder_block(conv_layer, encod3, 256)
    decod2 = decoder_block(decod1, encod2, 128)
    decod3 = decoder_block(decod2, encod1, 64)
    decod4 = decoder_block(decod3, inputs, 32)
    print('decoder1  layer size',decod1.get_shape().as_list())
    print('decoder2  layer size',decod2.get_shape().as_list())
    print('decoder3  layer size',decod3.get_shape().as_list())
    print('decoder4  layer size',decod4.get_shape().as_list())
    # The function returns the output layer of your model. "x" is the final layer obtained from the last decoder_block()
    return layers.Conv2D(num_classes, 3, activation='softmax', padding='same')(decod4)

def train_model(test, inputs, output_layer,image_shape):
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
