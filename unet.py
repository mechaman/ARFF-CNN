from functools import partial

from keras.layers import *
from keras.engine import Model
from keras.optimizers import Adam
from helper import create_convolution_block, concatenate
from metrics import weighted_dice_coefficient_loss
import numpy as np
import tensorflow as tf 
import pdb


def unet(inputShape=(1,150,256,256)):
       
    # paddedShape = (data_ch.shape[1]+2, data_ch.shape[2]+2, data_ch.shape[3]+2, data_ch.shape[4])

    #initial padding
    input_img = Input(shape=inputShape)
    print(input_img)
    # x = Lambda(lambda x: K.spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1))),
    #     output_shape=paddedShape)(input_img) #Lambda layers require output_shape

    #your original code without padding for MaxPooling layers (replace input_img with x)
    pdb.set_trace()
    # downsampling phase
    # pdb.set_trace()
    conv1 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(input_img)
    #print "conv1 shape:",conv1.shape
    conv1 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv1)
    #print "conv1 shape:",conv1.shape
    pool1 = MaxPooling3D(pool_size=2, data_format='channels_first')(conv1)
    #print "pool1 shape:",pool1.shape

    conv2 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(pool1)
    #print "conv2 shape:",conv2.shape
    conv2 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv2)
    #print "conv2 shape:",conv2.shape
    pool2 = MaxPooling3D(pool_size=2, data_format='channels_first')(conv2)
    #print "pool2 shape:",pool2.shape

    conv3 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(pool2)
    #print "conv3 shape:",conv3.shape
    conv3 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv3)
    #print "conv3 shape:",conv3.shape
    pool3 = MaxPooling3D(pool_size=2, data_format='channels_first')(conv3)
    #print "pool3 shape:",pool3.shape

    conv4 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(pool3)
    conv4 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling3D(pool_size=2, data_format='channels_first')(drop4)

    conv5 = Conv3D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(pool4)
    conv5 = Conv3D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv5)
    drop5 = Dropout(0.5)(conv5)

#upsampling portion of unet
    up6 = Conv3D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(UpSampling3D(size = 2, data_format='channels_first')(drop5))
    merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 1)
    conv6 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(merge6)
    conv6 = Conv3D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv6)

    up7 = Conv3D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(UpSampling3D(size = 2, data_format='channels_first')(conv6))
    merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 1)
    conv7 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(merge7)
    conv7 = Conv3D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv7)

    up8 = Conv3D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(UpSampling3D(size = 2, data_format='channels_first')(conv7))
    merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 1)
    conv8 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(merge8)
    conv8 = Conv3D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv8)

    up9 = Conv3D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(UpSampling3D(size = 2, data_format='channels_first')(conv8))
    merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 1)
    conv9 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(merge9)
    conv9 = Conv3D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv9)
    conv9 = Conv3D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal', data_format='channels_first')(conv9)
#binary sigmoid output for masking.
    conv10 = Conv3D(1, 1, activation = 'sigmoid')(conv9)

    model= Model(input_img, conv10)
    model.compile(optimizer='Adam', loss='binary_crossentropy')

    return model


