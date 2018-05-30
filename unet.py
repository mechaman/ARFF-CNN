from functools import partial

from keras.layers import *
from keras.engine import Model
from keras.optimizers import Adam
#from helper import create_convolution_block, concatenate
#from metrics import weighted_dice_coefficient_loss
import numpy as np
import tensorflow as tf 
import nibabel as nib
import pdb


def unet(inputShape=(1,None,256,256)):
       
    # paddedShape = (data_ch.shape[1]+2, data_ch.shape[2]+2, data_ch.shape[3]+2, data_ch.shape[4])

    #initial padding
    input_img = Input(shape=inputShape)
    print(input_img)
    # x = Lambda(lambda x: K.spatial_3d_padding(x, padding=((1, 1), (1, 1), (1, 1))),
    #     output_shape=paddedShape)(input_img) #Lambda layers require output_shape

    #your original code without padding for MaxPooling layers (replace input_img with x)
#    pdb.set_trace()
    # downsampling phase
    # pdb.set_trace()
    x= Conv3D(filters=8, kernel_size=3, activation='relu', padding='same', data_format='channels_first')(input_img)
     
    x= MaxPooling3D(pool_size=2, data_format='channels_first')(x)
     
    x= Conv3D(filters=8, kernel_size=3, activation='relu', padding='same', data_format='channels_first')(x)
    x= MaxPooling3D(pool_size=2, data_format='channels_first')(x)

    #upsampling phase
    x= UpSampling3D(size=2, data_format='channels_first')(x)
    x= Conv3D(filters=8, kernel_size=3, activation='relu', padding='same', data_format='channels_first')(x) # PADDING IS NOT THE SAME!!!!!
    x= UpSampling3D(size=2, data_format='channels_first')(x)
    x = ZeroPadding3D(padding=(1,0,0), data_format='channels_first')(x) 
    x= Conv3D(filters=1, kernel_size=1, activation='sigmoid', data_format='channels_first')(x)

    
    model= Model(input_img, x)
    model.compile(optimizer=Adam(lr=0.01), loss='binary_crossentropy', metrics=['mse'])

    return model


