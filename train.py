import os
import numpy as np
from keras.models import *
from keras.callbacks import ModelCheckpoint, Callback, CSVLogger

from unet import myUnet
from data_loader import *
from data_generator import DataGenerator

def train_zhi_unet(slice_type='side'):
    # Model prefix string for organized persistance
    model_prefix = 'zhi_unet_' + slice_type
    
    # Loading Data
    print('Loading Data...')
    partition={}
    (partition['x_train'],
     partition['y_train'],
     partition['x_val'],
     partition['y_val'],
     partition['x_test'],
     partition['y_test'])  = load_data('slice_data_side', split=(0.005, 0.005, 99.99))

    # Parameters for input data
    params = {'dim': (256,256),
                  'batch_size': 1,
                  'n_channels': 1,
                  'shuffle': True}
    training_generator = DataGenerator(partition['x_train'], partition['y_train'], **params)
    validation_generator = DataGenerator(partition['x_val'], partition['y_val'], **params)
    print('Loaded Data...')
    
    # Initialize UNet
    print("Instantiate UNET")
    unet = myUnet()
    model = unet.get_unet_zhi()
    model_fp = ('./models/' + model_prefix + '.hdf5')
    model_checkpoint = ModelCheckpoint(model_fp, monitor='loss',verbose=1,
                                           save_best_only=True)
    print(model.summary())
    
    # Train UNet
    print('Fitting Model...')
    log_fp = ('./logs/' + model_prefix + '.csv')
    csv_logger = CSVLogger(log_fp, append=True, separator=';')
    model.fit_generator(generator=training_generator,
                validation_data=validation_generator,
                validation_steps = 1,
                epochs=2,
                verbose=1,
                callbacks =[model_checkpoint, csv_logger],
                use_multiprocessing=True,
                workers=6)
    
    # Save weights
    weights_fp = ('./weights/' + model_prefix + '.hdf5')
    model.save_weights(weights_fp)


def train_basic_unet(slice_type = 'side'):
        # Model prefix string for organized persistance
    model_prefix = 'basic_unet_' + slice_type
    
    # Loading Data
    print('Loading Data...')
    partition={}
    (partition['x_train'],
     partition['y_train'],
     partition['x_val'],
     partition['y_val'],
     partition['x_test'],
     partition['y_test'])  = load_data('slice_data_side', split=(0.005, 0.005, 99.99))

    # Parameters for input data
    params = {'dim': (256,256),
                  'batch_size': 1,
                  'n_channels': 1,
                  'shuffle': True}
    training_generator = DataGenerator(partition['x_train'], partition['y_train'], **params)
    validation_generator = DataGenerator(partition['x_val'], partition['y_val'], **params)
    print('Loaded Data...')
    
    # Initialize UNet
    print("Instantiate UNET")
    unet = myUnet()
    model = unet.get_unet_basic()
    model_fp = ('./models/' + model_prefix + '.hdf5')
    model_checkpoint = ModelCheckpoint(model_fp, monitor='loss',verbose=1,
                                           save_best_only=True)
    print(model.summary())
    
    # Train UNet
    print('Fitting Model...')
    log_fp = ('./logs/' + model_prefix + '.csv')
    csv_logger = CSVLogger(log_fp, append=True, separator=';')
    model.fit_generator(generator=training_generator,
                validation_data=validation_generator,
                validation_steps = 1,
                epochs=2,
                verbose=1,
                callbacks =[model_checkpoint, csv_logger],
                use_multiprocessing=True,
                workers=6)
    
    # Save weights
    weights_fp = ('./weights/' + model_prefix + '.hdf5')
    model.save_weights(weights_fp)
    

if __name__ == '__main__':
    train_basic_unet(slice_type='side')
    #train_zhi_unet(slice_type='side')
    
