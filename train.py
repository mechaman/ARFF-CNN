import os
import numpy as np
from keras.optimizers import *
from keras.models import *
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, Callback, CSVLogger
from keras.utils.training_utils import multi_gpu_model

from unet import myUnet
from data_loader import *
from data_generator import DataGenerator



def train_zhi_unet(slice_type='side', dim = (256,256), epochs=2):        
    print('-------- Instantiating Zhi Unet for ', slice_type, dim, '---------------')
    # Model prefix string for organized persistance
    model_prefix = 'zhi_unet_' + slice_type
    train_dir = ('./slice_data_' + slice_type)
    weights_fp = ('./weights/' + model_prefix + '.hdf5')
    # Loading Data
    print('Loading Data...')
    partition={}
    (partition['x_train'],
    partition['y_train'],
    partition['x_val'],
    partition['y_val'],
    partition['x_test'],
    partition['y_test'])  = load_data(train_dir, split=(80, 10, 10))
    
    # Save x_val to text file
    #x_val_fn = ('./val_logs/val_' + slice_type + '.txt')
    #np.savetxt(x_val_fn, np.array(partition['x_val']), delimiter=',', fmt="%s")
    
    # Parameters for input data
    params1 = {'dim': dim,
              'batch_size': 32,
              'n_channels': 1,
              'shuffle': True}
    
    params2 = {'dim': dim,
          'batch_size': 16,
          'n_channels': 1,
          'shuffle': False}
    training_generator = DataGenerator(partition['x_train'], partition['y_train'], **params1)
    validation_generator = DataGenerator(partition['x_val'], partition['y_val'], **params2)
    print('-------------' , len(training_generator))
    print('-------------',  len(validation_generator))
    print('Loaded Data...')

    # Initialize UNet
    print("Instantiate UNET")
    unet = myUnet(img_rows=dim[0], img_cols=dim[1])
    model = unet.get_unet_zhi()
    # Initialize multi_model
    #model.load_weights(weights_fp)
    #multi_model = multi_gpu_model(model, gpus=2)
    #multi_model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy','mse'])
    model_fp = ('./models/' + model_prefix + '.hdf5')
    model_checkpoint = ModelCheckpoint(model_fp, monitor='loss',verbose=1,
                                       save_best_only=True)
    #print(multi_model.summary())

    # Train UNet
    print('Fitting Model...')
    log_fp = ('./logs/' + model_prefix + '2.csv')
    csv_logger = CSVLogger(log_fp, append=True, separator=';')
    model.fit_generator(generator=training_generator,
            validation_data=validation_generator,
            validation_steps = 1,
            epochs=epochs,
            verbose=1,
            callbacks =[model_checkpoint, csv_logger],
            use_multiprocessing=True,
            workers=6)
    
    # Save weights
    #print(multi_model.summary())
    #model = multi_model.get_layer('model_1')
    model.save_weights(weights_fp)
                                  

def predict(slice_type='side'):
	# @DEBUG MAKE SURE WORKS!!!!
	# Model prefix string for organized persistance
    model_prefix = 'zhi_unet_' + slice_type
    
    # Loading Data
    print('Loading Data...')
    partition={}

    (_,
     _,
     _,
     _,
     x_test,
     y_test)  = load_data('slice_data_side_test', split=(0.0, 0.0, 100))

    # Parameters for input data
    params = {'dim': (256,256),
                  'batch_size': 1,
                  'n_channels': 1,
                  'shuffle': True}
    test_generator = DataGenerator(x_test, y_test, **params)
    print('Loaded Data...')
    
    # Initialize UNet
    print("Instantiate UNET")
    unet = myUnet()
    model = unet.get_unet_zhi()
    model_fp = ('./models/' + model_prefix + '.hdf5')
    model_checkpoint = ModelCheckpoint(model_fp, monitor='loss',verbose=1,
                                           save_best_only=True)
    weight_fp = ('./weights/' + model_prefix + '.hdf5')
    model.load_weights(weights_fp)

    # Predict using UNet
    print("Predicting via UNet")
    unet.predict(model, test_generator, y_test, slice_type)

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
                  'shuffle': False}
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
    #train_basic_unet(slice_type='side')
    #train_zhi_unet(slice_type='side', dim = (256, 256), epochs=1)
    #train_zhi_unet(slice_type='top', dim = (256, 256), epochs=3)
    train_zhi_unet(slice_type='back', dim=(256, 256), epochs=2)
    
