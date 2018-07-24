import os
import numpy as np
from keras.optimizers import *
from keras.models import *
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, Callback, CSVLogger
from keras.utils.training_utils import multi_gpu_model

from utils_refactored import *
from unet_refactored import myUnet
from data_generator_refactored import DataGenerator

def get_2dUnet(dim=(256, 256)):
    '''
    Input -> slice : str, dim : (int, int)
    Instantiate 2d unet
    '''
    unet = myUnet(img_rows=dim[0], img_cols=dim[1])
    model = unet.get_unet()
    return model

def train_side(model, data, batch_size):
    '''
    Train on the side data.
    '''
    # Set input & output masks
    input_img = (data[0])[0,:,:,:]
    output_mask = (data[1])[0,:,:,:]
    # Prepare aux. vars. for iter. train on batch
    T, B, S = input_img.shape
    num_batches = int(np.ceil(S/batch_size))
    # Batch of side slices
    side_slice_batch_x = None
    side_slice_batch_y = None
    # Train on each batch@TODO
    for batch_num in range(0, num_batches):
        s_idx = batch_num*batch_size
        # Check if last batch
        if batch_num == (num_batches-1):
            side_slice_batch_x = input_img[:,:,s_idx:]
            side_slice_batch_y = output_mask[:,:,s_idx:]
        else:
            f_idx = s_idx + batch_size
            print(s_idx, f_idx) 
            side_slice_batch_x = input_img[:, :, s_idx:f_idx]
            side_slice_batch_y= output_mask[:, :, s_idx:f_idx]
        # Flip axes to make index to iterate over first 
        side_slice_batch_x = np.swapaxes(np.swapaxes(side_slice_batch_x,0,2), 1,2)[:,:,:,np.newaxis]
        side_slice_batch_y = np.swapaxes(np.swapaxes(side_slice_batch_y,0,2), 1,2)[:,:,:,np.newaxis]
        print(side_slice_batch_x.shape, side_slice_batch_y.shape)
        loss, dice = model.train_on_batch(side_slice_batch_x, side_slice_batch_y)
    return loss, dice

def train_top(model, data, batch_size):
    '''
    Train on top data.
    '''
    pass

def train_back(model, data, batch_size):
    '''
    Train on back data.
    '''
    pass

def train_2dUnet(model, data, batch_size, slice_type='side'):
    '''
    Input -> slice : str, dim : (int, int)
    Train the unet on approp. slice
    '''
    # Construct model & weight fp(s) 
    model_prefix = '2dunet_' + slice_type
    weights_fp = ('./weights/' + model_prefix + '.hdf5')
    model_fp = ('./models/' + model_prefix + '.hdf5')
    model_checkpoint = ModelCheckpoint(model_fp, monitor='loss',verbose=1,
                                       save_best_only=True)
    #@TODO Add train_on_batch and return list of [bce, dice]
     
    if slice_type == 'side':
        metrics = train_side(model, data, batch_size)
    elif slice_type == 'top':
        pass 
    elif slice_type == 'back':
        pass

def train_2dUnet_ensemble(dim = (256,256,256), epochs=2):        
    sides  = 3
    dim = (256, 256)
    batch_size = 32
    ## Load Data 
    print('Loading Data...')
    train_dir = './data'
    partition={}
    (partition['x_train'],
    partition['y_train'],
    partition['x_val'],
    partition['y_val'],
    partition['x_test'],
    partition['y_test'])  = load_partitioned_data(train_dir, split=(80, 10, 10))
    # Define Parameters for generators
    params1 = { 'dim': (256, 256, 256),
                'batch_size': 1,
                'n_channels': 1,
                'shuffle': False}
    # Instantiate dataset generators 
    training_generator = DataGenerator(partition['x_train'], partition['y_train'], **params1)
    validation_generator = DataGenerator(partition['x_val'], partition['y_val'], **params1)
    
    ## Instantiate Models
    print("Instantiate Unet Ensemble")
    # Create model array 
    model_arr = []
    for i in range(3):
        model_arr.append(get_2dUnet(dim=(256,256)))
    
    ## Train Models
    loss = []
    dice_2d = []
    dice_3d = []
    # Create a list of loss/dice for each model
    for i in range(0,len(model_arr)):
        loss.append([])
        dice_2d.append([])
    
    ## Training Loop 
    for img in training_generator:
        #@TODO loss, dice = train_2dUnet(model_arr[0], img[:,:,?]
        print(img[0].shape)
        metrics = train_2dUnet(model_arr[2], img, batch_size, slice_type='side')
        print(metrics)
        break
    '''
    log_fp = ('./logs/' + model_prefix + '3.csv')
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
    '''                              

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

    

if __name__ == '__main__':
    train_2dUnet_ensemble(dim=(256,256,256), epochs=2) 
    #train_2d_unet(slice_type='side', dim = (256, 256), epochs=1)
    #train_2d_unet(slice_type='top', dim = (256, 256), epochs=3)
    #train_2d_unet(slice_type='back', dim=(256, 256), epochs=1)
    
