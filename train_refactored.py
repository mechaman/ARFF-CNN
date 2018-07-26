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

def get_slice_batch(data, slice_type, s_idx, f_idx):
    '''
    Return a batch of slices for a particular side
    '''
    if slice_type == 'top':
        return data[s_idx:f_idx, :, :]
    elif slice_type == 'back':
        return data[:, s_idx:f_idx, :]
    elif slice_type == 'side':
        return data[:, :, s_idx:f_idx]

def flip_index(data, slice_type):
    '''
    Flips the index so that the slices are indexable by
    the keras model for training.
    '''
    if slice_type == 'top':
        return data[:, :, :, np.newaxis]
    elif slice_type == 'back':
        return np.swapaxes(data, 0, 1)[:,:,:,np.newaxis]
    elif slice_type == 'side':
        return np.swapaxes(np.swapaxes(data,0,2), 1,2)[:,:,:,np.newaxis]

def predict_side_vol(model, data, batch_size, slice_type):
    '''
    Predict on the side data.
    '''
    # Set input & output masks
    input_img = (data[0])[0,:,:,:]
    # Prepare aux. vars. for iter. train on batch
    T, B, S = input_img.shape
    # Number of batches is invariant to dim.
    num_batches = int(np.ceil(S/batch_size))
    # Batch of side slices
    side_slice_batch_x = None
    # Create list of predicted batches
    predicted_batch = []
    # Train on each batch
    for batch_num in range(0, num_batches):
        s_idx = batch_num*batch_size
        # Check if last batch
        if batch_num == (num_batches-1):
            f_idx = None
        else:
            f_idx = s_idx + batch_size
            print(s_idx, f_idx)
        side_slice_batch_x = get_slice_batch(input_img,
                                            slice_type,
                                            s_idx, f_idx)
        # Flip axes to make index to iterate over first 
        side_slice_batch_x = flip_index(side_slice_batch_x, slice_type = slice_type) 
        print(side_slice_batch_x.shape)
        predicted_batch.extend(model.predict_on_batch(side_slice_batch_x))
        if batch_num == 1:
            break
    predicted_vol = np.array(predicted_batch)
    # Compute the average loss and dice scores
    return predicted_vol


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
    # Loss & Dice array
    loss = []
    dice = []
    # Train on each batch
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
        metrics = model.train_on_batch(side_slice_batch_x, side_slice_batch_y)
        loss.append(metrics[0])
        dice.append(metrics[1])
        break
    # Compute the average loss and dice scores
    avg_loss = np.mean(loss)
    avg_dice = np.mean(dice)
    return avg_loss, avg_dice

def train_top(model, data, batch_size):
    '''
    Train on top data.
    '''
    # Set input & output masks
    input_img = (data[0])[0,:,:,:]
    output_mask = (data[1])[0,:,:,:]
    # Prepare aux. vars. for iter. train on batch
    T, B, S = input_img.shape
    num_batches = int(np.ceil(T/batch_size))
    # Batch of side slices
    side_slice_batch_x = None
    side_slice_batch_y = None
    # Loss & Dice array
    loss = []
    dice = []
    # Train on each batch
    for batch_num in range(0, num_batches):
        s_idx = batch_num*batch_size
        # Check if last batch
        if batch_num == (num_batches-1):
            side_slice_batch_x = input_img[s_idx:,:,:]
            side_slice_batch_y = output_mask[s_idx:,:,:]
        else:
            f_idx = s_idx + batch_size
            print(s_idx, f_idx) 
            side_slice_batch_x = input_img[s_idx:f_idx, :, :, np.newaxis]
            side_slice_batch_y= output_mask[s_idx:f_idx, :, :, np.newaxis]
        print(side_slice_batch_x.shape, side_slice_batch_y.shape)
        metrics = model.train_on_batch(side_slice_batch_x, side_slice_batch_y)
        loss.append(metrics[0])
        dice.append(metrics[1])
        break
    # Compute the average loss and dice scores
    avg_loss = np.mean(loss)
    avg_dice = np.mean(dice)
    return avg_loss, avg_dice

def train_back(model, data, batch_size):
    '''
    Train on back data.
    '''
    # Set input & output masks
    input_img = (data[0])[0,:,:,:]
    output_mask = (data[1])[0,:,:,:]
    # Prepare aux. vars. for iter. train on batch
    T, B, S = input_img.shape
    num_batches = int(np.ceil(B/batch_size))
    # Batch of side slices
    side_slice_batch_x = None
    side_slice_batch_y = None
    # Loss & Dice array
    loss = []
    dice = []
    # Train on each batch
    for batch_num in range(0, num_batches):
        s_idx = batch_num*batch_size
        # Check if last batch
        if batch_num == (num_batches-1):
            side_slice_batch_x = input_img[:,s_idx:,:]
            side_slice_batch_y = output_mask[:,s_idx:,:]
        else:
            f_idx = s_idx + batch_size
            print(s_idx, f_idx) 
            side_slice_batch_x = input_img[:, s_idx:f_idx, :]
            side_slice_batch_y= output_mask[:, s_idx:f_idx, :]
        # Flip axes to make index to iterate over first 
        side_slice_batch_x = np.swapaxes(side_slice_batch_x,0,1)[:,:,:,np.newaxis]
        side_slice_batch_y = np.swapaxes(side_slice_batch_y,0,1)[:,:,:,np.newaxis]
        print(side_slice_batch_x.shape, side_slice_batch_y.shape)
        metrics = model.train_on_batch(side_slice_batch_x, side_slice_batch_y)
        loss.append(metrics[0])
        dice.append(metrics[1])
        break
    # Compute the average loss and dice scores
    avg_loss = np.mean(loss)
    avg_dice = np.mean(dice)
    return avg_loss, avg_dice


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
        loss, dice = train_side(model, data, batch_size)
    elif slice_type == 'top':
        loss, dice = train_top(model, data, batch_size)
    elif slice_type == 'back':
        loss, dice = train_back(model, data, batch_size)
    return loss, dice

def predict3dMask(model_arr, data, batch_size):
    '''
    Generate the 3D Mask to compute dice coef. against
    '''
    # Predict Side Vol.
    side_vol = predict_side_vol(model_arr[0],
                                data, batch_size,
                                slice_type='side')
    print(side_vol.shape)
    # Predict Back Vol.
    back_vol = predict_side_vol(model_arr[1],
                                data, batch_size,
                                slice_type='back')
    print(back_vol.shape)
    # Predict Top Vol.
    top_vol = predict_side_vol(model_arr[2],
                                data, batch_size,
                                slice_type='top')
    print(top_vol.shape) 
    # Compute avg. voxels across top, side, and back vols
    avg_vol = np.mean([top_vol, back_vol, side_vol], axis=0)
    
    return avg_vol 


def train_2dUnet_ensemble(dim = (256,256,256), epochs=2):        
    sides  = 3
    dim = (256, 256)
    batch_size = 1
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
        ## Train 2D Models & Get Metrics 
        # Model 0 : Back Slice 
        img_loss, img_dice = train_2dUnet(model_arr[0], img, batch_size, slice_type='back')
        loss[0].append(img_loss)
        dice_2d[0].append(img_dice)
        # Model 2 : Side Slice 
        img_loss, img_dice = train_2dUnet(model_arr[1], img, batch_size, slice_type='top')
        loss[1].append(img_loss)
        dice_2d[1].append(img_dice)
        # Model 2 : Side Slice 
        img_loss, img_dice = train_2dUnet(model_arr[2], img, batch_size, slice_type='side')
        print(img_loss, img_dice)
        loss[2].append(img_loss)
        dice_2d[2].append(img_dice)
        ## Get 3D Dice
        # Predict 3D Mask
        mask_vol = predict3dMask(model_arr, img, batch_size)
        # Compute dice coef. 
        dice_3d.append(dice_coef(img[1], mask_vol, threshold=0.59))
        break

    print(dice_3d)
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
    
