import os
import numpy as np
from keras.optimizers import *
from keras.models import *
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, Callback, CSVLogger
from keras.utils.training_utils import multi_gpu_model

from utils import *
from metrics import *
from unet import myUnet
from data_generator import DataGenerator

def get_2dUnet(dim=(256, 256), slice_type='side', model_fp=''):
    '''
    Input -> slice : str, dim : (int, int)
    Instantiate 2d unet
    '''
    # Construct if model_fp not specified
    if model_fp == '':
        unet = myUnet(img_rows=dim[0], img_cols=dim[1])
        model = unet.get_unet()
    else:
        model = load_model(model_fp)
<<<<<<< HEAD
        print('Loaded existing model @ :', model_fp)
=======
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
    return model

def save_2dUnet(model, model_fp='', slice_type='side'):
    '''
    Save model.
    '''

    if model_fp == '':
        # Construct model & weight fp(s) 
        model_prefix = '2dunet_' + slice_type
        model_fp = ('./models/' + model_prefix + '.hdf5')
    model.save(model_fp)
    return True

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

<<<<<<< HEAD
def predict_vol(model, data, batch_size, slice_type):
=======
def predict_side_vol(model, data, batch_size, slice_type):
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
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
<<<<<<< HEAD
            #print(s_idx, f_idx)
=======
            print(s_idx, f_idx)
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
        side_slice_batch_x = get_slice_batch(input_img,
                                            slice_type,
                                            s_idx, f_idx)
        # Flip axes to make index to iterate over first 
        side_slice_batch_x = flip_index(side_slice_batch_x, slice_type = slice_type) 
<<<<<<< HEAD
        predicted_batch.extend(model.predict_on_batch(side_slice_batch_x))
    
=======
        print(side_slice_batch_x.shape)
        predicted_batch.extend(model.predict_on_batch(side_slice_batch_x))
        if batch_num == 1:
            break
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
    predicted_vol = np.array(predicted_batch)
    # Compute the average loss and dice scores
    return predicted_vol


<<<<<<< HEAD
def train_2dUnet(model, data, batch_size, slice_type):
=======
def train_side(model, data, batch_size, slice_type):
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
    '''
    Train on the side data.
    '''
    # Set input & output masks
    input_img = (data[0])[0,:,:,:]
    output_mask = (data[1])[0,:,:,:]
    # Prepare aux. vars. for iter. train on batch
    T, B, S = input_img.shape
    # Number of batches is invariant to dim.
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
            f_idx = None 
        else:
            f_idx = s_idx + batch_size
<<<<<<< HEAD
            #print(s_idx, f_idx) 
=======
            print(s_idx, f_idx) 
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
        side_slice_batch_x = get_slice_batch(input_img,
                                            slice_type,
                                            s_idx, f_idx)
        side_slice_batch_y= get_slice_batch(input_img,
                                            slice_type,
                                            s_idx, f_idx)

        # Flip axes to make index to iterate over first 
        side_slice_batch_x = flip_index(side_slice_batch_x, slice_type=slice_type)
        side_slice_batch_y = flip_index(side_slice_batch_y, slice_type=slice_type)
<<<<<<< HEAD
        #print(side_slice_batch_x.shape, side_slice_batch_y.shape)
=======
        print(side_slice_batch_x.shape, side_slice_batch_y.shape)
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
        # Compute and store loss & dice
        metrics = model.train_on_batch(side_slice_batch_x, side_slice_batch_y)
        loss.append(metrics[0])
        dice.append(metrics[1])
<<<<<<< HEAD
    
=======
        break
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
    # Compute the average loss and dice scores
    avg_loss = np.mean(loss)
    avg_dice = np.mean(dice)
    return avg_loss, avg_dice


<<<<<<< HEAD
=======
def train_2dUnet(model, data, batch_size, slice_type='side'):
    '''
    Input -> slice : str, dim : (int, int)
    Train the unet on approp. slice
    '''
    #model_checkpoint = ModelCheckpoint(model_fp, monitor='loss',verbose=1,
    #                                   save_best_only=True)
    #@TODO Add train_on_batch and return list of [bce, dice]
    loss, dice = train_side(model, data, batch_size, slice_type=slice_type)
    return loss, dice

>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
def predict_3d_mask(model_arr, data, batch_size):
    '''
    Generate the 3D Mask to compute dice coef. against
    '''
    # Predict Side Vol.
<<<<<<< HEAD
    side_vol = predict_vol(model_arr[0],
                                data, batch_size,
                                slice_type='side')
    #print(side_vol.shape)
    # Predict Back Vol.
    back_vol = predict_vol(model_arr[1],
                                data, batch_size,
                                slice_type='back')
    #print(back_vol.shape)
    # Predict Top Vol.
    top_vol = predict_vol(model_arr[2],
                                data, batch_size,
                                slice_type='top')
    #print(top_vol.shape) 
=======
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
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
    # Compute avg. voxels across top, side, and back vols
    avg_vol = np.mean([top_vol, back_vol, side_vol], axis=0)
    
    return avg_vol 

<<<<<<< HEAD
def save_stats(loss, dice_2d, dice_3d_train, dice_3d_val):
    '''
    Save collected stats from training and validation
    '''
    # Save loss during training
    loss_np = np.array(loss)
    np.savetxt('./logs/loss.csv', loss_np, delimiter=',')
    # Save dice_2d during training
    dice_2d_np = np.array(dice_2d)
    np.savetxt('./logs/dice_2d.csv', dice_2d_np, delimiter=',')
    # Save dice_3d_train during training
    dice_3d_train_np = np.array(dice_3d_train)
    np.savetxt('./logs/dice_3d_train.csv', dice_3d_train_np, delimiter=',')
    # save dice_3d_val during training
    dice_3d_val_np = np.array(dice_3d_val)
    np.savetxt('./logs/dice_3d_val.csv', dice_3d_val_np, delimiter=',')

    return True

def test_2dUnet_ensemble(dim=(256,256,256)):
    sides  = 3
    slice_types = ['back', 'top', 'side']
    dim = (256, 256)
    batch_size = 16
=======

def train_2dUnet_ensemble(dim = (256,256,256), epochs=2):        
    sides  = 3
    slice_types = ['back', 'top', 'side']
    dim = (256, 256)
    batch_size = 1
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
    ## Load Data 
    print('Loading Data...')
    train_dir = './data'
    partition={}
    (partition['x_train'],
    partition['y_train'],
    partition['x_val'],
    partition['y_val'],
    partition['x_test'],
<<<<<<< HEAD
    partition['y_test'])  = load_partitioned_data(train_dir, split=(100, 0, 0))
    # Define Parameters for generators
    params1 = { 'dim': (256, 256, 256),
                'batch_size': 1,
                'n_channels': 1,
                'shuffle': True}
    ## Instantiate Models
    print("Instantiate Unet Ensemble")
    # Create model array 
    model_arr = []
    # Only if models already exist 
    model_fp_arr = ['./models/2dunet_back.hdf5', './models/2dunet_top.hdf5', './models/2dunet_side.hdf5']
    for i in range(3):
        model_arr.append(get_2dUnet(dim=(256,256), slice_type=slice_types[i], model_fp=model_fp_arr[i]))
    # Instantiate Test Generator
    test_generator = DataGenerator(partition['x_test'], partition['y_test'], **params1)
    # Iterator through test images 
    for img in test_generator:
        #@TODO Generator won't stop, need this statement for now. Fix
        if test_generator.index > len(test_generator):
            break
        print('Validation on img number : ', test_generator.index)
        mask_vol = predict_3d_mask(model_arr, img, batch_size)
        #Ground Truth Shape:  (1, 256, 256, 256) Predicted Shape:  (256, 256, 256, 1)
        dice_3d_val.append(dice_coef(img[1][0,:,:,:], mask_vol[:,:,:,0], threshold=0.59))
    # Print mean dice score
    print(np.mean(dice_3d_val))

def train_2dUnet_ensemble(dim = (256,256,256), epochs=2):        
    sides  = 3
    slice_types = ['back', 'top', 'side']
    dim = (256, 256)
    batch_size = 16
    ## Load Data 
    print('Loading Data...')
    train_dir = './data'
    partition={}
    (partition['x_train'],
    partition['y_train'],
    partition['x_val'],
    partition['y_val'],
    partition['x_test'],
=======
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
=======
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
    partition['y_test'])  = load_partitioned_data(train_dir, split=(80, 10, 10))
    # Define Parameters for generators
    params1 = { 'dim': (256, 256, 256),
                'batch_size': 1,
                'n_channels': 1,
<<<<<<< HEAD
<<<<<<< HEAD
                'shuffle': True}
=======
=======
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
                'shuffle': False}
    # Instantiate dataset generators 
    training_generator = DataGenerator(partition['x_train'], partition['y_train'], **params1)
    validation_generator = DataGenerator(partition['x_val'], partition['y_val'], **params1)
<<<<<<< HEAD
    
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
    ## Instantiate Models
    print("Instantiate Unet Ensemble")
    # Create model array 
    model_arr = []
<<<<<<< HEAD
    # Only if models already exist 
    model_fp_arr = ['./models/2dunet_back.hdf5', './models/2dunet_top.hdf5', './models/2dunet_side.hdf5']
    for i in range(3):
        model_arr.append(get_2dUnet(dim=(256,256), slice_type=slice_types[i], model_fp=model_fp_arr[i]))
    
    # Iterate through Epochs 
    for i in range(0, epochs): 
        print('Epoch : ', i)
        ## Train Models
        loss = []
        dice_2d = []
        dice_3d_train = []
        # Create a list of loss/dice for each model
        for i in range(0,len(model_arr)):
            loss.append([])
            dice_2d.append([])
        
        # Instantiate dataset generators 
        training_generator = DataGenerator(partition['x_train'], partition['y_train'], **params1)
        validation_generator = DataGenerator(partition['x_val'], partition['y_val'], **params1)
=======
    for i in range(3):
        model_arr.append(get_2dUnet(dim=(256,256), slice_type=slice_types[i]))
    
    ## Train Models
    loss = []
    dice_2d = []
    dice_3d_train = []
    # Create a list of loss/dice for each model
    for i in range(0,len(model_arr)):
        loss.append([])
        dice_2d.append([])
=======
    
    ## Instantiate Models
    print("Instantiate Unet Ensemble")
    # Create model array 
    model_arr = []
    for i in range(3):
        model_arr.append(get_2dUnet(dim=(256,256), slice_type=slice_types[i]))
    
    ## Train Models
    loss = []
    dice_2d = []
    dice_3d_train = []
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
        mask_vol = predict_3d_mask(model_arr, img, batch_size)
        # Compute dice coef. 
        print('Output mask_vol & ground truth mask: ')
        #TODO remove [0,0,:,:] -> [:,:,:,:] - just testing out dice_coef
        dice_3d_train.append(dice_coef(img[1][0,0,:,:], mask_vol[0,:,:,0], threshold=0.59))
        break
    print(dice_3d_train)
    
    ## Validation Loop
    dice_3d_val = []
    for img in validation_generator:
        mask_vol = predict_3d_mask(model_arr, img, batch_size)
        #TODO remove [0,0,:,:] -> [:, :, :, :] - just testing out dice_coef 
        dice_3d_val.append(dice_coef(img[1][0,0,:,:], mask_vol[0,:,:,0], threshold=0.59))
        break
    print(dice_3d_val)
    
    ## Save the models
    for idx, model in enumerate(model_arr):
        save_2dUnet(model, slice_type=slice_types[idx])
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
    
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
        mask_vol = predict_3d_mask(model_arr, img, batch_size)
        # Compute dice coef. 
        print('Output mask_vol & ground truth mask: ')
        #TODO remove [0,0,:,:] -> [:,:,:,:] - just testing out dice_coef
        dice_3d_train.append(dice_coef(img[1][0,0,:,:], mask_vol[0,:,:,0], threshold=0.59))
        break
    print(dice_3d_train)
    
    ## Validation Loop
    dice_3d_val = []
    for img in validation_generator:
        mask_vol = predict_3d_mask(model_arr, img, batch_size)
        #TODO remove [0,0,:,:] -> [:, :, :, :] - just testing out dice_coef 
        dice_3d_val.append(dice_coef(img[1][0,0,:,:], mask_vol[0,:,:,0], threshold=0.59))
        break
    print(dice_3d_val)
    
    ## Save the models
    for idx, model in enumerate(model_arr):
        save_2dUnet(model, slice_type=slice_types[idx])
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
    
        ## Training Loop 
        print('Number of images to train on : ', len(training_generator))  
        for img in training_generator:
            #@TODO Generator won't stop, need this statement for now. Fix
            if training_generator.index > len(training_generator):
                break
            print('Training on img number : ', training_generator.index) 
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
            #print(img_loss, img_dice)
            loss[2].append(img_loss)
            dice_2d[2].append(img_dice)
            ## Get 3D Dice
            # Predict 3D Mask
            #mask_vol = predict_3d_mask(model_arr, img, batch_size)
            # Compute dice coef. 
            #Ground Truth Shape:  (1, 256, 256, 256) Predicted Shape:  (256, 256, 256, 1)
            #dice_3d_train.append(dice_coef(img[1][0,:,:,:], mask_vol[:,:,:,0], threshold=0.59))
        print(dice_3d_train[-10:])
         
        ## Validation Loop
        dice_3d_val = []
        '''print('Number of images to validate on : ', len(validation_generator))
        for img in validation_generator:
            #@TODO Generator won't stop, need this statement for now. Fix
            if validation_generator.index > len(validation_generator):
                break
            print('Validation on img number : ', validation_generator.index)
            mask_vol = predict_3d_mask(model_arr, img, batch_size)
            #Ground Truth Shape:  (1, 256, 256, 256) Predicted Shape:  (256, 256, 256, 1)
            dice_3d_val.append(dice_coef(img[1][0,:,:,:], mask_vol[:,:,:,0], threshold=0.59))
        print(dice_3d_val[-10:])
        '''
        # Save stats  
        save_stats(loss, dice_2d, dice_3d_train, dice_3d_val)

        ## Save the models
        for idx, model in enumerate(model_arr):
            save_2dUnet(model, slice_type=slice_types[idx])
        

if __name__ == '__main__':
<<<<<<< HEAD
<<<<<<< HEAD
    train_2dUnet_ensemble(dim=(256,256,256), epochs=1) 
=======
    train_2dUnet_ensemble(dim=(256,256,256), epochs=2) 
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
=======
    train_2dUnet_ensemble(dim=(256,256,256), epochs=2) 
>>>>>>> 2762a2c26d5e9500f607cdad8257ccec9d7929a7
    
