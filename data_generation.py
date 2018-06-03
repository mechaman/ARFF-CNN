from __future__ import division
from data_loader import *
import numpy as np
import nibabel as nib
import keras
import pdb


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, examples, labels, batch_size=32, dim=(256,256), n_channels=1
                 , shuffle=True, third_dimension=False):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = examples
        self.n_channels = n_channels
        self.third_dimension = third_dimension
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_ys_temp = [self.labels[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_ys_temp, self.third_dimension)
        return X, y


    def resize_image(self, image):
        new_dims = tuple((image.shape[0] + (self.dim[0] - image.shape[0]), image.shape[1], image.shape[2]))
        new_image = np.zeros(new_dims)
        new_image[:image.shape[0], :image.shape[1], :image.shape[2]] = image 
        return new_image 


    def normalizeImg(self, x):
        # Normalize x
	mean_val = np.mean(x)
        max_val = np.max(x)
        min_val = np.min(x)
        norm_x = (x-mean_val)/(max_val - min_val + 1e-7)
        return norm_x

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, list_ys_temp, third_dimension=False):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, self.dim[0], self.dim[1], self.dim[2]))
        y = np.empty((self.batch_size, self.n_channels, self.dim[0], self.dim[1], self.dim[2]))
        # Need to split larger image into slices
       
        x_slice_idx = 0
        y_slice_ix = 0
        num_slices = 60
        # Generate data
        # Num Slcies
        for i, ID in enumerate(list_IDs_temp):
            # Load scan data for raw scan & mask
            if third_dimension:
                # set channels as first dimension for 3D Unet.
              
                x_data = np.swapaxes(nib.load(ID).get_data().astype(np.float32), 0, -1)
                y_data = np.swapaxes(nib.load(list_ys_temp[i]).get_data().astype(np.float32), 0, -1)
                if x_data.shape[0] < self.dim[0]:
                    x_data = self.resize_image(x_data) 
                    y_data = self.resize_image(y_data)

                x_data = self.normalizeImg(x_data) 
                y_data = self.normalizeImg(y_data) 
                
                X[x_slice_idx,] = x_data
                y[y_slice_ix,] = y_data

                x_slice_idx += 1
                y_slice_ix += 1

            else:

                for x_idx in range(x_data.shape[2]):
                    X[xslice_idx,] = x_data[:, :, x_idx, np.newaxis]
                    xslice_idx+=1
                    if xslice_idx == num_slices:
                        break

                
                for y_idx in range(y_data.shape[2]):
                    y[yslice_idx,] = y_data[:, :, y_idx, np.newaxis]
                    yslice_idx+=1
                    if yslice_idx == num_slices:
                        break 
                if xslice_idx == num_slices and yslice_idx == num_slices:
                    break 
        x1 = X[:x_slice_idx, :, :, :].astype(np.float32)
        y1 = y[:y_slice_ix, :, :, :].astype(np.float32)
        return x1, y1
      
