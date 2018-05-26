
from data_loader import *
import numpy as np
import nibabel as nib
import keras
import pdb


class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, examples, labels, batch_size=32, dim=(256,256), n_channels=1
                 , shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = examples
        self.n_channels = n_channels
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
        X, y = self.__data_generation(list_IDs_temp, list_ys_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, list_ys_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        pdb.set_trace()
        print('Generating...')
        # Initialization
        X = np.empty((150*self.batch_size ,*self.dim, self.n_channels))
        y = np.empty((150*self.batch_size, *self.dim, self.n_channels))
        #print('X shape: ', X.shape)
        # Need to split larger image into slices
        xslice_idx = 0
        yslice_idx = 0
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Load scan data for raw scan & mask
            x_data = nib.load(ID).get_data().astype(np.float32)
            #print(x_data.shape)
            y_data = nib.load(list_ys_temp[i]).get_data().astype(np.float32)
            # Iterate through slices
            for x_idx in range(x_data.shape[2]):
                X[xslice_idx,] = x_data[:, :, x_idx, np.newaxis]
                xslice_idx+=1
            
            for y_idx in range(y_data.shape[2]):
                y[yslice_idx,] = y_data[:, :, y_idx, np.newaxis]
                yslice_idx+=1
        # Return view of data (slice a larger array than there is info) 
        # bs*150,  256, 256, 1
        x1 = X[:xslice_idx, :, :, :].astype(np.float32)
        y1 = y[:yslice_idx, :, :, :].astype(np.float32)
        print(x1.dtype)
        return x1, y1
