import keras
import numpy as np
import nibabel as nib
from pathlib import Path



class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, examples, labels, batch_size=1, dim=(256,256,256), n_channels=1,
                 n_classes=1, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = examples
        self.n_channels = n_channels
        self.shuffle = shuffle
        # Add an index var. to keep track of location
        self.index = 0
        self.on_epoch_end()
    '''
    def next(self):
        'Get the next element in the generator'
        return self.__getitem__(self.index)
    '''
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        #print(len(indexes))
        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        list_ys_temp = [self.labels[k] for k in indexes]
        #print(len(list_ys_temp))        
        # Generate data
        X, y = self.__data_generation(list_IDs_temp, list_ys_temp)
        # Update index
        self.index+=1
        return X, y, list_IDs_temp

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
            
    def normalizeImg(self, x):
        # Normalize x
        max_val = np.max(x)
        min_val = np.min(x)
        norm_x = (x-min_val)/(max_val - min_val + 1e-7)
        return norm_x

    def padImage(self, img, mask=False):
        '''
        Input -> img : numpy array, mask : Boolean
        ''' 
        #@TODO Remove padding image function since it ruins image
        # And not necessary anymore since we are resampling
        img_c = img.copy()
        # Get max dim
        dims = img.shape
        max_dim = max(dims)
        # Padding Args.
        pad_value = 0
        pad_arr = [(0,0), (0,0), (0,0)]
        # Check if mask or input to pad w. 1 or 0
        if mask:
            pad_value = 1
        # Check if other dims eq. largest dim
        for idx, dim in enumerate(dims):
            if dim != max_dim:
                pad_w = max_dim - dim
                pad_arr[idx] = (pad_w//2, pad_w//2)
                img_c = np.pad(img, pad_arr,
                                mode='constant', constant_values=(pad_value))

        return img_c

                
    def __data_generation(self, list_IDs_temp, list_ys_temp):
        'Generates data containing batch_size samples'
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        y = np.empty((self.batch_size, *self.dim))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # @TODO remove when file name consistent
            ID = ID.replace('.gz', '')
            # Check if file exists, continue if not
            if not Path(ID).is_file():
                print(ID, ' is missing.') 
                continue
            # Store sample
            X[i, ] = self.padImage(self.normalizeImg(nib.load(ID).get_data().astype(np.float32)))
            # Store sample segmentation
            y[i, ]  = self.padImage((nib.load(list_ys_temp[i]).get_data().astype(np.float32)), mask=True)

        return X, y

