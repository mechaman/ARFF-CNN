import keras
import numpy as np
import nibabel as nib



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
            
    def normalizeImg(self, x):
        # Normalize x
        max_val = np.max(x)
        min_val = np.min(x)
        norm_x = (x-min_val)/(max_val - min_val + 1e-7)
        return norm_x
    
    def padImage(self, img, mask=False):
        '''
        1) Find max dim. value and idx
        2) Check if autre dim. == to max dim
        3) True -> return orig. image
        4) False -> pad dims. != argmax(dims)
        5) Return padded image
        ''' 
        x_dim = img.shape[0]
        y_dim = img.shape[1]
        if x_dim == y_dim:
            return img
        else:
            pad_w = x_dim - y_dim
            if mask:
                return np.pad(img, [(0,0), (pad_w//2,pad_w//2), (0,0)],
                              mode='constant',
                              constant_values=(1))
            else:
                return np.pad(img, [(0,0), (pad_w//2,pad_w//2), (0,0)],
                              mode='constant')
                
                

    def __data_generation(self, list_IDs_temp, list_ys_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i, ] = self.padImage(self.normalizeImg((nib.load(ID).get_data().astype(np.float32))[:, :, np.newaxis]))
            # Store sample segmentation
            y[i, ]  = self.padImage((nib.load(list_ys_temp[i]).get_data().astype(np.float32))[:, :, np.newaxis], mask=True)

        return X, y

