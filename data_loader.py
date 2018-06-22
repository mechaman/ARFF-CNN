from __future__ import division

""" Make sure your MRI_Images directory is in the same directory as this file! """

import os
from random import shuffle
from math import floor 
import nibabel as nib
import numpy as np

def load_data(data_directory, split):
	"""Load the data into train, dev, and test set with the specified split. 
	Split should be a tuple of three percentages (Train%, Dev%, Test%) """ 

	total_files = get_file_list_from_dir(data_directory, y_label='_mask')
	shuffle(total_files) 
	train_split, val_split, test_split = data_split(total_files, split)
	X_train = [i[0] for i in train_split]
	y_train = [i[1] for i in train_split]
	X_val = [i[0] for i in val_split]
	y_val = [i[1] for i in val_split]
	X_test = [i[0] for i in test_split]
	y_test = [i[1] for i in test_split]
	return X_train, y_train, X_val, y_val, X_test, y_test

def get_file_list_from_dir(datadir, y_label='_mask'):
    """load data from the specified datadir"""
    print('updated')
    total_files = []
    for _,_,files in os.walk(datadir):
        for file in files:
            if y_label[1:] in file:
                non_defaced = file.replace(y_label, '')
                defaced = file
                total_files.append((datadir + '/' + non_defaced, datadir + '/' + defaced))
        break
    return total_files



def data_split(file_list, split):
    train, val, test = split
    train_end = train / 100
    val_end = train_end + (val / 100)
    test_start = val_end 
    number_of_files = len(file_list)

    train_split = file_list[:int(train_end * number_of_files)]
    val_split = file_list[int(train_end * number_of_files):int(number_of_files*val_end)]
    test_split = file_list[int(number_of_files*test_start):]

    return train_split, val_split, test_split



class Dataset:
    def __init__(self, X, y, batch_size):
        self.X = X
        self.y = y 
        self.batch_size = batch_size

    def __iter__(self):
        N, B = len(self.X), self.batch_size
        idxs = np.arange(N)
        for i in range(0, N, B):
            batch_X = self.X[i:i+B]
            batch_y = self.y[i:i+B]
            # @DEBUG
            if i == 3:
                print(batch_y[0])
            batch_X = [nib.load(X_data) for X_data in batch_X]
            #print('Loaded X batch...')
            batch_y = [nib.load(y_data) for y_data in batch_y]
            #print('Loaded Y batch...')
            batch_X = [img.get_data() for img in batch_X]
            batch_y = [img.get_data() for img in batch_y]
            yield (np.array(batch_X), np.array(batch_y))


    def __len__(self):
        return len(self.X) 

'''
#example
X_train, y_train, X_val, y_val, X_test, y_test  = load_data('data', split=(90,5,5))

#example
train_dset = Dataset(X_train, y_train, batch_size=1) 
val_dset = Dataset(X_val, y_val, batch_size=1)
test_dset = Dataset(X_val, y_val, batch_size=1)
'''

