''' Authors : Julien H. & Anish K.
'''

import os
from random import shuffle
import nibabel as nib
import numpy as np

def load_partitioned_data(data_dir, split, y_label='_mask'):
    ''' 
    Input -> data_dir : str, split : (float, float, float), y_label : str
    Load data ito train, val, test set w. specified split.
    '''
    total_files = get_file_list_from_dir(data_dir, y_label=y_label)
    shuffle(total_files) 
    train_split, val_split, test_split = data_split(total_files, split)
    x_train = [i[0] for i in train_split]
    y_train = [i[1] for i in train_split]
    x_val = [i[0] for i in val_split]
    y_val = [i[1] for i in val_split]
    x_test = [i[0] for i in test_split]
    y_test = [i[1] for i in test_split]
    return x_train, y_train, x_val, y_val, x_test, y_test

def get_file_list_from_dir(data_dir, y_label='_mask'):
    '''
    Input -> data_dir : str, y_label : str
    Load filenames from the specified directory.
    '''
    total_files = []
    for _,_,files in os.walk(data_dir):
        for file in files:
            if y_label[1:] in file:
                non_defaced = file.replace(y_label, '')
                defaced = file
                total_files.append((data_dir + '/' + non_defaced, data_dir + '/' + defaced))
        break
    return total_files

def data_split(file_list, split):
    '''
    Input -> file_list : [str], split : (float, float, float)
    Split data fn into appropriate data set. 
    '''	
    train, val, test = split
    train_end = train / 100
    val_end = train_end + (val / 100)
    test_start = val_end 
    number_of_files = len(file_list)

    train_split = file_list[:int(train_end * number_of_files)]
    val_split = file_list[int(train_end * number_of_files):int(number_of_files*val_end)]
    test_split = file_list[int(number_of_files*test_start):]

    return train_split, val_split, test_split
