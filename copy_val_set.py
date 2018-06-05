from __future__ import division

import re
import os
from random import shuffle
from math import floor 
import nibabel as nib
import numpy as np
import random
from shutil import copyfile

from data_loader import *


# Paths
def copy_val_set(slice_type):
    input_dir = './slice_data_' + slice_type 
    output_dir = './slice_data_' + slice_type + '_val'
    # Load paths to val files
    val_fp = ('./val_logs/val_' + slice_type + '.txt') 
    val_fns = list(np.genfromtxt(val_fp, dtype='str'))
    # Generate pairs of (input,masks) to move over
    side_val_files = []
    for input_patient in val_fns:
        mask_patient = input_patient.replace((slice_type+'_'), (slice_type+'_mask_'))
        side_val_files.append((input_patient, mask_patient))
    # slice_data_side val
    print('Creating ' + slice_type + ' val...')
    for side_file_x,side_file_y in side_val_files:	
        val_path_y = side_file_y.replace(input_dir, output_dir)	
        val_path_x = side_file_x.replace(input_dir, output_dir)
        # Relocate x
        copyfile(side_file_x, val_path_x)	
        # Relocate y
        copyfile(side_file_y, val_path_y)	
        print(val_path_x)
        print(val_path_y)
        print('------------------')

# Copy the val set over for different sides
#copy_val_set('side')
#copy_val_set('back')
copy_val_set('top')
