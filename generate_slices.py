from __future__ import division

import os
from random import shuffle
from math import floor 
import nibabel as nib
import numpy as np

from data_loader import *

def getImgData(fn):
    return(nib.load(fn).get_data())

def getMaskData(normal, defaced):
    normalized_norm = ((normal - np.min(defaced))/
                      (np.max(normal)-np.min(defaced)))
    delta = defaced - normalized_norm
    delta[delta >= 0] = 1
    delta[delta < 0] = 0
    return(delta.astype(np.int16))
    

def writeSlices(img_data,
                output_fp, 
                mask=False,
                view=[True,True,True]):
    # Collect image dimensions
    t_dim, b_dim, s_dim = img_data.shape
    # fp Mask or Original
    if not mask:
        img_data = img_data.astype(np.int32)
    # Output slices for side
    if view[0]:
        suffix = '_side_mask_' if mask else '_side_' 
        for s_idx in range(s_dim):
            side_slice = img_data[:, :, s_idx]
            # Construct final output path
            side_output_fp = (output_fp.replace('.nii', '') + 
                              suffix + str(s_idx) + '.nii')
            # Write slice
            (nib.Nifti1Image(side_slice, np.eye(4)).
             to_filename(side_output_fp))
        
    # Output slices for back
    if view[1]:
        suffix = '_back_mask_' if mask else '_back_' 
        for b_idx in range(b_dim):
            back_slice = img_data[:, b_idx, :]
            # Construct final output path
            back_output_fp = (output_fp.replace('.nii', '') + 
                              suffix + str(b_idx) + '.nii')
            # Write slice
            (nib.Nifti1Image(back_slice, np.eye(4)).
             to_filename(back_output_fp))
        
    # Output slices for top
    if view[2]:
        suffix = '_top_mask_' if mask else '_top_' 
        for t_idx in range(t_dim):
            top_slice = img_data[t_idx, :, :]
            # Construct final output path
            top_output_fp = (output_fp.replace('.nii', '') + 
                              suffix + str(t_idx) + '.nii')
            # Write slice
            (nib.Nifti1Image(top_slice, np.eye(4)).
             to_filename(top_output_fp))
        
    return True
        

### File Management ###
input_data_dir = './test_set_mri_new/test_set_mri'
output_data_dir = './slice_data_side_test'
# total_files : list(tuple(normal, defaced))
total_files = get_file_list_from_dir(input_data_dir, y_label='_defaced')
print(len(total_files))
i = 0
view = [True, False, False]
### Slicing & Outputing Files ###
for file in total_files:
    normal = file[0]
    defaced = file[1]
    print('Patient: ', normal)
    # Output file path
    output_fp = normal.replace(input_data_dir, output_data_dir)
    # Check if normal and defaced image exist
    if (not os.path.isfile(normal) or not os.path.isfile(defaced)):
        print(normal, 'or', defaced, 'doesn\'t exist?')
        continue
    # Read img & defaced data
    norm_data = getImgData(normal)
    def_data = getImgData(defaced)
    # Gen. mask data
    mask_data = getMaskData(norm_data,
                            def_data)
    ## Write Slices of Original & Mask data
    if writeSlices(norm_data, output_fp, mask=False, view=view):
        print('Normal Success! : ', output_fp)
    
    if writeSlices(mask_data, output_fp, mask=True, view=view):
        print('Mask Success! : ', output_fp)
    
    
