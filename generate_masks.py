from __future__ import division

import os
from random import shuffle
from math import floor 
import nibabel as nib
import numpy as np

from data_loader import *

data_dir = './data'
total_files = get_file_list_from_dir(data_dir)
i = 0
for file in total_files:
    if i == 1:
        break
    # execute for 1 image (remove if over full data set)
    # Set file names
    normal = file[0]
    defaced = file[1]
    # Create mask image filename
    mask_fn = normal.replace('.nii','') + '_mask.nii'
    # Check if mask already created
    if os.path.isfile(mask_fn):
        continue
    # Check if normal/defaced file doesn't exist
    if (not os.path.isfile(normal) or not os.path.isfile(defaced)):
        print(normal, 'or', defaced, 'doesn\'t exist?')
        continue
    # Load nii files
    normal_nii = nib.load(normal)
    defaced_nii = nib.load(defaced)
    # Extract image data
    normal_img = normal_nii.get_data()
    defaced_img = defaced_nii.get_data()
    # defaced - normal - negative values where mask is located
    delta_img = defaced_img - normal_img
    # Set all other pixels not part of mask to 1
    delta_img[delta_img >= 0] = 0
    # Set pixels part of mask to 0
    delta_img[delta_img < 0] = 1
    # Create mask image and save
    mask_fn = normal.replace('.nii','') + '_mask.nii'
    print(mask_fn)
    mask_img = nib.Nifti1Image(delta_img, normal_nii.affine)
    mask_img.to_filename(mask_fn)
    i+=1