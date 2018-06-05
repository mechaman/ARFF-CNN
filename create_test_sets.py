from __future__ import division

import re
import os
from random import shuffle
from math import floor 
import nibabel as nib
import numpy as np
import random

from data_loader import *

# Get the data list from test_set_mri
test_fp = './test_set_mri'
test_files = get_file_list_from_dir(test_fp, y_label='_defaced')
test_patients = {(patient[0].replace('.nii', '')).replace((test_fp+'/'),''):True for patient in test_files}

# slice_data_side test
print('Creating side test...')
input_data_dir = './slice_data_side'
output_data_dir = './slice_data_side_test'
side_files = get_file_list_from_dir(input_data_dir, y_label='_mask')
for side_file_x,side_file_y in side_files:	
	patient = (re.search('\./slice_data_side(.*)_side*', side_file_x).group(1))[1:]	
	if patient in test_patients:		
		test_path_y = side_file_y.replace(input_data_dir, output_data_dir)	
		test_path_x = side_file_y.replace(input_data_dir, output_data_dir)
		# Relocate x
		os.rename(side_file_x, test_path_x)
		# Relocate y
		os.rename(side_file_y, test_path_y)	
		print(test_path_x)
		print(test_path_y)
		print('------------------')

# slice_data_top test
print('Creating top test...')
input_data_dir = './slice_data_top'
output_data_dir = './slice_data_top_test'
top_files = get_file_list_from_dir(input_data_dir, y_label='_mask')
for top_file_x,top_file_y in top_files:	
	patient = (re.search('\./slice_data_top(.*)_top*', top_file_x).group(1))[1:]	
	if patient in test_patients:		
		test_path_y = top_file_y.replace(input_data_dir, output_data_dir)	
		test_path_x = top_file_y.replace(input_data_dir, output_data_dir)
		# Relocate x
		os.rename(top_file_x, test_path_x)
		# Relocate y
		os.rename(top_file_y, test_path_y)	
		print(test_path_x)
		print(test_path_y)
		print('------------------')

# slice_data_back test
print('Creating back test...')
input_data_dir = './slice_data_back'
output_data_dir = './slice_data_back_test'
back_files = get_file_list_from_dir(input_data_dir, y_label='_mask')
for back_file_x,back_file_y in back_files:	
	patient = (re.search('\./slice_data_back(.*)_back*', back_file_x).group(1))[1:]	
	if patient in test_patients:		
		test_path_y = back_file_y.replace(input_data_dir, output_data_dir)	
		test_path_x = back_file_y.replace(input_data_dir, output_data_dir)
		# Relocate x
		os.rename(back_file_x, test_path_x)
		# Relocate y
		os.rename(back_file_y, test_path_y)	
		print(test_path_x)
		print(test_path_y)
	print('------------------')
