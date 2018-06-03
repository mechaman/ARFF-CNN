from __future__ import division

import os
from random import shuffle
from math import floor 
import nibabel as nib
import numpy as np
import random

from data_loader import *

# slice_data_side test
print('Creating side test...')
input_data_dir = './slice_data_side'
output_data_dir = './slice_data_side_test'
side_files = get_file_list_from_dir(input_data_dir, y_label='_mask')
sampled_side_files = random.sample(side_files, 3500)
for side_file in sampled_side_files:
	# Relocate x
	test_path_x = side_file[0].replace(input_data_dir, output_data_dir)
	os.rename(side_file[0], test_path_x)
	# Relocate y
	test_path_y = side_file[1].replace(input_data_dir, output_data_dir)	
	os.rename(side_file[1], test_path_y)	
	print(test_path_x)
	print(test_path_y)
	print('------------------')

# slice_data_top
print('Creating top test...')
input_data_dir = './slice_data_top'
output_data_dir = './slice_data_top_test'
top_files = get_file_list_from_dir(input_data_dir, y_label='_mask')
sampled_top_files = random.sample(top_files, 3500)
for top_file in sampled_top_files:
	# Relocate x
	test_path_x = top_file[0].replace(input_data_dir, output_data_dir)
	os.rename(top_file[0], test_path_x)
	# Relocate y
	test_path_y = top_file[1].replace(input_data_dir, output_data_dir)	
	os.rename(top_file[1], test_path_y)	
	print(test_path_x)
	print(test_path_y)
	print('------------------')

# slice_data_back test
print('Creating back test...')
input_data_dir = './slice_data_back'
output_data_dir = './slice_data_back_test'
back_files = get_file_list_from_dir(input_data_dir, y_label='_mask')
sampled_back_files = random.sample(back_files, 3500)
for back_file in sampled_back_files:
	# Relocate x
	test_path_x = back_file[0].replace(input_data_dir, output_data_dir)
	os.rename(back_file[0], test_path_x)
	# Relocate y
	test_path_y = back_file[1].replace(input_data_dir, output_data_dir)	
	os.rename(back_file[1], test_path_y)	
	print(test_path_x)
	print(test_path_y)
	print('------------------')
