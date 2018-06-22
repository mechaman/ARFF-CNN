from data_loader import *
import nibabel as nib
import re
import os

def writeProbVolume(vol, slice_type):
    ### Save Patient Data ###
    output_fn = (patient_id + '_mask_' +'.nii')
    img_nii = nib.Nifti1Image(vol, np.eye(4))
    img_nii.to_filename(os.path.join(output_fp, output_fn))
    
test_fp = './test_set_mri_new/test_set_mri'
test_files = get_file_list_from_dir(test_fp, y_label='_defaced')
for patient_fn,_ in test_files:
    