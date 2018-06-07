from data_loader import *
import nibabel as nib
import re
import os

output_fp = './3D_test'
test_fp = './test_set_mri_new/test_set_mri'
test_files = get_file_list_from_dir(test_fp, y_label='_defaced')

for patient_fn,_ in test_files:
    # Load patient data to get dims
    patient_data = nib.load(patient_fn).get_data()
    X,Y,Z = patient_data.shape 
    # Extract patient ID
    patient_id = re.search('\./test_set_mri_new/test_set_mri/(.*)\.nii',
                           patient_fn).group(1)
    print('-------- Patient Details --------')
    print(X,Y,Z)
    print(patient_id)
    
    pad = (X-Z)//2

    
    # Allocate tensor
    patient_output = np.zeros((X,Y,Z))
    
    ### Side Slices ###
    side_fp = './slice_data_side_test_pred/'
    # Iterate through slices
    for z_idx in range(Z):
        # Construct path
        side_slice_fp = (side_fp +
                         patient_id +
                         '_side_mask_pred_' + 
                         str(z_idx) +
                         '.nii')
        # Read data
        side_data = nib.load(side_slice_fp).get_data()
        # Push to patient output
        patient_output[:,:,z_idx] = side_data[:,:,0]
        

    # Top Slices
    top_fp = './slice_data_top_test_pred/'
    # Iterate through slices
    for x_idx in range(X):
        # Construct path
        top_slice_fp = (top_fp +
                         patient_id +
                         '_top_mask_pred_' + 
                         str(x_idx) +
                         '.nii')
        # Read data
        top_data = nib.load(top_slice_fp).get_data()
        # Push to patient output
        patient_output[x_idx,:,:] = top_data[:,pad:-pad,0]

    # Back Slices
    back_fp = './slice_data_back_test_pred/'
    # Iterate through slices
    for y_idx in range(Y):
        # Construct path
        back_slice_fp = (back_fp +
                         patient_id +
                         '_back_mask_pred_' + 
                         str(y_idx) +
                         '.nii')
        # Read data
        back_data = nib.load(back_slice_fp).get_data()
        # Push to patient output
        patient_output[:,y_idx,:] = top_data[:,pad:-pad,0]
    
    ### Save Patient Data ###
    output_fn = (patient_id + '_mask_pred.nii')
    img_nii = nib.Nifti1Image(patient_output, np.eye(4))
    img_nii.to_filename(os.path.join(output_fp, output_fn))
    break