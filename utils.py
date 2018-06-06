from metrics import dice_coef
from data_loader import *
import nibabel as nib

def computeAverageDice(thresh, pred_dir, gt_dir, n_samples = 500):
    '''
    1) Extract predicted masks from slice_data_side_pred
    2) Extract masks from slice_data_side
    3) Compute dice_coef
    4) Append to list
    5) return mean
    '''
    # Retrieve ground truth masks
    input_mask_pairs = get_file_list_from_dir(gt_dir, y_label='_mask')
    print(str(len(input_mask_pairs)) + ' in validation set.')
    # List of dice coefficients
    dc_list = []
    idx = 0
    for input_x, mask_y in input_mask_pairs:
        # Check if n_samples is reached
        if n_samples == idx:
            break
        # Construct predicted filepath
        mask_pred_fn = mask_y.replace('_mask', '_mask_pred')
        mask_pred_fp = mask_pred_fn.replace(gt_dir, pred_dir)
        # Read in files
        mask_gt = (nib.load(mask_y)).get_data()
        # If dir. doesn't contain side
        if 'side' not in pred_dir:
            mask_pred = ((nib.load(mask_pred_fp)).get_data())[:, :mask_gt.shape[1], :]
        else:
            mask_pred = (nib.load(mask_pred_fp)).get_data()
        # Dice Coefficient computation
        dc = dice_coef(mask_gt, mask_pred, threshold = thresh)
        dc_list.append(dc)
        # Compute dice coef.
        
        if idx%100 == 0:
            print('Avg. Dice Coefficient : ', np.mean(dc_list))
            #print(dc)
            print('Example Number', idx)
        
        idx+=1
        
    return(np.mean(dc_list))
   
        
        

avg_dc = computeAverageDice(0.5, './slice_data_side_val_pred', './slice_data_side_val', n_samples = 500)
