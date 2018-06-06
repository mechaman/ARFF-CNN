import numpy as np

''' Dice Coefficient '''
def dice_coef(mask_true, mask_pred, threshold = 0.5):
	# Use threshold to create a mask of 1s and 0s 
	mask_pred[mask_pred >= threshold] = 1.0
	mask_pred[mask_pred < threshold] = 0.0
	# Invert the values so mask is 1 else 0
	inv_mask_true = 1.0-mask_true
	inv_mask_pred = 1.0-mask_pred
	# Compute Dice Coeff
	d = ((2.0 * np.sum( inv_mask_pred[inv_mask_true == 1.0]) /
		(np.sum(inv_mask_true) + np.sum(inv_mask_pred) + 1e-9)))
	return d  

''' Dice (Jaccard)  Coefficient Metric '''
def dice_coef_alt(y_true, y_pred, smooth=1):
    """
    Dice = (2*|X & Y|)/ (|X|+ |Y|)
         =  2*sum(|A*B|)/(sum(A^2)+sum(B^2))
    ref: https://arxiv.org/pdf/1606.04797v1.pdf
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true),-1) + K.sum(K.square(y_pred),-1) + smooth)
