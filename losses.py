from keras import backend as K
from metrics import dice_coefficient

''' Dice Ceofficient Loss '''
def dice_coef_loss(y_true, y_pred):
    return 1-dice_coefficient(y_true, y_pred)
    
