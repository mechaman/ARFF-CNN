# Force Keras to use CPU
import os
# Enable to run on cpu only
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, concatenate, Conv2DTranspose
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback, CSVLogger
from keras import backend as keras
from keras.utils.training_utils import multi_gpu_model
from keras.backend import binary_crossentropy as bce
import tensorflow as tf

from metrics import dice_coef

def weightedLoss(originalLossFunc, weightsList):

    def lossFunc(true, pred):

        axis = -1 #if channels last 
        #axis=  1 #if channels first


        #argmax returns the index of the element with the greatest value
        #done in the class axis, it returns the class index    
        classSelectors = K.argmax(true, axis=axis) 

        #considering weights are ordered by class, for each class
        #true(1) if the class index is equal to the weight index  
        # c
        classSelectors = [K.equal(tf.cast(i, tf.int64), tf.cast(classSelectors, tf.int64)) for i in range(len(weightsList))]

        #casting boolean to float for calculations  
        #each tensor in the list contains 1 where ground true class is equal to its index 
        #if you sum all these, you will get a tensor full of ones. 
        classSelectors = [K.cast(x, K.floatx()) for x in classSelectors]

        #for each of the selections above, multiply their respective weight
        weights = [sel * w for sel,w in zip(classSelectors, weightsList)] 

        #sums all the selections
        #result is a tensor with the respective weight for each element in predictions
        weightMultiplier = weights[0]
        for i in range(1, len(weights)):
            weightMultiplier = weightMultiplier + weights[i]
        

        #make sure your originalLossFunc only collapses the class axis
        #you need the other axes intact to multiply the weights tensor
        loss = originalLossFunc(true,pred) 
        loss = loss[:,:,:,0] * weightMultiplier

        return loss
    return lossFunc
        
class myUnet(object):

    def __init__(self, img_rows = 256, img_cols = 256, custom=False):
        self.img_rows = img_rows
        self.img_cols = img_cols
        self.custom = custom

    def get_unet(self):

        inputs = Input((self.img_rows, self.img_cols,1))

        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = concatenate([drop4, up6], axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        #merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
        merge7 = concatenate([conv3, up7], axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        #merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
        merge8 = concatenate([conv2, up8], axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        #merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
        merge9 = concatenate([conv1, up9], axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        model = Model(input = inputs, output = conv10)
        # Custom bce
        chosen_loss = bce
        if self.custom:
            weights = [4, 1]
            chosen_loss = weightedLoss(bce, weights)
        model.compile(optimizer = Adam(lr = 1e-4), loss = chosen_loss)#'binary_crossentropy')#, metrics = ['accuracy'])#dice_coef])
        
        return model
    
    def predict(self, model, test_gen, y_test_fn, view, set_type):
         
        for i in range(len(test_gen)):
            # Temp:

            x_batch,y_batch = test_gen[i]
            predicted_mask = model.predict_on_batch(x=x_batch) 

            # Save predicted masks
            self.save_predictions([y_test_fn[i]],
                                  predicted_mask,
                                  view=view,
                                  set_type=set_type)
            
    def save_predictions(self, file_names, predictions, view, set_type):
        # Iterate through filenames and save predictions
        for idx, fn in enumerate(file_names):
            pred_fn = fn.replace('mask', 'mask_pred')
            pred_fp = pred_fn.replace((set_type + '/'), (set_type+'_pred/'))
            #print('predicted fn : ', pred_fp)
            self.save_img(predictions[idx], fn = pred_fp)
            print(pred_fp, ' saved!') 

    def save_img(self, img, fp = '.', fn = 'd_mask.nii'):
        img_nii = nib.Nifti1Image(img, np.eye(4))
        img_nii.to_filename(os.path.join(fp, fn))
        return True

if __name__ == '__main__':
    myunet = myUnet()
    model = myunet.get_unet()
    print(model.summary())


