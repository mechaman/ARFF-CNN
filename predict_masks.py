import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import numpy as np
from keras.optimizers import *
from keras.models import *
from keras.utils.training_utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint, Callback, CSVLogger
from keras.utils.training_utils import multi_gpu_model

from unet import myUnet
from data_loader import *
from data_generator import DataGenerator


def predict_mask(slice_type, set_type='val'):
    # Gen. file paths for weights
    model_prefix = 'zhi_unet_' + slice_type
    weights_fp = ('./weights/' + model_prefix + '.hdf5')
    # Gen. path for data
    data_dir = ('slice_data_' + slice_type + '_' + set_type)

    # Load the data into generator
    (_,
     _,
     _,
     _,
     x_test,
     y_test)  = load_data(data_dir, split=(0.0, 0.0, 100))

    # Parameters for input data
    params = {'dim': (256,256),
                  'batch_size': 1,
                  'n_channels': 1,
                  'shuffle': False}
    test_generator = DataGenerator(x_test, y_test, **params)
    print('Loaded Data...')

    # Initialize UNet
    print("Instantiate UNET")
    unet = myUnet()
    model = unet.get_unet_zhi()
    model.load_weights(weights_fp)
    # Predict using UNet
    print("Predicting via UNet")
    unet.predict(model, test_generator, y_test, slice_type, set_type)

# Run predict_mask over each slice_type
predict_mask(slice_type='side', set_type='test')
    
