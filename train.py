import unet_three_d
from keras import backend as K
from data_loader import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from data_generation import DataGenerator
import nibabel as nib
import pdb 
import tensorflow as tf
import os

def save_img(img, fn = 'd_mask.nii'):
	img_nii = nib.Nifti1Image(img, np.eye(4))
	img_nii.to_filename(os.path.join('.', fn))
	return True

def save_prediction(file_names, predictions):
	for idx, fn in enumerate(file_names):
		pred_fn = os.path.splitext(fn) + '_mask.nii'  
		save_img(predictions[idx], fn=pred_fn) 
		print(pred_fn, 'saved.')
def train():
	K.clear_session()
#	pdb.set_trace()
	partition = {}
	model = unet_three_d.unet((1,160,256,256))
	print(model.summary())
	with tf.device('/cpu:0'):
		(partition['x_train'],
	    partition['y_train'],
	    partition['x_val'],
	    partition['y_val'],
	    partition['x_test'],
	    partition['y_test'])  = load_data('data', split=(2,2,0), DEBUG=True, third_dimension=True)
#		print(partition['x_train'][0], partition['y_train'][0])
#		partition['x_train'] = [partition['x_train'][0]]
#		partition['y_train'] = [partition['y_train'][0]]
#		print(partition['x_train'], partition['y_train'])
		params = {
					'dim': (160,256,256),
		            'batch_size': 1,
		            'n_channels': 1,
		            'shuffle': True,
	                    'third_dimension': True
	             }

		training_generator = DataGenerator(partition['x_train'], partition['y_train'], **params)
		validation_generator = DataGenerator(partition['x_val'], partition['y_val'], **params)
	#	testing_generator = DataGenerator(partition['x_test'], partition['y_test'], **params)
		print('Loaded Data')

		print('Instantiate 3D-Unet') 

		model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)


		model.fit_generator(generator=training_generator,
	                    validation_data=training_generator,
	                    #steps_per_epoch = 1,
	                    validation_steps = 1,
		  	    epochs=10,
	                    verbose=1)
		model.save_weights('unet_3d_regression.hdfs')

		print('Predicting ...')
#		predict = model.predict_generator(generator=training_generator)
		for i in range(len(validation_generator)):
			x_batch, _ = validation_generator[i] 
			predicted_mask = model.predict_on_batch(x=x_batch)
			save_prediction([partition['y_val'][i]], predicted_mask)






if __name__ == '__main__':
	train() 
