import unet_three_d
from keras import backend as K
from keras.models import *
from data_loader import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from data_generation import DataGenerator
import nibabel as nib
import pdb 
import tensorflow as tf
import os
from metrics import dice_coefficient
import argparse
import time

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--evaluate', default=False, action='store_true') 


args = parser.parse_args()

def save_img(img, fn = 'd_mask.nii'):
	img_nii = nib.Nifti1Image(img, np.eye(4))
	img_nii.to_filename(os.path.join('.', fn))
	return True

def save_prediction(file_names, predictions):
	for idx, fn in enumerate(file_names):
		pred_fn = '3dpreds/' + os.path.basename(os.path.normpath(os.path.splitext(fn)[0])) + '_pred.nii'  
		save_img(predictions[idx], fn=pred_fn) 
		print(pred_fn, 'saved.')



def predict(model, validation_generator, test_set):
	print('Predicting ...')
	for i in range(len(validation_generator)):
		if i == 5:
			break
		x_batch, _ = validation_generator[i] 
		start = time.time() 
		predicted_mask = model.predict_on_batch(x=x_batch)
		end = time.time() 
		print('Prediction time:', end - start) 
		save_prediction(test_set, predicted_mask)


def evaluate():
	partition = {}
	model = load_model('unet_regres.hdf5', custom_objects={'dice_coefficient': dice_coefficient}) 
	model.load_weights('unet_3d_regression.hdfs')
	(_,
	    _,
	    _,
	    _,
	    partition['x_test'],
	    partition['y_test'])  = load_data('data', split=(0,0,10), DEBUG=True, third_dimension=True)

	params = {
		'dim': (160,256,256),
        'batch_size': 1,
        'n_channels': 1,
        'shuffle': True,
            'third_dimension': True
     }

	testing_generator = DataGenerator(['data/IXI365-Guys-0923-T1.nii'], ['data/IXI365-Guys-0923-T1_defaced.nii'], **params)


	predict(model, testing_generator, ['data/IXI365-Guys-0923-T1.nii'])


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
	    partition['y_test'])  = load_data('data', split=(10,10,0), DEBUG=True, third_dimension=True)
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

		model_checkpoint = ModelCheckpoint('unet_regres.hdf5', monitor='loss',verbose=1, save_best_only=True)

		
		model.fit_generator(generator=training_generator,
	                    validation_data=validation_generator,
	                    #steps_per_epoch = 1,
	                    validation_steps = 1,
		  	    epochs=10,
			    callbacks = [model_checkpoint],
			    use_multiprocessing=True,
			    workers=6,
	                    verbose=1)
		model.save_weights('unet_3d_regression.hdfs')

		print('Predicting ...')
		predict(model, validation_generator, [partition['y_val'][i]])
#		predict = model.predict_generator(generator=training_generator)
#		pdb.set_trace()
		# for i in range(len(validation_generator)):
		# 	if i == 5:
		# 		break
		# 	x_batch, _ = validation_generator[i] 
		# 	predicted_mask = model.predict_on_batch(x=x_batch)
		# 	save_prediction([partition['y_val'][i]], predicted_mask)






if __name__ == '__main__':
	args = parser.parse_args()
	if args.evaluate:
		evaluate() 
	else:
		train() 
