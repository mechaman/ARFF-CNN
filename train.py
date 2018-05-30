import unet
from keras import backend as K
from data_loader import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from data_generation import DataGenerator
import pdb 

def train():
	K.clear_session()
	partition = {}
	(partition['x_train'],
    partition['y_train'],
    partition['x_val'],
    partition['y_val'],
    partition['x_test'],
    partition['y_test'])  = load_data('data', split=(10,10,80), DEBUG=True, third_dimension=True)


	params = {
				'dim': (150,256,256),
	            'batch_size': 1,
	            'n_channels': 1,
	            'shuffle': True,
                    'third_dimension': True
             }

	training_generator = DataGenerator(partition['x_train'], partition['y_train'], **params)
	validation_generator = DataGenerator(partition['x_val'], partition['y_val'], **params)
	testing_generator = DataGenerator(partition['x_test'], partition['y_test'], **params)
	training_generator[1]
	print('Loaded Data')


	print('Instantiate 3D-Unet') 
	model = unet.unet()
	print(model)

	model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)


	model.fit_generator(generator=training_generator,
                    validation_data=training_generator,
                    #steps_per_epoch = 1,
                    validation_steps = 1,
                    epochs=70,
                    verbose=0)
       model.save_weights('unet_3d_binary_cross_entropy.hdfs')
















if __name__ == '__main__':
	train() 
