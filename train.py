import unet
from keras import backend as K
from data_loader import *
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
    partition['y_test'])  = load_data('data', split=(90,5,5), DEBUG=True)
	print(partition['x_train'])
	print(partition['y_train'])


	params = {
				'dim': (256,256,150),
	            'batch_size': 1,
	            'n_channels': 1,
	            'shuffle': True
             }

	training_generator = DataGenerator(partition['x_train'], partition['y_train'], **params)
	model = unet.UNet3D() 
 #    validation_generator = DataGenerator(partition['x_val'], partition['y_val'], **params)
 #    testing_generator = DataGenerator(partition['x_test'], partition['y_test'], **params)

















if __name__ == '__main__':
	train() 