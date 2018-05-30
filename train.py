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
                    verbose=1,
                    epochs=70,
                    callbacks=[model_checkpoint],
                    use_multiprocessing=True, 
                    workers=6)
	model.save_weights('unet_3d_binary_cross_entropy.hdfs')

	print('Predicting ...')
	predict = model.predict_generator(generator=training_generator)
        self.save_img(predict[2], fn='d_mask2.nii')
        self.save_img(predict[1], fn='d_mask1.nii')
        self.save_img(predict[0], fn='d_mask0.nii')
        for img_name in (partition['x_train'])[0:5]:
        	print(img_name)












if __name__ == '__main__':
	train() 
