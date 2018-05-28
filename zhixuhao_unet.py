import os 
#os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from keras.models import *
from keras.layers import Input, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
#from data_generation import DataGenerator
from data_loader import *
from data_generator import DataGenerator 

class myUnet(object):

    def __init__(self, img_rows = 256, img_cols = 256):

        self.img_rows = img_rows
        self.img_cols = img_cols

    def load_data(self):

        mydata = dataProcess(self.img_rows, self.img_cols)
        imgs_train, imgs_mask_train = mydata.load_train_data()
        imgs_test = mydata.load_test_data()
        return imgs_train, imgs_mask_train, imgs_test

    def get_unet(self):

        inputs = Input((self.img_rows, self.img_cols,1))

        '''
        unet with crop(because padding = valid) 

        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(inputs)
        print "conv1 shape:",conv1.shape
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv1)
        print "conv1 shape:",conv1.shape
        crop1 = Cropping2D(cropping=((90,90),(90,90)))(conv1)
        print "crop1 shape:",crop1.shape
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        print "pool1 shape:",pool1.shape

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool1)
        print "conv2 shape:",conv2.shape
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv2)
        print "conv2 shape:",conv2.shape
        crop2 = Cropping2D(cropping=((41,41),(41,41)))(conv2)
        print "crop2 shape:",crop2.shape
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        print "pool2 shape:",pool2.shape

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool2)
        print "conv3 shape:",conv3.shape
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv3)
        print "conv3 shape:",conv3.shape
        crop3 = Cropping2D(cropping=((16,17),(16,17)))(conv3)
        print "crop3 shape:",crop3.shape
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        print "pool3 shape:",pool3.shape

        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        crop4 = Cropping2D(cropping=((4,4),(4,4)))(drop4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = merge([crop4,up6], mode = 'concat', concat_axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = merge([crop3,up7], mode = 'concat', concat_axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = merge([crop2,up8], mode = 'concat', concat_axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = merge([crop1,up9], mode = 'concat', concat_axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'valid', kernel_initializer = 'he_normal')(conv9)
        '''
	#downsampling portion of unet"

        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        #print "conv1 shape:",conv1.shape
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        #print "conv1 shape:",conv1.shape
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        #print "pool1 shape:",pool1.shape

        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        #print "conv2 shape:",conv2.shape
        conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
        #print "conv2 shape:",conv2.shape
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
        #print "pool2 shape:",pool2.shape

        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        #print "conv3 shape:",conv3.shape
        conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        #print "conv3 shape:",conv3.shape
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
        #print "pool3 shape:",pool3.shape

        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
        drop4 = Dropout(0.5)(conv4)
        pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv5 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        drop5 = Dropout(0.5)(conv5)

	#upsampling portion of unet
        up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
        merge6 = merge([drop4,up6], mode = 'concat', concat_axis = 3)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
        conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

        up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
        merge7 = merge([conv3,up7], mode = 'concat', concat_axis = 3)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
        conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

        up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
        merge8 = merge([conv2,up8], mode = 'concat', concat_axis = 3)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
        conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)

        up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
        merge9 = merge([conv1,up9], mode = 'concat', concat_axis = 3)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
        conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
	#binary sigmoid output for masking.
        conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

        model = Model(input = inputs, output = conv10)

        model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

        return model
    
    
    def expandDataSet(self, data, samples=1):
        ''' Input : data -> x, y
            Output : x, y
            Takes in an iterator and iterates through each image to make
            each slice its own example. Each image is (256, 256, 150) and
            the output data set will be (Number_Of_Images, 256, 256, 1) where
            the Number_Of_Images = original number of images x 150. 
        '''
        # hardcode 150 for now
        number_images = len(data)*150
        x_list = []
        y_list = []
        i = 0
        for imgs_in, imgs_lab in data:
            if i == samples:
                break
            #print(imgs_in.shape)
            x = imgs_in[0, :, :, None, 74]
            y = imgs_lab[0, :, :, None, 74]
            # Normalize
            #x,y = self.normalizeImg(x,y)
            x_list.append(x)
            y_list.append(y)
            i+=1
        return np.array(x_list),np.array(y_list)
    
    def train3(self):
        print("Loading Data.")
        # Partition data : x_train, y_train, ... , x_test, y_test
        partition = {}
        (partition['x_train'],
         partition['y_train'],
         partition['x_val'],
         partition['y_val'],
         partition['x_test'],
         partition['y_test'])  = load_data('slice_data_side', split=(90,5,5))
        print('shape of training x :' , len(partition['x_train']))
        
        

        # Parameters
        params = {'dim': (256,256),
                  'batch_size': 1,
                  'n_channels': 1,
                  'shuffle': True}
        training_generator = DataGenerator(partition['x_train'], partition['y_train'], **params)
        validation_generator = DataGenerator(partition['x_val'], partition['y_val'], **params)
        #testing_generator = DataGenerator(partition['x_test'], partition['y_test'], **params)
        
        print(len(training_generator))
        
        print('Loaded Data')
        
        print("Instantiate UNET")
        # Check if checkpoint exists
        model = load_model('unet.hdf5') 
        #else:
        #model = self.get_unet()
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1,
                                           save_best_only=True)
        
        print('Fitting Model...')
        model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    #steps_per_epoch = 1,
                    validation_steps = 1,
                    epochs=1,
                    verbose=1,
                    callbacks =[model_checkpoint],
                    use_multiprocessing=True,
                    workers=6)
        # Save weights
        #model.save_weights('unet_weights.hdf5')
        
        print('Predicting w. Model...')
        predict = model.predict_generator(generator=training_generator)
        print(np.min(predict[1]), np.max(predict[1]))
        self.save_img(predict[1])
        #@DEBUG 
        for img_name in partition['x_train']:
            print(img_name)
        
    def train2(self):
        from keras import backend as K
        K.clear_session()
        print("Loading Data.")
        # Partition data : x_train, y_train, ... , x_test, y_test
        pdb.set_trace()
        partition = {}
        (partition['x_train'],
         partition['y_train'],
         partition['x_val'],
         partition['y_val'],
         partition['x_test'],
         partition['y_test'])  = load_data('data', split=(1,1,98))
        

        print(len(partition['x_train']))
        print(len(partition['y_train']))
        # Parameters
        params = {'dim': (256,256),
                  'batch_size': 1,
                  'n_channels': 1,
                  'shuffle': True}

        # Generators
        training_generator = DataGenerator(partition['x_train'], partition['y_train'], **params)
        
        validation_generator = DataGenerator(partition['x_val'], partition['y_val'], **params)
        #testing_generator = DataGenerator(partition['x_test'], partition['y_test'], **params)
        print('Loaded Data')
        
        print("Instantiate UNET")
        model = self.get_unet()
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
        
        print('Fitting Model...')
        model.fit_generator(generator=training_generator,
                    validation_data=validation_generator,
                    steps_per_epoch = 1,
                    validation_steps = 1,
                    epochs=1,
                    verbose=1,
                    callbacks =[model_checkpoint],
                    use_multiprocessing=True,
                    workers=6)
        
         
    def train(self):
        print("Loading Data.")
        #(X_train, y_train, X_val, y_val, X_test, y_test)  = load_data('data', split=(90,5,5))
        #train_dset = Dataset(X_train, y_train, batch_size=1)
        #val_dset = Dataset(X_val, y_val, batch_size=1)
        #test_dset = Dataset(X_val, y_val, batch_size=1)
        
        ## Generate slices
       # x_train, y_train = self.expandDataSet(train_dset, samples=150)
        #x_val, y_val = self.expandDataSet(val_dset)
        #x_test, y_test = self.expandDataSet(test_dset)
        
        
        print("Finished Loading Data.")
        
        model = self.get_unet()
        print(model.summary())
        '''
        print("Instantiate UNET")
        
        model_checkpoint = ModelCheckpoint('unet.hdf5', monitor='loss',verbose=1, save_best_only=True)
        print('Fitting model...')
        
        print('Training Model...')

        model.fit(x_train, y_train, batch_size=1, nb_epoch=2, validation_data= (x_val, y_val), verbose=1, callbacks=[model_checkpoint])
        
        print('Trained Model')

        print('Predict Test Data')

        print('xtrain shape', x_train.shape)

        a_slice = x_train[3, None, :, :]
        print('shape : ', a_slice.shape)
        imgs_mask_test = model.predict(a_slice, batch_size=1, verbose=1)
        #t1 = time.time()
        #print(t1-t0)
        self.save_img(imgs_mask_test)
        '''

    def save_img(self, img, fn = 'd_mask.nii'):
        img_nii = nib.Nifti1Image(img, np.eye(4))
        img_nii.to_filename(os.path.join('.', fn))
        return True




if __name__ == '__main__':
    myunet = myUnet()
    myunet.train3()
    #myunet.save_img()


