'''
Created on Apr 8, 2017

@author: mbayer
'''
from keras import backend as K
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator 
from keras.models import model_from_yaml, model_from_json, load_model
from keras.utils import np_utils
from keras.callbacks import CSVLogger

def cifar10(model_file):
    logger = CSVLogger('cifar10_training_log.csv')
    
    batch_size = 32
    nb_classes = 10
    nb_epoch = 200
    data_augmentation = True
    
#     # input image dimensions
#     img_rows, img_cols = 32, 32
#     # The CIFAR10 images are RGB.
#     img_channels = 3
#     
#     if K.image_data_format() == 'channels_last':
#         im_dims = (img_rows, img_cols, img_channels)
#     else:
#         im_dims = (img_channels, img_rows, img_cols)
        
    # The data, shuffled and split between train and test sets:
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')
    
    # Convert class vectors to binary class matrices.
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    
    model = loadModel(model_file)
    
    
        # Let's train the model using RMSprop
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    X_train /= 255
    X_test /= 255
    
    if not data_augmentation:
        print('Not using data augmentation.')
        model.fit(X_train, Y_train,
                  batch_size=batch_size,
                  nb_epoch=nb_epoch,
                  validation_data=(X_test, Y_test),
                  shuffle=True,
                  callbacks=[logger])
    else:
        print('Using real-time data augmentation.')
        # This will do preprocessing and realtime data augmentation:
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images
    
        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(X_train)
    
        # Fit the model on the batches generated by datagen.flow().
        model.fit_generator(datagen.flow(X_train, Y_train,
                                         batch_size=batch_size),
                            samples_per_epoch=X_train.shape[0],
                            nb_epoch=nb_epoch,
                            validation_data=(X_test, Y_test),
                            callbacks=[logger])
    
    
def loadModel(filename):
    parts = filename.split('.')
    if parts[1].lower() == 'json':
        model = _loadModelDefJson(filename)
    else:
        model = _loadModelDefYaml(filename)
        
    return model

def _loadModelDefYaml(filename):
    with open(filename, 'r') as f:
        model = model_from_yaml(f.read())  
    return model

def _loadModelDefJson(filename):
    with open(filename, 'r') as f:
        model = model_from_json(f.read())  
    return model

if __name__ == '__main__':
    pass