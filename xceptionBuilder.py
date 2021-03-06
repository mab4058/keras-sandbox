'''
Xception Builder

References:
Xception: Deep Learning with Depthwise Separable Convolutions [https://arxiv.org/pdf/1610.02357.pdf]
Based on original code by F. Chollet [https://github.com/fchollet/keras/blob/master/keras/applications/xception.py]

@author: mbayer
'''
import os

from keras import backend as K
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.pooling import GlobalAveragePooling2D
from keras.models import Model

from buildingBlocks.xception_building_blocks import xceptionEntryFlow, xceptionMiddleFlow, xceptionExitFlow


def build(input_shape, num_outputs, repetitions, dropout_rate=None):
    """Builder function to create Xception model architecture.
    # Arguments
    input_shape: tuple, (n_rows, n_cols, n_channels)
        num_outputs: int, The number of outputs from the model.
        repetitions: int, The repetition amounts for the middle flow.
        dropout_rate: float, optional: Number in range 0-1 for dropout. Enter none
                      to ignore.
    
    # Returns
        Output model architecture.
    """
    if K.image_data_format() == 'channels_first':
        input_shape = (input_shape[2], input_shape[0], input_shape[1])
        
    input_layer = Input(input_shape)
    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(input_layer)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)

    x = xceptionEntryFlow(x)
    
    for i in range(repetitions):
        x = xceptionMiddleFlow(x, i)
        
    x = xceptionExitFlow(x)
    
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
        
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
        
    x = Dense(num_outputs, activation='softmax', name='out_fc')(x)

    model = Model(input_layer, x, name='xception')
    
    model.summary()
    
    return model

def saveModel(filename, model):
    modelContents = model.to_json()
    
    # Keras plot_model requires pydot and graphviz to be installed.
    h, t = os.path.split(filename)
    n, _ = os.path.splitext(t)
    plot_file = os.path.join(h, n + '.png')
    from keras.utils import plot_model
    plot_model(model, to_file=plot_file, show_shapes=True)
    
    with open(filename, 'w+') as f:
        f.write(modelContents)

class XceptionBuilder(object):
    @staticmethod
    def buildXception(input_shape, num_outputs, filename=None):
        model = build(input_shape, num_outputs, 8)
        if filename is not None:
            saveModel(filename, model)
        return model


if __name__=='__main__':
    m = build((400,600,3), 3, 4, dropout_rate=0.3)
    saveModel(r'C:\Users\mbayer\Desktop\Keras_Models\xceptionModel.json', m)