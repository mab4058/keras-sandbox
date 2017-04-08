'''
Basic building blocks for model architectures.

@author: mbayer
'''
from keras import backend as K
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation

def conv2d_bn(x, 
              n_filters,
              kernel_size,
              strides=(1, 1),
              dilation_rate=(1,1),
              padding='same',   
              activation='relu',  
              name=None):
    """Conv2d + Batchnorm block.
    """
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
        
    if K.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    
    x = Conv2D(n_filters,
               kernel_size,
               strides=strides,
               padding=padding,
               name=conv_name)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name)(x)
    x = Activation(activation)(x)
    
    return x

if __name__ == '__main__':
    pass