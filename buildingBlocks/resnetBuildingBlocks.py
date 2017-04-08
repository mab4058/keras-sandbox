'''
Helper functions to compile ResNet building blocks.

@author: mbayer
'''
from keras import backend as K
from keras import layers
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import Activation


def identityBlock(input_tensor, n_filters, kernel_size, stage, block, bottleneck=False):
    """Block with no impedance shortcut. 
    # Arguments
        input_tensor: input tensor
        n_filters: int, Number of filters for conv layers.
        kernel_size: int or tuple, Kernel size for non-bottleneck layers.        
        stage: int, current stage label, used for generating layer names
        block: int, current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """    
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res_stage{}_block{}_branch'.format(stage, block)
    bn_name_base = 'bn_stage{}_block{}_branch'.format(stage, block)
    
    if not bottleneck:
        x = Conv2D(n_filters, kernel_size,
                   padding='same', name=conv_name_base + 'a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(n_filters, kernel_size,
                   padding='same', name=conv_name_base + 'b')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'b')(x)
        x = Activation('relu')(x)
    
        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
    else:
        x = Conv2D(n_filters, (1, 1), name=conv_name_base + 'a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(n_filters, kernel_size,
                   padding='same', name=conv_name_base + 'b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'b')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(n_filters*4, (1, 1), name=conv_name_base + 'c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'c')(x)
    
        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
    return x

def convBlock(input_tensor, n_filters, kernel_size, stage, block, strides=(2, 2), bottleneck=False):
    """Block with conv impedance shortcut.
    # Arguments
        input_tensor: input tensor
        n_filters: int, Number of filters for conv layers.
        kernel_size: int or tuple, Kernel size for non-bottleneck layers.        
        stage: int, current stage label, used for generating layer names
        block: int, current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res_stage{}_block{}_branch'.format(stage, block)
    bn_name_base = 'bn_stage{}_block{}_branch'.format(stage, block)
    
    if not bottleneck:
        x = Conv2D(n_filters, kernel_size, strides=strides, padding='same',
                   name=conv_name_base + 'a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(n_filters, kernel_size, padding='same',
                   name=conv_name_base + 'b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'b')(x)
        x = Activation('relu')(x)
    
        shortcut = Conv2D(n_filters, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
    
        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
    else:
        x = Conv2D(n_filters, (1, 1), strides=strides,
                   name=conv_name_base + 'a')(input_tensor)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'a')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(n_filters, kernel_size, padding='same',
                   name=conv_name_base + 'b')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'b')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(n_filters*4, (1, 1), name=conv_name_base + 'c')(x)
        x = BatchNormalization(axis=bn_axis, name=bn_name_base + 'c')(x)
    
        shortcut = Conv2D(n_filters*4, (1, 1), strides=strides,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)
        
        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        
    return x

if __name__ == '__main__':
    pass