'''
ResNet building blocks.

References:
    For dropout: 
        [Study of Residual Networks for Image Recognition](http://cs231n.stanford.edu/reports/2016/pdfs/264_Report.pdf)

@author: mbayer
'''
from keras import backend as K
from keras import layers
from keras.layers import Activation
from keras.layers.convolutional import Conv2D
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2


def resnetIdentityBlock(input_tensor,
                        n_filters,
                        kernel_size,
                        stage,
                        block,
                        dropout_rate=None,
                        l2_rate=None,
                        bottleneck=False):
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
    bn_name_base = 'bn_stage{}_block{}_branch_'.format(stage, block)
    conv_name_base = 'res_stage{}_block{}_branch_'.format(stage, block)
    
    if l2_rate is None:
        kernel_reg = None
    else:
        kernel_reg = l2(l2_rate)
    
    if not bottleneck:
        x = Conv2D(n_filters,
                   kernel_size,
                   padding='same',
                   kernel_regularizer=kernel_reg,
                   name=conv_name_base + 'a')(input_tensor)
        x = BatchNormalization(axis=bn_axis,
                               name=bn_name_base + 'a')(x)
        x = Activation('relu')(x)
        
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
    
        x = Conv2D(n_filters,
                   kernel_size,
                   padding='same',
                   kernel_regularizer=kernel_reg,
                   name=conv_name_base + 'b')(x)
        x = BatchNormalization(axis=bn_axis,
                               name=bn_name_base + 'b')(x)
    
        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
    else:
        x = Conv2D(n_filters,
                   1,
                   kernel_regularizer=kernel_reg,
                   name=conv_name_base + 'a')(input_tensor)
        x = BatchNormalization(axis=bn_axis,
                               name=bn_name_base + 'a')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(n_filters,
                   kernel_size,
                   padding='same',
                   kernel_regularizer=kernel_reg,
                   name=conv_name_base + 'b')(x)
        x = BatchNormalization(axis=bn_axis,
                               name=bn_name_base + 'b')(x)
        x = Activation('relu')(x)
        
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
    
        x = Conv2D(n_filters * 4,
                   1,
                   kernel_regularizer=kernel_reg,
                   name=conv_name_base + 'c')(x)
        x = BatchNormalization(axis=bn_axis,
                               name=bn_name_base + 'c')(x)
    
        x = layers.add([x, input_tensor])
        x = Activation('relu')(x)
    
    return x

def resnetConvBlock(input_tensor,
                    n_filters,
                    kernel_size,
                    stage,
                    block,
                    strides=(2, 2),
                    dropout_rate=None,
                    l2_rate=None,
                    bottleneck=False):
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
    conv_name_base = 'res_stage{}_block{}_branch_'.format(stage, block)
    bn_name_base = 'bn_stage{}_block{}_branch_'.format(stage, block)
    
    if l2_rate is None:
        kernel_reg = None
    else:
        kernel_reg = l2(l2_rate)
    
    if not bottleneck:
        x = Conv2D(n_filters,
                   kernel_size,
                   strides=strides,
                   padding='same',
                   kernel_regularizer=kernel_reg,
                   name=conv_name_base + 'a')(input_tensor)
        x = BatchNormalization(axis=bn_axis,
                               name=bn_name_base + 'a')(x)
        x = Activation('relu')(x)
        
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
    
        x = Conv2D(n_filters,
                   kernel_size,
                   padding='same',
                   kernel_regularizer=kernel_reg,
                   name=conv_name_base + 'b')(x)
        x = BatchNormalization(axis=bn_axis,
                               name=bn_name_base + 'b')(x)
    
        shortcut = Conv2D(n_filters,
                          1,
                          strides=strides,
                          kernel_regularizer=kernel_reg,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis,
                                      name=bn_name_base + '1')(shortcut)
    
        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
    else:
        x = Conv2D(n_filters,
                   1,
                   strides=strides,
                   kernel_regularizer=kernel_reg,
                   name=conv_name_base + 'a')(input_tensor)
        x = BatchNormalization(axis=bn_axis,
                               name=bn_name_base + 'a')(x)
        x = Activation('relu')(x)
    
        x = Conv2D(n_filters,
                   kernel_size,
                   padding='same',
                   kernel_regularizer=kernel_reg,
                   name=conv_name_base + 'b')(x)
        x = BatchNormalization(axis=bn_axis,
                               name=bn_name_base + 'b')(x)
        x = Activation('relu')(x)
        
        if dropout_rate is not None:
            x = Dropout(dropout_rate)(x)
    
        x = Conv2D(n_filters * 4,
                   1,
                   kernel_regularizer=kernel_reg,
                   name=conv_name_base + 'c')(x)
        x = BatchNormalization(axis=bn_axis,
                               name=bn_name_base + 'c')(x)
    
        shortcut = Conv2D(n_filters * 4,
                          1,
                          strides=strides,
                          kernel_regularizer=kernel_reg,
                          name=conv_name_base + '1')(input_tensor)
        shortcut = BatchNormalization(axis=bn_axis,
                                      name=bn_name_base + '1')(shortcut)
        
        x = layers.add([x, shortcut])
        x = Activation('relu')(x)
        
    return x

if __name__ == '__main__':
    pass
