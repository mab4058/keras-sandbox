'''
ResNet Builder

References:
Deep Residual Learning for Image Recognition [https://arxiv.org/pdf/1512.03385.pdf]
Identity Mappings in Deep Residual Networks [https://arxiv.org/pdf/1603.05027v2.pdf]


@author: mbayer
'''
from keras import backend as K
from keras.layers import Input
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model

from buildingBlocks.basicBuildingBlocks import conv2d_bn
from buildingBlocks.resnetBuildingBlocks import resnetConvBlock, resnetIdentityBlock


def build(input_shape, num_outputs, repetitions, dropout_rate=None, bottleneck=False):
    """Builder function to create ResNet model architecture.
    # Arguments
    input_shape: tuple, (n_rows, n_cols, n_channels)
        num_outputs: int, The number of outputs from the model.
        repetitions: list, The repetition amounts per stage.
        dropout_rate: float, optional: Number in range 0-1 for dropout. Enter none
                      to ignore.
        bottleneck: bool, optional: If true, bottleneck structures will be used for
                    each block.
    
    # Returns
        Output model architecture.
    """
    if K.image_data_format() == 'channels_first':
        input_shape = (input_shape[2], input_shape[0], input_shape[1])
        
    input_layer = Input(input_shape)
    input_conv = conv2d_bn(input_layer, 64, 7, strides=(2,2), name='input_conv')
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same', name='input_pool')(input_conv)
    
    n_filters = 64
    kernel_size = 3
    for stage_num, r in enumerate(repetitions):
        for block_num in range(r):
            if block_num == 0 and stage_num == 0:
                x = resnetConvBlock(x, 
                                    n_filters, 
                                    kernel_size, 
                                    stage=stage_num, 
                                    block=block_num, 
                                    strides=(1,1), 
                                    bottleneck=bottleneck)
            elif block_num == 0 and stage_num > 0:
                x = resnetConvBlock(x, 
                                    n_filters, 
                                    kernel_size, 
                                    stage=stage_num, 
                                    block=block_num, 
                                    bottleneck=bottleneck)
            elif block_num > 0:
                x = resnetIdentityBlock(x, 
                                        n_filters, 
                                        kernel_size, 
                                        stage=stage_num, 
                                        block=block_num, 
                                        bottleneck=bottleneck)                
        n_filters *= 2
        
    # Get previous layer shape for average pooling layer.
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
        
#     x = Flatten()(x)
#     x = Dense(512, activation='relu')(x)
    
    if dropout_rate is not None:
        x = Dropout(dropout_rate)(x)
        
    x = Dense(num_outputs, activation='softmax', name='out_fc')(x)

    model = Model(input_layer, x, name='resnet')
    
    model.summary()
    
    return model

def saveModel(filename, model):
    modelContents = model.to_json()
    with open(filename, 'w+') as f:
        f.write(modelContents)

class ResnetBuilder(object):
    @staticmethod
    def buildResnet18(input_shape, num_outputs, filename=None):
        model = build(input_shape, num_outputs, [2, 2, 2, 2], bottleneck=False)
        if filename is not None:
            saveModel(filename,model)
        return model
    
    @staticmethod
    def buildResnet34(input_shape, num_outputs, filename=None):
        model = build(input_shape, num_outputs, [3, 4, 6, 3], bottleneck=False)
        if filename is not None:
            saveModel(filename,model)
        return model
                
    @staticmethod
    def buildResnet50(input_shape, num_outputs, filename=None):
        model = build(input_shape, num_outputs, [3, 4, 6, 3], bottleneck=True)
        if filename is not None:
            saveModel(filename,model)
        return model
                
    @staticmethod
    def buildResnet101(input_shape, num_outputs, filename=None):
        model = build(input_shape, num_outputs, [3, 4, 23, 3], bottleneck=True)
        if filename is not None:
            saveModel(filename,model)
        return model
            
    @staticmethod
    def buildResnet152(input_shape, num_outputs, filename=None):
        model = build(input_shape, num_outputs, [3, 8, 36, 3], bottleneck=True)
        if filename is not None:
            saveModel(filename,model)
        return model        