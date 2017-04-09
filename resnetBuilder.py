'''
ResNet Builder

References:
Deep Residual Learning for Image Recognition [https://arxiv.org/pdf/1512.03385.pdf]
Identity Mappings in Deep Residual Networks [https://arxiv.org/pdf/1603.05027v2.pdf]


@author: mbayer
'''
from keras.layers import Input
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Flatten, Dense
from keras.models import Model

from keras import backend as K
from buildingBlocks.resnetBuildingBlocks import identityBlock, convBlock
from buildingBlocks.basicBuildingBlocks import conv2d_bn

def build(input_shape, num_outputs, repetitions, bottleneck=False):
    """Builder function to create ResNet model architecture.
    # Arguments
    input_shape: tuple, (n_rows, n_cols, n_channels)
        num_outputs: int, The number of outputs from the model.
        repetitions: list, The repetition amounts per stage.
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
                x = convBlock(x, 
                              n_filters, 
                              kernel_size, 
                              stage=stage_num, 
                              block=block_num, 
                              strides=(1,1), 
                              bottleneck=bottleneck)
            elif block_num == 0 and stage_num > 0:
                x = convBlock(x, 
                              n_filters, 
                              kernel_size, 
                              stage=stage_num, 
                              block=block_num, 
                              bottleneck=bottleneck)
            elif block_num > 0:
                x = identityBlock(x, 
                                  n_filters, 
                                  kernel_size, 
                                  stage=stage_num, 
                                  block=block_num, 
                                  bottleneck=bottleneck)                
        n_filters *= 2
        
    # Get previous layer shape for average pooling layer.
    temp = x.shape
    s = (temp[0].value, temp[1].value, temp[2].value)
    x = AveragePooling2D((s[1],s[2]), name='avg_pool')(x)
        
    x = Flatten()(x)
    x = Dense(num_outputs, activation='softmax', name='fc1000')(x)

    model = Model(input_layer, x, name='resnet')
    
    model.summary()
    
    return model

class ResnetBuilder(object):
    @staticmethod
    def buildResnet18(input_shape, num_outputs):
        return build(input_shape, num_outputs, [2, 2, 2, 2], bottleneck=False)
    
    @staticmethod
    def buildResnet34(input_shape, num_outputs):
        return build(input_shape, num_outputs, [3, 4, 6, 3], bottleneck=False)
    
    @staticmethod
    def buildResnet50(input_shape, num_outputs):
        return build(input_shape, num_outputs, [3, 4, 6, 3], bottleneck=True)
    
    @staticmethod
    def buildResnet101(input_shape, num_outputs):
        return build(input_shape, num_outputs, [3, 4, 23, 3], bottleneck=True)
    
    @staticmethod
    def buildResnet152(input_shape, num_outputs):
        return build(input_shape, num_outputs, [3, 8, 36, 3], bottleneck=True)