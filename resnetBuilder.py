"""
ResNet Builder

References:
Deep Residual Learning for Image Recognition [https://arxiv.org/pdf/1512.03385.pdf]
Identity Mappings in Deep Residual Networks [https://arxiv.org/pdf/1603.05027v2.pdf]


@author: mbayer
"""
import os

from keras import backend as K
from keras.layers import Input
from keras.layers.core import Dense, Dropout
from keras.layers.pooling import MaxPooling2D, GlobalAveragePooling2D
from keras.models import Model

from buildingBlocks.basic_building_blocks import conv2d_bn
from buildingBlocks.resnet_building_blocks import resnet_conv_block, resnet_identity_block


def build(input_shape,
          num_outputs,
          repetitions,
          n_filters_start=32,
          dropout_rate=(None, None),
          l2_rate=None,
          init='glorot_normal',
          bottleneck=False):
    """Builder function to create ResNet model architecture.
    # Arguments
    input_shape: tuple, (n_rows, n_cols, n_channels)
        num_outputs: int, The number of classes.
        repetitions: list, The repetition amounts per stage.
        n_filters_start: Number of filters to start. This will grow.
        dropout_rate: tuple, optional: Number in range 0-1 for dropout. Enter none
                      to ignore. First index occurs in resnet blocks and second
                      occurs before classification layer.
        bottleneck: bool, optional: If true, bottleneck structures will be used for
                    each block.
    
    # Returns
        Output model architecture.
    """
    if K.image_data_format() == 'channels_first':
        input_shape = (input_shape[2], input_shape[0], input_shape[1])

    input_layer = Input(input_shape)
    input_conv = conv2d_bn(input_layer,
                           n_filters_start,
                           7,
                           strides=(2, 2),
                           l2_rate=l2_rate,
                           init=init,
                           name='input_conv')
    x = MaxPooling2D(pool_size=(3, 3),
                     strides=(2, 2),
                     padding='same',
                     name='input_pool')(input_conv)

    n_filters = n_filters_start
    kernel_size = 3
    for stage_num, r in enumerate(repetitions):
        for block_num in range(r):
            if block_num == 0 and stage_num == 0:
                x = resnet_conv_block(x,
                                      n_filters,
                                      kernel_size,
                                      stage=stage_num,
                                      block=block_num,
                                      strides=(1, 1),
                                      dropout_rate=dropout_rate[1],
                                      l2_rate=l2_rate,
                                      init=init,
                                      bottleneck=bottleneck)
            elif block_num == 0 and stage_num > 0:
                # x = MaxPooling2D(pool_size=(2,2))(x)
                x = resnet_conv_block(x,
                                      n_filters,
                                      kernel_size,
                                      stage=stage_num,
                                      block=block_num,
                                      dropout_rate=dropout_rate[1],
                                      strides=(2, 2),
                                      l2_rate=l2_rate,
                                      init=init,
                                      bottleneck=bottleneck)
            elif block_num > 0:
                x = resnet_identity_block(x,
                                          n_filters,
                                          kernel_size,
                                          stage=stage_num,
                                          block=block_num,
                                          dropout_rate=dropout_rate[1],
                                          l2_rate=l2_rate,
                                          init=init,
                                          bottleneck=bottleneck)
        n_filters *= 2

    if dropout_rate is not None:
        x = Dropout(dropout_rate[0])(x)

    # Get previous layer shape for average pooling layer.
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)

    x = Dense(num_outputs,
              activation='softmax',
              name='out_fc')(x)

    model = Model(input_layer, x, name='resnet')

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


class ResnetBuilder(object):
    @staticmethod
    def buildResnet18(input_shape, num_outputs, dropout_rate=(None, None), l2_rate=None,
                      filename=None):
        model = build(input_shape, num_outputs, [2, 2, 2, 2], dropout_rate=dropout_rate,
                      l2_rate=l2_rate, bottleneck=False)
        if filename is not None:
            saveModel(filename, model)
        return model

    @staticmethod
    def buildResnet34(input_shape, num_outputs, dropout_rate=(None, None), l2_rate=None,
                      filename=None):
        model = build(input_shape, num_outputs, [3, 4, 6, 3], dropout_rate=dropout_rate,
                      l2_rate=l2_rate, bottleneck=False)
        if filename is not None:
            saveModel(filename, model)
        return model

    @staticmethod
    def buildResnet50(input_shape, num_outputs, dropout_rate=(None, None), l2_rate=None,
                      filename=None):
        model = build(input_shape, num_outputs, [3, 4, 6, 3], dropout_rate=dropout_rate,
                      l2_rate=l2_rate, bottleneck=True)
        if filename is not None:
            saveModel(filename, model)
        return model

    @staticmethod
    def buildResnet101(input_shape, num_outputs, dropout_rate=(None, None), l2_rate=None,
                       filename=None):
        model = build(input_shape, num_outputs, [3, 4, 23, 3], dropout_rate=dropout_rate,
                      l2_rate=l2_rate, bottleneck=True)
        if filename is not None:
            saveModel(filename, model)
        return model

    @staticmethod
    def buildResnet152(input_shape, num_outputs, dropout_rate=(None, None), l2_rate=None,
                       filename=None):
        model = build(input_shape, num_outputs, [3, 8, 36, 3], dropout_rate=dropout_rate,
                      l2_rate=l2_rate, bottleneck=True)
        if filename is not None:
            saveModel(filename, model)
        return model


if __name__ == '__main__':
    filename = r'C:\Users\mbayer\Desktop\Keras_Models\resnetTest.json'
    m = build((150, 200, 3),
              6,
              [3, 4, 6, 3],
              n_filters_start=16,
              dropout_rate=(0.5, None),
              l2_rate=0.0001,
              bottleneck=True)
    saveModel(filename, m)
