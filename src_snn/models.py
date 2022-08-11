
import keras
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose
from keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization, AveragePooling2D, BatchNormalizationV1, BatchNormalizationV2
from keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
import tensorflow as tf

from time import time
import numpy as np

def mobilenet(input_shape, n_classes):
  
  def mobilenet_block(x, f, s=1):
    x = DepthwiseConv2D(3, strides=s, padding='same')(x)
    x = BatchNormalizationV2()(x)
    x = ReLU()(x)
    
    x = Conv2D(f, 1, strides=1, padding='same', activation=tf.nn.relu)(x)
    x = BatchNormalizationV2()(x)
    x = ReLU()(x)
    return x
    
  
  input = Input(input_shape)

  conv0 = Conv2D(32, 3, strides=2, padding='same', activation=tf.nn.relu)(input)
  x = BatchNormalizationV2()(conv0)
  x = ReLU()(x)

  x = mobilenet_block(x, 64)
  x = mobilenet_block(x, 128, 2)
  x = mobilenet_block(x, 128)

  x = mobilenet_block(x, 256, 2)
  x = mobilenet_block(x, 256)

  x = mobilenet_block(x, 512, 2)
  for _ in range(5):
    x = mobilenet_block(x, 512)

  x = mobilenet_block(x, 1024, 2)
  x = mobilenet_block(x, 1024)
  
  x = GlobalAvgPool2D()(x)
  
  output = Dense(n_classes, activation='softmax')(x)
  
  model = Model(input, output)
  return model, input, output, conv0
