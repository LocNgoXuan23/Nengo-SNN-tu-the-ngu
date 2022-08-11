from keras.models import Model
from keras.layers import Input, Dense, Conv2D, Conv3D, DepthwiseConv2D, SeparableConv2D, Conv3DTranspose, AveragePooling2D
from keras.layers import Flatten, MaxPool2D, AvgPool2D, GlobalAvgPool2D, UpSampling2D, BatchNormalization
from keras.layers import Concatenate, Add, Dropout, ReLU, Lambda, Activation, LeakyReLU, PReLU

from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot

from time import time
import numpy as np
import nengo_dl
from utils import *
from tqdm import tqdm
import cv2
from dataset import *
from models import *

times = 5

for i in range(times):
    print(f'-----------TIME {i}------------')
    model, input, output, conv0 = mobilenet_relu((224,224,1),9)

    train_data = get_data_cnn('../data/train.json')
    test_data = get_data_cnn('../data/val.json')

    print("train_images_shape ",train_data[0].shape)
    print("test_images_shape ",test_data[0].shape)
    print("train_labels_shape ",train_data[1].shape)
    print("test_labels_shape ",test_data[1].shape)

    model.compile(optimizer=tf.optimizers.Adam(0.001),
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[tf.metrics.sparse_categorical_accuracy],)

    file_name = f'mobilenet_relu_{time()}'
    checkpoint_filepath = os.path.join('cktp', f'{file_name}.h5')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='val_sparse_categorical_accuracy',
        mode='max',
        save_best_only=True,
        verbose=1
        )

    history = model.fit(train_data[0], train_data[1],
        batch_size=64,
        epochs=300,
        validation_data=(test_data[0], test_data[1]),
        verbose=1,
        callbacks=[model_checkpoint_callback],
        )
    print('MAX ACC : ', max(history.history['val_sparse_categorical_accuracy']))

    model.load_weights(checkpoint_filepath)
    print("Evaluate on test data")
    results = model.evaluate(test_data[0], test_data[1], batch_size=64)
    print("test loss, test acc:", results)

