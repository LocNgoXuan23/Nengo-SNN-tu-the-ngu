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

import keras_spiking
import nengo

from time import time
import numpy as np
import nengo_dl
from utils import *
from tqdm import tqdm
import cv2
from dataset import *
from models import *

import keras_spiking
import nengo
import os
import matplotlib.pyplot as plt

def run_network(
    activation,
    model,
    conv0,
    test_images, 
    test_labels,
    n_steps=120,
    scale_firing_rates=5,
    synapse=None,
    n_test=400,
):
    # print("11111111111111111")
    nengo_converter = nengo_dl.Converter(
        model=model,
        swap_activations={tf.nn.relu: activation},
        scale_firing_rates=scale_firing_rates,
        synapse=synapse,
    )
    # print("222222222222222")

    nengo_input = nengo_converter.inputs[model.inputs]
    nengo_output = nengo_converter.outputs[model.outputs]

    sample_neurons = np.linspace(
        0,
        np.prod(conv0.shape[1:]),
        1000,
        endpoint=False,
        dtype=np.int32,
    )
    # print("333333333333333333")
    # print(test_images.shape)


    with nengo_converter.net:
        conv0_probe = nengo.Probe(nengo_converter.layers[conv0][sample_neurons])

    tiled_test_images = np.tile(test_images[:n_test], (1, n_steps, 1))
    
    # print("444444444444444444")

    with nengo_converter.net:
        nengo_dl.configure_settings(planner=nengo_dl.graph_optimizer.noop_planner)
        nengo_dl.configure_settings(stateful=False)
        #nengo_dl.configure_settings(trainable=False)

    # print("5555555555555555555")
    # print(tiled_test_images.shape)

    with nengo_dl.Simulator(
        nengo_converter.net, minibatch_size=10, device="/gpu:2", progress_bar=True
    ) as nengo_sim:
        data = nengo_sim.predict({nengo_input: tiled_test_images})
    
    predictions = np.argmax(data[nengo_output][:, -1], axis=-1)
    accuracy = (predictions == test_labels[:n_test, 0, 0]).mean()
    print("DO CHINH XAC:", accuracy)

    for ii in range(3):
        plt.figure(figsize=(24, 8))

        plt.subplot(1, 3, 1)
        plt.title("Input image")
        plt.imshow(test_images[ii, 0].reshape((224,224)), cmap="gray")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        scaled_data = data[conv0_probe][ii] * scale_firing_rates
        if isinstance(activation, nengo.SpikingRectifiedLinear):
            scaled_data *= 0.001
            rates = np.sum(scaled_data, axis=0) / (n_steps * nengo_sim.dt)
            plt.ylabel("Number of spikes")
        else:
            rates = scaled_data
            plt.ylabel("Firing rates (Hz)")
        plt.xlabel("Timestep")
        plt.title(
            f"Neural activities (conv0 mean={rates.mean():.1f} Hz, "
            f"max={rates.max():.1f} Hz)"
        )
        plt.plot(scaled_data)

        plt.subplot(1, 3, 3)
        plt.title("Output predictions")
        plt.plot(tf.nn.softmax(data[nengo_output][ii]))
        plt.legend(["Posture " + str(j) for j in range(9)], loc="upper left")
        plt.xlabel("Timestep")
        plt.ylabel("Probability")

        plt.tight_layout()

        plt.savefig(f'out_imgs/{file_name}/out{ii}_{file_name}_{synapse}_{scale_firing_rates}_{accuracy}.jpg')
    return accuracy

model, input, output, conv0 = mobilenet_relu_2((224,224,1),9)

train_data = get_data_cnn('../data/train.json')
test_data = get_data_cnn('../data/val.json')

print("train_images_shape ",train_data[0].shape)
print("test_images_shape ",test_data[0].shape)
print("train_labels_shape ",train_data[1].shape)
print("test_labels_shape ",test_data[1].shape)

model.compile(optimizer=tf.optimizers.Adam(0.001),
            loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=[tf.metrics.sparse_categorical_accuracy],)

file_name = 'mobilenet_relu_2'
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
    epochs=1,
    validation_data=(test_data[0], test_data[1]),
    verbose=1,
    callbacks=[model_checkpoint_callback],
    )
print('MAX ACC : ', max(history.history['val_sparse_categorical_accuracy']))

model.load_weights(checkpoint_filepath)
print("LOAD WEIGHT DONE !!")
print("Evaluate on test data")
results = model.evaluate(test_data[0], test_data[1], batch_size=64)
print("test loss, test acc:", results)


train_data = get_data('../data/train.json')
test_data = get_data('../data/val.json')


print("train_images_shape ",train_data[0].shape)
print("test_images_shape ",test_data[0].shape)
print("train_labels_shape ",train_data[1].shape)
print("test_labels_shape ",test_data[1].shape)
print("LOAD DATA SNN DONE !!")

hist = []

for s in [0.005]:
    # for scale in [2, 5, 10, 20, 50, 100, 200, 500]:
    # for scale in [2, 5, 10, 20, 50, 100, 200, 500, 1000, 5000]:
    for scale in [500]:
    # for scale in [2, 5]:
        print(f"Synapse={s:.3f}", f"scale_firing_rates={scale:.3f}")
        acc = run_network(activation=nengo.SpikingRectifiedLinear(), model=model, conv0=conv0, test_labels=test_data[1],test_images=test_data[0], scale_firing_rates=scale, n_steps=150, synapse=s,)

        hist.append([s, scale, acc])

        print('\n')
        print('THE ENERGY OF SNN WITH SCALE ', scale)
        energy = keras_spiking.ModelEnergy(model, example_data=np.ones((32, 224, 224,1))*scale)
        energy.summary(
            columns=(
          "name",
        #   "synop_energy loihi",
        #   "neuron_energy loihi",
          'energy loihi ',
          "energy cpu",
          "energy gpu",
        #   "synop_energy cpu",
        #   "synop_energy gpu",
        #   "neuron_energy cpu",
        #   "neuron_energy gpu"
            ),
            print_warnings=False,
        )

print(hist)