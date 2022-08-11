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

model, input, output, conv0 = vgg((224,224,1),9)

converter = nengo_dl.Converter(model)
net_train = converter.net

with net_train:
	nengo_dl.configure_settings(planner=nengo_dl.graph_optimizer.noop_planner)

train_data = get_data('../data/train.json')
test_data = get_data('../data/val.json')

print("train_images_shape ",train_data[0].shape)
print("test_images_shape ",test_data[0].shape)
print("train_labels_shape ",train_data[1].shape)
print("test_labels_shape ",test_data[1].shape)


with nengo_dl.Simulator(net_train, minibatch_size=64,seed=0, device="/gpu:3") as sim:
	# run training
	sim.compile(
		optimizer=tf.optimizers.Adam(0.001),
		loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
		metrics=[tf.metrics.sparse_categorical_accuracy],
	)
	print("--------------------Start Training--------------------")
	sim.fit(
		{converter.inputs[input]: train_data[0]},
		{converter.outputs[output]: train_data[1]},
		validation_data=(
			{converter.inputs[input]: test_data[0]},
			{converter.outputs[output]: test_data[1]},
		),
		epochs=300,
		#callbacks=[model_checkpoint_callback],
	)
	# save the parameters to file
	# path_params = "SNN_PARAMS/SNN_BED_LOSO_{}".format(1)
	sim.save_params("weights")

print("Training Done !!")



def run_network(
    activation,
    model,
    conv0,
    test_images, 
    test_labels,
    params_file="weights",
    n_steps=120,
    scale_firing_rates=5,
    synapse=None,
    n_test=400,
):
    print("11111111111111111")
    nengo_converter = nengo_dl.Converter(
        model=model,
        swap_activations={tf.keras.activations.relu: activation},
        scale_firing_rates=scale_firing_rates,
        synapse=synapse,
    )
    print("222222222222222")

    nengo_input = nengo_converter.inputs[model.inputs]
    nengo_output = nengo_converter.outputs[model.outputs]

    sample_neurons = np.linspace(
        0,
        np.prod(conv0.shape[1:]),
        1000,
        endpoint=False,
        dtype=np.int32,
    )
    print("333333333333333333")
    print(test_images.shape)


    with nengo_converter.net:
        conv0_probe = nengo.Probe(nengo_converter.layers[conv0][sample_neurons])

    tiled_test_images = np.tile(test_images[:n_test], (1, n_steps, 1))
    
    print("444444444444444444")

    with nengo_converter.net:
        nengo_dl.configure_settings(planner=nengo_dl.graph_optimizer.noop_planner)
        nengo_dl.configure_settings(stateful=False)
        #nengo_dl.configure_settings(trainable=False)

    print("5555555555555555555")
    print(tiled_test_images.shape)

    with nengo_dl.Simulator(
        nengo_converter.net, minibatch_size=10, progress_bar=True
    ) as nengo_sim:
        #params = list(nengo_sim.keras_model.weights)
        #print(len(params))
        nengo_sim.load_params(params_file)
        data = nengo_sim.predict({nengo_input: tiled_test_images})
    
    predictions = np.argmax(data[nengo_output][:, -1], axis=-1)
    accuracy = (predictions == test_labels[:n_test, 0, 0]).mean()
    print("DO CHINH XAC:", accuracy)
    

#model, input, output, conv0 = mobilenet((224,224,1),9)
#model.load_weights("weights")

for s in [ 0.01]:
    for scale in [50]:
        print(f"Synapse={s:.3f}", f"scale_firing_rates={scale:.3f}")
        run_network(activation=nengo.SpikingRectifiedLinear(), model=model, conv0=conv0, test_labels=test_data[1],test_images=test_data[0], scale_firing_rates=scale, n_steps=120, synapse=s,)
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





