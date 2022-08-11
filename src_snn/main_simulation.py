import nengo_dl
import tensorflow as tf
import numpy as np
import nengo
from models import *
import keras_spiking



def run_network(
    activation,
    model,
    conv0,
    test_images, 
    test_labels,
    params_file="../weights",
    n_steps=120,
    scale_firing_rates=5,
    synapse=None,
    n_test=400,
):
    print("11111111111111111")
    nengo_converter = nengo_dl.Converter(
        model=model,
        swap_activations={tf.nn.relu: activation},
        scale_firing_rates=scale_firing_rates,
        synapse=synapse,
        freeze_batchnorm=True
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

    print("5555555555555555555")
    print(tiled_test_images.shape)

    with nengo_dl.Simulator(
        nengo_converter.net, minibatch_size=10, progress_bar=False
    ) as nengo_sim:
        params = list(nengo_sim.keras_model.weights)
        # print(len(params))
        nengo_sim.load_params(params_file)
        data = nengo_sim.predict({nengo_input: tiled_test_images})
    
    predictions = np.argmax(data[nengo_output][:, -1], axis=-1)
    accuracy = (predictions == test_labels[:n_test, 0, 0]).mean()
    print("DO CHINH XAC:", accuracy)


model, input, output, conv0 = mobilenet((224,224,1),9)
train_data = get_data('../data/train.json')
test_data = get_data('../data/val.json')

for s in [ 0.01]:
    for scale in [50]:
        print(f"Synapse={s:.3f}", f"scale_firing_rates={scale:.3f}")
        run_network(
        activation=nengo.SpikingRectifiedLinear(),
        model=model,
        conv0=conv0,
        test_labels=test_data[1],test_images=test_data[0],
        scale_firing_rates=scale,
        n_steps=120,
        synapse=s,
        )
        print('\n')
        print('THE ENERGY OF SNN WITH SCALE ', scale)
        energy = keras_spiking.ModelEnergy(model, example_data=np.ones((32, 64, 32,1))*scale)
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


