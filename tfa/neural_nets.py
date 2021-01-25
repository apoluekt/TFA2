# Copyright 2017 CERN for the benefit of the LHCb collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
import tensorflow as tf
import sys
import math
import matplotlib.pyplot as plt

import amplitf.interface as atfi
import tfa.rootio as atfr
import tfa.plotting as atfp


def create_weights_biases(n_input, layers, sigma=1.0, n_output=1):
    """
    Create arrays of weights and vectors of biases for the multilayer perceptron
    if a given configuration (with a single output neuron).
      n_input : number of input neurons
      layers  : list of numbers of neurons in the hidden layers
    output :
      weights, biases
    """
    n_hidden = [n_input] + layers
    weights = []
    biases = []
    for i in range(len(n_hidden) - 1):
        weights += [
            tf.Variable(
                sigma * np.random.normal(size=[n_hidden[i], n_hidden[i + 1]]),
                dtype=atfi.fptype(),
            )
        ]
        biases += [
            tf.Variable(
                sigma * np.random.normal(size=[n_hidden[i + 1]]), dtype=atfi.fptype()
            )
        ]
    weights += [
        tf.Variable(
            sigma * np.random.normal(size=[n_hidden[-1], n_output]), dtype=atfi.fptype()
        )
    ]
    biases += [
        tf.Variable(sigma * np.random.normal(size=[n_output]), dtype=atfi.fptype())
    ]
    return (weights, biases)


def init_weights_biases(init):
    """
    Initialise variable weights and biases from numpy array
    """
    init_weights = init[0]
    init_biases = init[1]
    weights = []
    biases = []
    for i in range(len(init_weights) - 1):
        weights += [tf.Variable(init_weights[i], dtype=atfi.fptype())]
        biases += [tf.Variable(init_biases[i], dtype=atfi.fptype())]
    weights += [tf.Variable(init_weights[-1], dtype=atfi.fptype())]
    biases += [tf.Variable(init_biases[-1], dtype=atfi.fptype())]
    return (weights, biases)


def init_fixed_weights_biases(init):
    """
    Initialise constant weights and biases from numpy array
    """
    init_weights = init[0]
    init_biases = init[1]
    weights = []
    biases = []
    for i in range(len(init_weights) - 1):
        weights += [tf.constant(init_weights[i], dtype=atfi.fptype())]
        biases += [tf.constant(init_biases[i], dtype=atfi.fptype())]
    weights += [tf.constant(init_weights[-1], dtype=atfi.fptype())]
    biases += [tf.constant(init_biases[-1], dtype=atfi.fptype())]
    return (weights, biases)


def normalise(array, ranges):
    v = [
        (array[:, i] - ranges[i][0]) / (ranges[i][1] - ranges[i][0])
        for i in range(len(ranges))
    ]
    return tf.stack(v, axis=1)


@tf.function
def multilayer_perceptron(x, ranges, weights, biases, multiple=False):
    """
    Multilayer perceptron with fully connected layers defined by matrices of weights and biases.
    Use sigmoid function as activation.
    """
    layer = normalise(x, ranges)
    for i in range(len(weights)):
        layer = tf.nn.sigmoid(tf.add(tf.matmul(layer, weights[i]), biases[i]))
    if multiple:
        return layer
    else:
        return layer[:, 0]


def l2_regularisation(weights):
    """
    L2 regularisation term for a list of weight matrices
    """
    penalty = 0.0
    for w in weights:
        penalty += tf.reduce_sum(tf.square(w))
    return penalty


def estimate_density(
    phsp,
    data,
    ranges,
    labels,
    weight=None,
    transform=None,
    transform_ranges=None,
    learning_rate=0.001,
    training_epochs=100000,
    norm_size=1000000,
    print_step=50,
    display_step=500,
    weight_penalty=1.0,
    n_hidden=[32, 8],
    initfile="init.npy",
    outfile="train",
    seed=1,
    fig=None,
    axes=None,
):

    tf.compat.v1.disable_eager_execution()

    n_input = len(ranges)

    bins = n_input * [50]

    tf.compat.v1.set_random_seed(seed)
    np.random.seed(seed + 12345)

    try:
        init_w = np.load(initfile, allow_pickle=True)
    except:
        init_w = None

    if isinstance(init_w, np.ndarray):
        print("Loading saved weights")
        (weights, biases) = init_weights_biases(init_w)
    else:
        print("Creating random weights")
        (weights, biases) = create_weights_biases(n_input, n_hidden)

    data_ph = tf.compat.v1.placeholder(atfi.fptype(), shape=(None, None), name="data")
    norm_ph = tf.compat.v1.placeholder(atfi.fptype(), shape=(None, None), name="norm")

    if not transform_ranges:
        transform_ranges = ranges

    def model(x):
        if transform:
            x2 = transform(x)
        else:
            x2 = x
        # to make sure PDF is always strictly positive
        return multilayer_perceptron(x2, transform_ranges, weights, biases) + 1e-20

    data_model = model(data_ph)
    norm_model = model(norm_ph)

    def unbinned_nll(pdf, integral):
        return -tf.reduce_sum(atfi.log(pdf / integral))

    def integral(pdf):
        return tf.reduce_mean(pdf)

    # Define loss and optimizer
    nll = (
        unbinned_nll(data_model, integral(norm_model))
        + l2_regularisation(weights) * weight_penalty
    )

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(nll)

    # Initializing the variables
    init = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init)

        data_sample = sess.run(phsp.filter(data_ph), feed_dict={data_ph: data})

        norm_sample = sess.run(phsp.uniform_sample(norm_size))
        print("Normalisation sample size = ", len(norm_sample))
        print(norm_sample)
        print("Data sample size = ", len(data_sample))
        print(data_sample)

        # Training cycle
        best_cost = 1e10

        display = atfp.MultidimDisplay(
            data_sample, norm_sample, bins, ranges, labels, fig, axes
        )
        plt.ion()
        plt.show()
        plt.pause(0.1)

        for epoch in range(training_epochs):

            _, c = sess.run(
                [train_op, nll], feed_dict={data_ph: data_sample, norm_ph: norm_sample}
            )

            if epoch % display_step == 0 and fig:
                w = sess.run(norm_model, feed_dict={norm_ph: norm_sample})
                display.draw(w)
                plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
                plt.draw()
                plt.pause(0.1)
                plt.savefig(outfile + ".pdf")

            if epoch % print_step == 0:
                s = "Epoch %d, cost %.9f" % (epoch + 1, c)
                print(s)
                if c < best_cost:
                    best_cost = c
                    w = sess.run(norm_model, feed_dict={norm_ph: norm_sample})
                    scale = 1.0 / np.mean(w)
                    np.save(
                        outfile, [scale, transform_ranges] + sess.run([weights, biases])
                    )
                    f = open(outfile + ".txt", "w")
                    f.write(s + "\n")
                    f.close()

        print("Optimization Finished!")
