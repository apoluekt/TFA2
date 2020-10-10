import amplitf.interface as atfi
from amplitf.phasespace.four_body_angular_phasespace import FourBodyAngularPhaseSpace

import tfa.plotting as tfp
import tfa.rootio as tfr
import tfa.neural_nets as tfn

import numpy as np

atfi.set_single_precision()

phsp = FourBodyAngularPhaseSpace()

ranges = phsp.bounds()

data = tfr.read_tuple("toys.root", ["cth1", "cth2", "phi"])

import matplotlib.pyplot as plt
tfp.set_lhcb_style(size = 10, usetex = False)
fig, ax = plt.subplots(nrows = 3, ncols = 3, figsize = (10, 9) )

tfn.estimate_density(
    phsp, data,
    ranges = ranges,
    labels = (r"$\cos\theta_{\mu\mu}$",r"$\cos\theta_{KK}$", r"$\phi$"),
    learning_rate=0.002,
    training_epochs=10000,
    norm_size=1000000,
    print_step=50,
    display_step=500,
    weight_penalty=0.1,
    n_hidden=[32, 8],
    initfile="init.npy",
    outfile="train",
    seed = 1,
    fig = fig, 
    axes = ax, 
)

ann = np.load("train.npy", allow_pickle = True)
print(ann)
scale, ranges = ann[:2]
weights, biases = tfn.init_fixed_weights_biases(ann[2:])

# Print the value of the parametrised PDF at the point (0,0,1)
data = atfi.const([[0., 0., 1.]])
v = scale*tfn.multilayer_perceptron(data, ranges, weights, biases)
print(v)

# Print the values of the parametrised PDF on the 5x5x5 grid
cth1, cth2, phi = [ i.flatten() for i in np.meshgrid(
                          np.linspace(-1., 1., 5), 
                          np.linspace(-1., 1., 5), 
                          np.linspace(0., 2.*3.14, 5)
                         ) ]
data = atfi.const(np.stack([cth1, cth2, phi], axis = 1))
v = scale*tfn.multilayer_perceptron(data, ranges, weights, biases)
print(np.stack([cth1, cth2, phi, v], axis = 1))
