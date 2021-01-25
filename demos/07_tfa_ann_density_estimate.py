# Example of a 3D density estimation using ANNs

import amplitf.interface as atfi
from amplitf.phasespace.four_body_angular_phasespace import FourBodyAngularPhaseSpace

import tfa.plotting as tfp
import tfa.rootio as tfr
import tfa.neural_nets as tfn

import numpy as np

# Single precision is usually sufficient for training ANNs,
# and works faster on consumer GPUs
atfi.set_single_precision()

# Create phase space object (four-body decay with fixed masses
# of intermediate particles)
phsp = FourBodyAngularPhaseSpace()

# Read previously generated data
data = tfr.read_tuple("toys.root", ["cth1", "cth2", "phi"])

# Open matplotlib window
import matplotlib.pyplot as plt

tfp.set_lhcb_style(size=10, usetex=False)
fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(10, 9))

# Perform density estimation using ANNs
tfn.estimate_density(
    phsp,
    data,
    ranges=phsp.bounds(),  # list of ranges for observables
    labels=(r"$\cos\theta_{\mu\mu}$", r"$\cos\theta_{KK}$", r"$\phi$"),
    learning_rate=0.002,  # Tunable meta-parameter
    weight_penalty=0.1,  # Tunable meta-parameter (larger for smoother distribution)
    n_hidden=[32, 8],  # Structure of hidden layers (2 layers, 32 and 8 neurons)
    training_epochs=10000,  # Number of training iterations (epochs)
    norm_size=1000000,  # Size of the normalisation sample
    print_step=50,  # Print cost function every 50 epochs
    display_step=500,  # Display status of training every 500 epochs
    initfile="init.npy",  # Init file (e.g. if continuing previously interrupted training)
    outfile="train",  # Name prefix for output files (.pdf, .npy, .txt)
    seed=1,  # Random seed
    fig=fig,  # matplotlib window references
    axes=ax,
)

# tfn.estimate_density is running in TF v1 compatibility mode for now.
# We cannot switch back to v2 mode, so we can only exit the script.
# Loading and using the trained ANN in in the next script.
