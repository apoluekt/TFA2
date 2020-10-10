# Example of a 3D density estimation using ANNs 

import amplitf.interface as atfi
from amplitf.phasespace.four_body_angular_phasespace import FourBodyAngularPhaseSpace

import tfa.plotting as tfp
import tfa.rootio as tfr
import tfa.neural_nets as tfn

import numpy as np

# Create phase space object (four-body decay with fixed masses 
# of intermediate particles)
phsp = FourBodyAngularPhaseSpace()

# Load the parameters of trained ANN
ann = np.load("train.npy", allow_pickle = True)
print(ann)
scale, ranges = ann[:2] # Overall scale of the PDF and ranges of input data
weights, biases = tfn.init_fixed_weights_biases(ann[2:]) # Initialise ANN parameters

# Print the value of the parametrised PDF at the point (0,0,1)
data = atfi.const([[0., 0., 1.]])
v = scale*tfn.multilayer_perceptron(data, ranges, weights, biases)
print(f"PDF value is {v.numpy()}")

# Print the values of the parametrised PDF on the 5x5x5 grid
cth1, cth2, phi = [ i.flatten() for i in np.meshgrid(
                          np.linspace(-1., 1., 5), 
                          np.linspace(-1., 1., 5), 
                          np.linspace(0., 2.*3.14, 5)
                         ) ]
data = atfi.const(np.stack([cth1, cth2, phi], axis = 1))
v = scale*tfn.multilayer_perceptron(data, ranges, weights, biases)
print("PDF values on 5x5x5 grid (cth1, cth2, phi, v): \n", np.stack([cth1, cth2, phi, v], axis = 1))
