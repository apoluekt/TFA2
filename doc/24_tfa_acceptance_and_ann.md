# Acceptance profile of Bs->Jpsi phi decay and ANNs for its parametrisation (part 2)

This example is based on three scripts that run sequentially: 
   * https://github.com/apoluekt/TFA2/blob/master/demos/06_tfa_fast_mc.py
   * https://github.com/apoluekt/TFA2/blob/master/demos/07_tfa_ann_density_estimate.py
   * https://github.com/apoluekt/TFA2/blob/master/demos/08_tfa_ann_density_use.py

In the second script in this sequence, https://github.com/apoluekt/TFA2/blob/master/demos/07_tfa_ann_density_estimate.py, we are reading the ROOT file with the Bs->Jpsi phi decays generated uniformly at the previous step, and using artificial neural network (ANN) to obtain the functional form of the acceptance (perform density estimation). 

The idea of this technique is very simple. We are using the fully-connected ANN ([__multilayer perceptron__](https://en.wikipedia.org/wiki/Multilayer_perceptron)) with 3 input neurons corresponding to three angular variables of the kinematic phase space of the decay, and one output neuron (for the PDF density) with a few hidden layers, as a way to approximate any function f(x,y,z) of three parameters. 

The trainable parameters of the ANN are the weights (w_ij) and biases (b_i) of neurons. The training procedure consists of finding the configuration of training parameters that minimises a certain __cost function__. In the case of density estimation, we are using the cost funtion that is simply the negative log. likelihood of the ANN response wrt. the scattered data we are parametrising. So, effectively, we are doing the unbinned maximum likelihood fit of the function represented by ANN to our dataset, where the parameters of the ANN are floated. More details about this procedure can be obtained in this preprint: https://arxiv.org/abs/1902.01452. 

