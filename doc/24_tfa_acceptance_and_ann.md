# Acceptance profile of Bs->Jpsi phi decay and ANNs for its parametrisation (part 2)

This example is based on three scripts that run sequentially: 
   * https://github.com/apoluekt/TFA2/blob/master/demos/06_tfa_fast_mc.py
   * https://github.com/apoluekt/TFA2/blob/master/demos/07_tfa_ann_density_estimate.py
   * https://github.com/apoluekt/TFA2/blob/master/demos/08_tfa_ann_density_use.py

In the second script in this sequence, https://github.com/apoluekt/TFA2/blob/master/demos/07_tfa_ann_density_estimate.py, we are reading the ROOT file with the Bs->Jpsi phi decays generated uniformly at the previous step, and using artificial neural network (ANN) to obtain the functional form of the acceptance (perform density estimation). 
