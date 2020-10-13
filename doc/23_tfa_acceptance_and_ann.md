# Acceptance profile of Bs->Jpsi phi decay and ANNs for its parametrisation (part 1)

This example is based on three scripts that run sequentially: 
   * https://github.com/apoluekt/TFA2/blob/master/demos/06_tfa_fast_mc.py
   * https://github.com/apoluekt/TFA2/blob/master/demos/07_tfa_ann_density_estimate.py
   * https://github.com/apoluekt/TFA2/blob/master/demos/08_tfa_ann_density_use.py

In the first script, https://github.com/apoluekt/TFA2/blob/master/demos/06_tfa_fast_mc.py, we will demonstrate how one can use various kinematical functions of TFA: Lorentz boosts, rotations, _etc_. 

The aim of this scripts is to perform simple ("fast") MC simulation of the decays of Bs mesons into Jpsi(->mu+mu-)phi(->K+K-) four-body final state. The initial Bs mesons will be generated in roughly LHCb production environment, uniformly in the pseudorapidity `eta` and with the exponential distribution in the transverse momentum `pT`. 

<img src="https://www.researchgate.net/profile/Amol_Dighe2/publication/45934706/figure/fig1/AS:669398557790229@1536608461930/The-description-of-the-angles-th-m-K-and-ph-in-the-angular-distribution-of-B-K.png" width="400" align="right">The final state of the decay can be characterised by three angles: two helicity angles of the mu+mu- and K+K- systems (`theta_pipi` and `theta_KK`), as well as the angle `phi` between the mu+mu- and K+K- planes in the Bs rest frame. In our script, we will generate the decays uniformly in this phase space, with the cosines of the helicity angles cos(`theta`) distributed uniformly from -1 to 1, and the angle `phi` uniformly from 0 to 2pi. 

Finally, we will apply the fictional "trigger selection" to our sample, only selecting the events where each of the decay products has transverse momentum `pt` and total momentum `p` above certain threshold in the laboratory frame, and requiring that the angle between each pair of tracks is not smaller than certain value. The resulting distribution of events (expressed in terms of the 3 angles characterising the internal kinematics of the decay) will be not uniform as a result. The 2D projections of this 3D distribution will be plotted. 
