import amplitf.interface as atfi
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace

import tfa.plotting as tfp
import tfa.rootio as tfr
import tfa.neural_nets as tfn

md  = 1.8646
mpi = 0.139
mk  = 0.498

atfi.set_single_precision()

phsp = DalitzPhaseSpace(mpi, mk, mpi, md)

data = tfr.read_tuple("toys.root", ["m2ab", "m2bc"])

import matplotlib.pyplot as plt
tfp.set_lhcb_style(size = 12, usetex = False)
fig, ax = plt.subplots(nrows = 2, ncols = 2, figsize = (8, 6) )

tfn.estimate_density(
    phsp, data,
    ranges = ((0.1, 3.1), (0.1, 3.1)),
    labels = (r"$m^2(K_S^0\pi^+)$",r"$m^2(K_S^0\pi^-)$"),
    learning_rate=0.001,
    training_epochs=10000,
    norm_size=1000000,
    print_step=50,
    display_step=500,
    weight_penalty=1.,
    n_hidden=[32, 8],
    initfile="init.npy",
    outfile="train",
    seed = 1,
    fig = fig, 
    axes = ax, 
)
