# Example of MC generation and plotting of a 1D Breit-Wigner distribution 
# using TensorFlowAnalysis

# Import AmpliTF modules
import amplitf.interface as atfi
import amplitf.dynamics as atfd
from amplitf.phasespace.rectangular_phasespace import RectangularPhaseSpace

# Import TFA2 modules
import tfa.toymc as tft
import tfa.plotting as tfp

npoints = 1000000   # Number of points to generate 

# Phase space for toy MC generation: 1D interval from 0 to 1500 MeV
phsp = RectangularPhaseSpace( ((0., 1500.), ) )

# Probability density function for generation (Breit-Wigner with parameters)
def bw(m, m0, gamma) : 
  ampl = atfd.relativistic_breit_wigner(m**2, m0, gamma)
  return atfd.density(ampl)

# Toy MC model (single input argument is a 2D tensor, 
# in our case the 2nd dimension is only 1 element)
def model(x) : 
  return bw(x[:,0], atfi.const(770.), atfi.const(150.))

# Run TFA2 toy MC generation
toy_sample = tft.run_toymc(model, phsp, npoints, maximum = 1.e-20, chunk = 1000000).numpy()

print(toy_sample)

# Plot results
import matplotlib.pyplot as plt
tfp.set_lhcb_style(size = 12, usetex = False)   # Adjust plotting style for LHCb papers
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4, 3) )  # Single subplot on the figure

# Plot 1D histogram from the toy MC sample
tfp.plot_distr1d(toy_sample[:,0], bins = 100, range = (0., 1500.), ax = ax, label = r"$m(\pi\pi)$", units = "MeV")

# Show the plot
plt.tight_layout(pad=1., w_pad=1., h_pad=1.)
plt.show()
