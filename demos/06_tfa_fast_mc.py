import amplitf.interface as atfi
import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
import amplitf.likelihood as atfl
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace

import tfa.toymc as tft
import tfa.plotting as tfp
import tfa.optimisation as tfo
import tfa.rootio as tfr

import tensorflow as tf

md  = 1.8646
mpi = 0.139
mk  = 0.498

npoints = 10000000

def random_rotation_and_boost(moms, rnd) : 
  """
    Apply random boost and rotation to the list of 4-vectors
      moms : list of 4-vectors
      rnd  : random array of shape (N, 6), where N is the length of 4-vector array
  """
  pt  = -5.*atfi.log(rnd[:,0])
  eta = rnd[:,1]*3. + 2.
  phi = rnd[:,2]*2.*atfi.pi()

  theta = 2.*atfi.atan(atfi.exp(-eta)) # Theta angle
  p  = pt/atfi.sin(theta)              # Full momentum
  e  = atfi.sqrt(p**2 + md**2)         # Energy 

  px = p*atfi.sin(theta)*atfi.sin(phi) # 3-momentum of initial particle
  py = p*atfi.sin(theta)*atfi.cos(phi)
  pz = p*atfi.cos(theta)

  boost = atfk.lorentz_vector( atfk.vector(px, py, pz), e)

  rot_theta = atfi.acos(rnd[:,3]*2.-1.)
  rot_phi = rnd[:,4]*2.*atfi.pi()
  rot_psi = rnd[:,5]*2.*atfi.pi()

  moms1 = []
  for m in moms : 
    m1 = atfk.rotate_lorentz_vector(m, rot_phi, rot_theta, rot_psi)
    moms1 += [ atfk.boost_from_rest(m1, boost) ]

  return moms1


phsp = DalitzPhaseSpace(mpi, mk, mpi, md)

toy_data = phsp.uniform_sample(npoints)
moms = phsp.final_state_momenta(phsp.m2ab(toy_data), phsp.m2bc(toy_data))

rnd = tf.random.uniform((len(toy_data), 6), dtype = tf.float64)
boosted_moms = random_rotation_and_boost(moms, rnd)

print(f"Number of events generated: {len(boosted_moms[0])}")

filtered_moms = [ mom[(atfk.pt(boosted_moms[0])>2.) & (atfk.pt(boosted_moms[2])>2.)] for mom in boosted_moms ]

print(f"Number of events after filtering: {len(filtered_moms[0])}")

filtered_data = phsp.from_vectors(atfk.mass_squared(filtered_moms[0] + filtered_moms[1]), 
                                  atfk.mass_squared(filtered_moms[1] + filtered_moms[2]))

print(f"Filtered Dalitz plot data: {filtered_data}")

# Write Dalitz plot entries to ROOT file
tfr.write_tuple("toys.root", filtered_data, ["m2ab", "m2bc"])

# Plot the generated 2D distribution. 
# Initialise matplotlib. 
import matplotlib.pyplot as plt
tfp.set_lhcb_style(size = 12, usetex = False)   # Adjust plotting style for LHCb papers
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4, 3) )  # Single subplot on the figure

# Plot 2D histogram from the toy MC sample
tfp.plot_distr2d(filtered_data[:,0], filtered_data[:,1], bins = (50, 50), 
                 ranges = ((0.3, 3.1), (0.3, 3.1)), 
                 fig = fig, ax = ax, labels = (r"$m^2(K_S^0\pi^+)$", r"$m^2(K_S^0\pi^-)$"), 
                 units = ("MeV$^2$", "MeV$^2$"))

# Show the plot
plt.tight_layout(pad=1., w_pad=1., h_pad=1.)
plt.show()
