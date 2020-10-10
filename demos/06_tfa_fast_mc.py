import amplitf.interface as atfi
import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
import amplitf.likelihood as atfl
from amplitf.phasespace.four_body_angular_phasespace import FourBodyAngularPhaseSpace

import tfa.toymc as tft
import tfa.plotting as tfp
import tfa.optimisation as tfo
import tfa.rootio as tfr

import tensorflow as tf

md  = atfi.const(1.8646)
mmu = atfi.const(0.1056)
mk  = atfi.const(0.498)
mphi = atfi.const(1.020)
mjpsi = atfi.const(3.097)

npoints = 3000000

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


phsp = FourBodyAngularPhaseSpace()

def final_state_momenta(data) : 
  cos_theta_jpsi = phsp.cos_theta1(data)
  cos_theta_phi = phsp.cos_theta2(data)
  phi = phsp.phi(data)

  p0 = atfk.two_body_momentum(md, mjpsi, mphi)
  pjpsi = atfk.two_body_momentum(mjpsi, mmu, mmu)
  pphi = atfk.two_body_momentum(mphi, mk, mk)

  zeros = atfi.zeros(phi)
  ones = atfi.ones(phi)

  p3jpsi = atfk.rotate_euler(atfk.vector(zeros, zeros, pjpsi*ones), zeros, atfi.acos(cos_theta_jpsi), zeros)
  p3phi = atfk.rotate_euler(atfk.vector(zeros, zeros, pphi*ones), zeros, atfi.acos(cos_theta_phi), phi)

  ejpsi = atfi.sqrt(p0 ** 2 + mjpsi ** 2)
  ephi = atfi.sqrt(p0 ** 2 + mphi ** 2)
  v0jpsi = atfk.vector(zeros, zeros, p0 / ejpsi * ones)
  v0phi = atfk.vector(zeros, zeros, -p0 / ephi * ones)

  p4mu1 = atfk.lorentz_boost(atfk.lorentz_vector( p3jpsi, atfi.sqrt(mmu ** 2 + pjpsi ** 2)*ones), v0jpsi)
  p4mu2 = atfk.lorentz_boost(atfk.lorentz_vector(-p3jpsi, atfi.sqrt(mmu ** 2 + pjpsi ** 2)*ones), v0jpsi)
  p4k1  = atfk.lorentz_boost(atfk.lorentz_vector( p3phi,  atfi.sqrt(mk ** 2 + pphi ** 2)*ones), v0phi)
  p4k2  = atfk.lorentz_boost(atfk.lorentz_vector(-p3phi,  atfi.sqrt(mk ** 2 + pphi ** 2)*ones), v0phi)

  return (p4mu1, p4mu2, p4k1, p4k2)

toy_data = phsp.uniform_sample(npoints)
moms = final_state_momenta(toy_data)

rnd = tf.random.uniform((len(toy_data), 6), dtype = tf.float64)
boosted_moms = random_rotation_and_boost(moms, rnd)

print(f"Number of events generated: {len(boosted_moms[0])}")

filtered_data = toy_data[
  (atfk.pt(boosted_moms[0])>2.) & 
  (atfk.pt(boosted_moms[1])>2.) & 
  (atfk.scalar_product(atfk.unit_vector(boosted_moms[0]), atfk.unit_vector(boosted_moms[1]))<0.9999) & 
  (atfk.scalar_product(atfk.unit_vector(boosted_moms[0]), atfk.unit_vector(boosted_moms[2]))<0.9999) & 
  (atfk.scalar_product(atfk.unit_vector(boosted_moms[0]), atfk.unit_vector(boosted_moms[3]))<0.9999) & 
  (atfk.scalar_product(atfk.unit_vector(boosted_moms[1]), atfk.unit_vector(boosted_moms[2]))<0.9999) & 
  (atfk.scalar_product(atfk.unit_vector(boosted_moms[1]), atfk.unit_vector(boosted_moms[3]))<0.9999) & 
  (atfk.scalar_product(atfk.unit_vector(boosted_moms[2]), atfk.unit_vector(boosted_moms[3]))<0.9999)
]

print(f"Number of events after filtering: {len(filtered_data)}")

print(f"Filtered data: {filtered_data}")

# Write Dalitz plot entries to ROOT file
tfr.write_tuple("toys.root", filtered_data, ["cth1", "cth2", "phi"])

# Plot the generated 2D distribution. 
# Initialise matplotlib. 
import matplotlib.pyplot as plt
tfp.set_lhcb_style(size = 12, usetex = False)   # Adjust plotting style for LHCb papers
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4, 3) )  # Single subplot on the figure

# Plot 2D histogram from the toy MC sample
tfp.plot_distr2d(filtered_data[:,0], filtered_data[:,1], bins = (50, 50), 
                 ranges = ((-1., 1.), (-1., 1.)), 
                 fig = fig, ax = ax, labels = (r"$\cos\theta_{\mu\mu}$", r"$\cos\theta_{KK}$"))

# Show the plot
plt.tight_layout(pad=1., w_pad=1., h_pad=1.)
plt.show()
