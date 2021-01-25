# Example of fast MC to obtain acceptance for the decays B->Jpsi(mumu) phi(KK)

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
import math

# Masses of initial, intermediate and final-state particles
mb = atfi.const(5.367)
mmu = atfi.const(0.1056)
mk = atfi.const(0.498)
mphi = atfi.const(1.020)
mjpsi = atfi.const(3.097)

# Number of events to generate (before filtering)
nevents = 3000000


def random_rotation_and_boost(moms, rnd):
    """
    Apply random boost and rotation to the list of 4-vectors
      moms : list of 4-vectors
      rnd  : random array of shape (N, 6), where N is the length of 4-vector array
    """
    pt = -5.0 * atfi.log(
        rnd[:, 0]
    )  # Random pT, exponential distribution with mean 5 GeV
    eta = rnd[:, 1] * 3.0 + 2.0  # Uniform distribution in pseudorapidity eta
    phi = rnd[:, 2] * 2.0 * atfi.pi()  # Uniform distribution in phi

    theta = 2.0 * atfi.atan(atfi.exp(-eta))  # Theta angle is a function of eta
    p = pt / atfi.sin(theta)  # Full momentum
    e = atfi.sqrt(p ** 2 + mb ** 2)  # Energy of the Bs

    px = (
        p * atfi.sin(theta) * atfi.sin(phi)
    )  # 3-momentum of initial particle (Bs meson)
    py = p * atfi.sin(theta) * atfi.cos(phi)
    pz = p * atfi.cos(theta)

    boost = atfk.lorentz_vector(
        atfk.vector(px, py, pz), e
    )  # Boost vector of the Bs meson

    rot_theta = atfi.acos(
        rnd[:, 3] * 2.0 - 1.0
    )  # Random Euler rotation angles for Bs decay
    rot_phi = rnd[:, 4] * 2.0 * atfi.pi()
    rot_psi = rnd[:, 5] * 2.0 * atfi.pi()

    # Apply rotation and boost to the momenta in input list
    moms1 = []
    for m in moms:
        m1 = atfk.rotate_lorentz_vector(m, rot_phi, rot_theta, rot_psi)
        moms1 += [atfk.boost_from_rest(m1, boost)]

    return moms1


# Phase space for Bs->Jpsi(mumu) phi(KK) decay is a 3D angular space
# given by 2 helicity angles theta1 and theta2, and the
# angle phi between Jpsi->mumu and phi->KK planes
phsp = FourBodyAngularPhaseSpace()

# Calculate final state momenta in Bs rest frame, given the
# input data in the kinematic phase space (tensor data of shape (N, 3),
# where N in the number of events, and 3 is the number of kinematic
# parameters characterising the event)
def final_state_momenta(data):

    # Obtain the vectors of angles from the input tensor using the functions
    # provided by phasespace object
    cos_theta_jpsi = phsp.cos_theta1(data)
    cos_theta_phi = phsp.cos_theta2(data)
    phi = phsp.phi(data)

    # Rest-frame momentum of two-body Bs->Jpsi phi decay
    p0 = atfk.two_body_momentum(mb, mjpsi, mphi)
    # Rest-frame momentum of two-body Jpsi->mu mu decay
    pjpsi = atfk.two_body_momentum(mjpsi, mmu, mmu)
    # Rest-frame momentum of two-body phi->K K decay
    pphi = atfk.two_body_momentum(mphi, mk, mk)

    # Vectors of zeros and ones of the same size as the data sample
    # (needed to use constant values that do not depend on the event)
    zeros = atfi.zeros(phi)
    ones = atfi.ones(phi)

    # 3-vectors of Jpsi->mumu and phi->KK decays (in the corresponding rest frames),
    # rotated by the helicity angles
    p3jpsi = atfk.rotate_euler(
        atfk.vector(zeros, zeros, pjpsi * ones), zeros, atfi.acos(cos_theta_jpsi), zeros
    )
    p3phi = atfk.rotate_euler(
        atfk.vector(zeros, zeros, pphi * ones), zeros, atfi.acos(cos_theta_phi), phi
    )

    ejpsi = atfi.sqrt(p0 ** 2 + mjpsi ** 2)  # Energy of Jpsi in Bs rest frame
    ephi = atfi.sqrt(p0 ** 2 + mphi ** 2)  # Energy of phi in Bs rest frame
    v0jpsi = atfk.vector(
        zeros, zeros, p0 / ejpsi * ones
    )  # 3-vector of Jpsi in Bs rest frame
    v0phi = atfk.vector(
        zeros, zeros, -p0 / ephi * ones
    )  # 3-vector of phi in Bs rest frame

    # Boost momenta of final-state particles into Bs rest frame
    p4mu1 = atfk.lorentz_boost(
        atfk.lorentz_vector(p3jpsi, atfi.sqrt(mmu ** 2 + pjpsi ** 2) * ones), v0jpsi
    )
    p4mu2 = atfk.lorentz_boost(
        atfk.lorentz_vector(-p3jpsi, atfi.sqrt(mmu ** 2 + pjpsi ** 2) * ones), v0jpsi
    )
    p4k1 = atfk.lorentz_boost(
        atfk.lorentz_vector(p3phi, atfi.sqrt(mk ** 2 + pphi ** 2) * ones), v0phi
    )
    p4k2 = atfk.lorentz_boost(
        atfk.lorentz_vector(-p3phi, atfi.sqrt(mk ** 2 + pphi ** 2) * ones), v0phi
    )

    return (p4mu1, p4mu2, p4k1, p4k2)


# Generate uniformly distributed sample of events in the phase space
toy_data = phsp.uniform_sample(nevents)
# Calculate final state momenta in Bs rest frame
moms = final_state_momenta(toy_data)

# Apply random rotation and boost to lab frame
rnd = tf.random.uniform((len(toy_data), 6), dtype=tf.float64)
boosted_moms = random_rotation_and_boost(moms, rnd)

print(f"Number of events generated: {len(boosted_moms[0])}")

# Filter data according to "trigger conditions"
#   (minimun p and pT for tracks, and minimum angles between pairs of tracks).
filtered_data = toy_data[
    (atfk.pt(boosted_moms[0]) > 1.0)
    & (atfk.pt(boosted_moms[1]) > 1.0)
    & (atfk.pt(boosted_moms[2]) > 0.5)
    & (atfk.pt(boosted_moms[3]) > 0.5)
    & (atfk.p(boosted_moms[0]) > 5.0)
    & (atfk.p(boosted_moms[1]) > 5.0)
    & (atfk.p(boosted_moms[2]) > 3.0)
    & (atfk.p(boosted_moms[3]) > 3.0)
    & (
        atfk.scalar_product(
            atfk.unit_vector(boosted_moms[0]), atfk.unit_vector(boosted_moms[1])
        )
        < 0.9999
    )
    & (
        atfk.scalar_product(
            atfk.unit_vector(boosted_moms[0]), atfk.unit_vector(boosted_moms[2])
        )
        < 0.9999
    )
    & (
        atfk.scalar_product(
            atfk.unit_vector(boosted_moms[0]), atfk.unit_vector(boosted_moms[3])
        )
        < 0.9999
    )
    & (
        atfk.scalar_product(
            atfk.unit_vector(boosted_moms[1]), atfk.unit_vector(boosted_moms[2])
        )
        < 0.9999
    )
    & (
        atfk.scalar_product(
            atfk.unit_vector(boosted_moms[1]), atfk.unit_vector(boosted_moms[3])
        )
        < 0.9999
    )
    & (
        atfk.scalar_product(
            atfk.unit_vector(boosted_moms[2]), atfk.unit_vector(boosted_moms[3])
        )
        < 0.9999
    )
]

print(f"Number of events after filtering: {len(filtered_data)}")

print(f"Filtered data: {filtered_data}")

# Write Dalitz plot entries to ROOT file
tfr.write_tuple("toys.root", filtered_data, ["cth1", "cth2", "phi"])

# Plot the generated 2D distribution.
# Initialise matplotlib.
import matplotlib.pyplot as plt

tfp.set_lhcb_style(size=12, usetex=False)  # Adjust plotting style for LHCb papers
fig, ax = plt.subplots(
    nrows=1, ncols=3, figsize=(10, 3)
)  # Single subplot on the figure

# Plot 2D histogram from the toy MC sample
tfp.plot_distr2d(
    filtered_data[:, 0],
    filtered_data[:, 1],
    bins=(50, 50),
    ranges=((-1.0, 1.0), (-1.0, 1.0)),
    fig=fig,
    ax=ax[0],
    labels=(r"$\cos\theta_{\mu\mu}$", r"$\cos\theta_{KK}$"),
)

tfp.plot_distr2d(
    filtered_data[:, 0],
    filtered_data[:, 2],
    bins=(50, 50),
    ranges=((-1.0, 1.0), (0.0, 2.0 * math.pi)),
    fig=fig,
    ax=ax[1],
    labels=(r"$\cos\theta_{\mu\mu}$", r"$\phi$"),
)

tfp.plot_distr2d(
    filtered_data[:, 1],
    filtered_data[:, 2],
    bins=(50, 50),
    ranges=((-1.0, 1.0), (0.0, 2.0 * math.pi)),
    fig=fig,
    ax=ax[2],
    labels=(r"$\cos\theta_{KK}$", r"$\phi$"),
)

# Show the plot
plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
plt.show()
