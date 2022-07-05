# Copyright 2017 CERN for the benefit of the LHCb collaboration
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#
# Example of three-body amplitude fit for baryonic decay Lambda_b^0 -> D0 p pi-
# The initial Lambda_b0 is assumed to be unpolarised, so only two degrees of
# freedom remain, the fit is two-dimensional.
#
# The amplitude formalism is taken from https://arxiv.org/abs/1701.07873
#

import tensorflow as tf

import sys, os

#sys.path.append("../")
# os.environ["CUDA_VISIBLE_DEVICES"] = ""   # Do not use GPU

import amplitf.interface as atfi
import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
import amplitf.likelihood as atfl
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace

# Import TFA modules
import tfa.toymc as tft
import tfa.plotting as tfp
import tfa.optimisation as tfo


# Calculate orbital momentum for a decay of a particle
# of a given spin and parity to a proton (J^p=1/2+) and a pseudoscalar.
# All the spins and total momenta are expressed in units of 1/2
def OrbitalMomentum(spin, parity):
    l1 = (spin - 1) / 2  # Lowest possible momentum
    p1 = 2 * (l1 % 2) - 1  # p=(-1)^(L1+1), e.g. p=-1 if L=0
    if p1 == parity:
        return l1
    return l1 + 1


# Return the sign in front of the complex coupling
# for amplitudes with baryonic intermediate resonances
# See Eq. (3), page 3 of LHCB-ANA-2015-072,
# https://svnweb.cern.ch/cern/wsvn/lhcbdocs/Notes/ANA/2015/072/drafts/lb2dppi_aman_v3r4.pdf
def CouplingSign(spin, parity):
    jp = 1
    jd = 0
    pp = 1
    pd = -1
    s = 2 * (((jp + jd - spin) / 2 + 1) % 2) - 1
    s *= pp * pd * parity
    return s


if __name__ == "__main__":

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 10)],
        )

    # Default flags that can be overridden by command line options
    norm_grid = 1000
    toy_sample = 100000

    # Masses of initial and final state particles
    mlb = 5.620
    md = 1.865
    mpi = 0.140
    mp = 0.938

    # Create phase space object for 3-body baryonic decay
    # Use only a subrange of D0p invariant masses
    phsp = DalitzPhaseSpace(md, mp, mpi, mlb, mabrange=(0.0, 3.0))

    # Constant parameters of intermediate resonances
    mass_lcst = atfi.const(2.88153)
    width_lcst = atfi.const(0.0058)

    mass_lcx = atfi.const(2.857)
    width_lcx = atfi.const(0.060)

    mass_lcstst = atfi.const(2.945)
    width_lcstst = atfi.const(0.026)

    mass0 = atfi.const(3.0)

    # Blatt-Weisskopf radii
    db = atfi.const(5.0)
    dr = atfi.const(1.5)

    # Slope parameters for exponential nonresonant amplitudes
    alpha12p = tfo.FitParameter("alpha12p", 2.3, 0.0, 10.0, 0.01)
    alpha12m = tfo.FitParameter("alpha12m", 1.0, 0.0, 10.0, 0.01)
    alpha32p = tfo.FitParameter("alpha32p", 2.5, 0.0, 10.0, 0.01)
    alpha32m = tfo.FitParameter("alpha32m", 2.6, 0.0, 10.0, 0.01)

    # List of complex couplings
    couplings = [
        ((atfi.const(1.0), atfi.const(0.0)), (atfi.const(0.0), atfi.const(0.0))),
        (
            (
                tfo.FitParameter("ArX1", -0.38, -10.0, 10.0, 0.01),
                tfo.FitParameter("AiX1", 0.86, -10.0, 10.0, 0.01),
            ),
            (
                tfo.FitParameter("ArX2", 6.59, -10.0, 10.0, 0.01),
                tfo.FitParameter("AiX2", -0.38, -10.0, 10.0, 0.01),
            ),
        ),
        (
            (
                tfo.FitParameter("Ar29401", 0.53, -10.0, 10.0, 0.01),
                tfo.FitParameter("Ai29401", 0.14, -10.0, 10.0, 0.01),
            ),
            (
                tfo.FitParameter("Ar29402", -1.24, -10.0, 10.0, 0.01),
                tfo.FitParameter("Ai29402", 0.02, -10.0, 10.0, 0.01),
            ),
        ),
        (
            (
                tfo.FitParameter("Ar12p1", 0.05, -10.0, 10.0, 0.01),
                tfo.FitParameter("Ai12p1", 0.23, -10.0, 10.0, 0.01),
            ),
            (
                tfo.FitParameter("Ar12p2", -0.16, -10.0, 10.0, 0.01),
                tfo.FitParameter("Ai12p2", -2.86, -10.0, 10.0, 0.01),
            ),
        ),
        (
            (
                tfo.FitParameter("Ar12m1", 1.17, -10.0, 10.0, 0.01),
                tfo.FitParameter("Ai12m1", 0.76, -10.0, 10.0, 0.01),
            ),
            (
                tfo.FitParameter("Ar12m2", -2.55, -10.0, 10.0, 0.01),
                tfo.FitParameter("Ai12m2", 3.86, -10.0, 10.0, 0.01),
            ),
        ),
        (
            (
                tfo.FitParameter("Ar32p1", 0.0, -100.0, 100.0, 0.01),
                tfo.FitParameter("Ai32p1", 0.0, -100.0, 100.0, 0.01),
            ),
            (
                tfo.FitParameter("Ar32p2", 0.0, -100.0, 100.0, 0.01),
                tfo.FitParameter("Ai32p2", 0.0, -100.0, 100.0, 0.01),
            ),
        ),
        (
            (
                tfo.FitParameter("Ar32m1", 0.95, -10.0, 10.0, 0.01),
                tfo.FitParameter("Ai32m1", -0.45, -10.0, 10.0, 0.01),
            ),
            (
                tfo.FitParameter("Ar32m2", -2.27, -10.0, 10.0, 0.01),
                tfo.FitParameter("Ai32m2", 0.95, -10.0, 10.0, 0.01),
            ),
        ),
    ]

    pars = [alpha12p, alpha12m, alpha32p, alpha32m] + [
        k for i in couplings for j in i for k in j if isinstance(k, tfo.FitParameter)
    ]

    # Model description

    @atfi.function
    def model(x, pars):

        m2dp = phsp.m2ab(x)
        m2ppi = phsp.m2bc(x)

        p4d, p4p, p4pi = phsp.final_state_momenta(m2dp, m2ppi)
        dp_theta_r, dp_phi_r, dp_theta_d, dp_phi_d = atfk.helicity_angles_3body(
            p4d, p4p, p4pi
        )

        # List of intermediate resonances corresponds to arXiv link above.
        resonances = [
            (
                atfd.breit_wigner_lineshape(
                    m2dp,
                    mass_lcst,
                    width_lcst,
                    md,
                    mp,
                    mpi,
                    mlb,
                    dr,
                    db,
                    OrbitalMomentum(5, 1),
                    2,
                ),
                5,
                1,
                (1., 0.),
                (0., 0.)
            ),
            (
                atfd.breit_wigner_lineshape(
                    m2dp,
                    mass_lcx,
                    width_lcx,
                    md,
                    mp,
                    mpi,
                    mlb,
                    dr,
                    db,
                    OrbitalMomentum(3, 1),
                    1,
                ),
                3,
                1,
                (pars["ArX1"], pars["AiX1"]),
                (pars["ArX2"], pars["AiX2"]),
            ),
            (
                atfd.breit_wigner_lineshape(
                    m2dp,
                    mass_lcstst,
                    width_lcstst,
                    md,
                    mp,
                    mpi,
                    mlb,
                    dr,
                    db,
                    OrbitalMomentum(3, -1),
                    1,
                ),
                3,
                -1,
                (pars["Ar29401"], pars["Ai29401"]),
                (pars["Ar29402"], pars["Ai29402"]),
            ),
            (
                atfd.exponential_nonresonant_lineshape(
                    m2dp, mass0, pars["alpha12p"], md, mp, mpi, mlb, OrbitalMomentum(1, 1), 0
                ),
                1,
                1,
                (pars["Ar12p1"], pars["Ai12p1"]),
                (pars["Ar12p2"], pars["Ai12p2"]),
            ),
            (
                atfd.exponential_nonresonant_lineshape(
                    m2dp, mass0, pars["alpha12m"], md, mp, mpi, mlb, OrbitalMomentum(1, -1), 0
                ),
                1,
                -1,
                (pars["Ar12m1"], pars["Ai12m1"]),
                (pars["Ar12m2"], pars["Ai12m2"]),
            ),
            (
                atfd.exponential_nonresonant_lineshape(
                    m2dp, mass0, pars["alpha32p"], md, mp, mpi, mlb, OrbitalMomentum(3, 1), 1
                ),
                3,
                1,
                (pars["Ar32p1"], pars["Ai32p1"]),
                (pars["Ar32p2"], pars["Ai32p2"]),
            ),
            (
                atfd.exponential_nonresonant_lineshape(
                    m2dp, mass0, pars["alpha32m"], md, mp, mpi, mlb, OrbitalMomentum(3, -1), 1
                ),
                3,
                -1,
                (pars["Ar32m1"], pars["Ai32m1"]),
                (pars["Ar32m2"], pars["Ai32m2"]),
            ),
        ]

        density = atfi.const(0.0)

        # Decay density is an incoherent sum over initial and final state polarisations
        # (assumong no polarisation for Lambda_b^0), and for each polarisation combination
        # it is a coherent sum over intermediate states (including two polarisations of
        # the intermediate resonance).
        for pol_lb in [-1, 1]:
            for pol_p in [-1, 1]:
                ampl = atfi.complex(atfi.const(0.0), atfi.const(0.0))
                for r in resonances:
                    lineshape = r[0]
                    spin = r[1]
                    parity = r[2]
                    if pol_p == -1:
                        sign = CouplingSign(spin, parity)
                        coupling1 = atfi.complex(atfi.const(r[3][0]), atfi.const(r[3][1])) * sign
                        coupling2 = atfi.complex(atfi.const(r[4][0]), atfi.const(r[4][1])) * sign
                    else:
                        coupling1 = atfi.complex(atfi.const(r[3][0]), atfi.const(r[3][1]))
                        coupling2 = atfi.complex(atfi.const(r[4][0]), atfi.const(r[4][1]))
                    ampl += (
                        coupling1
                        * lineshape
                        * atfk.helicity_amplitude_3body(
                            dp_theta_r,
                            dp_phi_r,
                            dp_theta_d,
                            dp_phi_d,
                            1,
                            spin,
                            pol_lb,
                            1,
                            0,
                            pol_p,
                            0,
                        )
                    )
                    ampl += (
                        coupling2
                        * lineshape
                        * atfk.helicity_amplitude_3body(
                            dp_theta_r,
                            dp_phi_r,
                            dp_theta_d,
                            dp_phi_d,
                            1,
                            spin,
                            pol_lb,
                            -1,
                            0,
                            pol_p,
                            0,
                        )
                    )
                density += atfd.density(ampl)

        return density

    def toy_model(x) : 
      return model(x, {p.name : p.init_value for p in pars} )

    atfi.set_seed(2)

    # Produce normalisation sample (rectangular 2D grid of points)
    norm_sample = phsp.rectangular_grid_sample(norm_grid, norm_grid)
    print("Normalisation sample size = ", norm_sample.shape)
    print(norm_sample)

    # Create toy MC data sample
    data_sample = tft.run_toymc(toy_model, phsp, toy_sample, 0., chunk=1000000)
    print(data_sample)

    def nll(data, norm):
        @atfi.function
        def _nll(pars):
            return atfl.unbinned_nll(
                model(data, pars), atfl.integral(model(norm, pars))
            )
        return _nll

    result = tfo.run_minuit(nll(data_sample, norm_sample), pars)

    # Store fit result in a text file
    print(result)

    print(f"{result['time']/result['func_calls']} sec per function call")
