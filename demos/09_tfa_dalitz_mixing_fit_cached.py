import amplitf.interface as atfi
import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
import amplitf.likelihood as atfl
import amplitf.mixing as atfm
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace
from amplitf.phasespace.decaytime_phasespace import DecayTimePhaseSpace
from amplitf.phasespace.combined_phasespace import CombinedPhaseSpace

import tfa.toymc as tft
import tfa.plotting as tfp
import tfa.optimisation as tfo

ntoys = 100000  # Number of points to generate
nnorm = 10*ntoys  # Number of normalisation points

mk = atfi.const(0.498)
mpi = atfi.const(0.139)
md = atfi.const(1.8646)

rd = atfi.const(5.0)
rr = atfi.const(1.5)

mkst = atfi.const(0.892)
wkst = atfi.const(0.050)
mrho = atfi.const(0.770)
wrho = atfi.const(0.150)

xmix = atfi.const(0.04) # x10 to show effects with small sample size
ymix = atfi.const(0.064)  # x10 to show effects with small sample size
qop = atfi.const(1.0)
qop_phase = atfi.const(0.0)  # radians

phsp = DalitzPhaseSpace(mpi, mk, mpi, md)
tdz = atfi.const(1.0)
tphsp = DecayTimePhaseSpace(tdz)
c_phsp = CombinedPhaseSpace(phsp, tphsp)


def model_dz_amp(x):

    m2ab = phsp.m2ab(x)
    m2bc = phsp.m2bc(x)
    m2ac = phsp.m2ac(x)

    hel_ab = atfd.helicity_amplitude(phsp.cos_helicity_ab(x), 1)
    hel_bc = atfd.helicity_amplitude(phsp.cos_helicity_bc(x), 1)
    hel_ac = atfd.helicity_amplitude(phsp.cos_helicity_ac(x), 1)

    bw1 = atfd.breit_wigner_lineshape(m2ab, mkst, wkst, mpi, mk, mpi, md, rd, rr, 1, 1)
    bw2 = atfd.breit_wigner_lineshape(m2bc, mkst, wkst, mpi, mk, mpi, md, rd, rr, 1, 1)
    bw3 = atfd.breit_wigner_lineshape(m2ac, mrho, wrho, mpi, mpi, mk, md, rd, rr, 1, 1)

    def _model(a1r, a1i, a2r, a2i, a3r, a3i, switches=4 * [1]):

        a1 = atfi.complex(a1r, a1i)
        a2 = atfi.complex(a2r, a2i)
        a3 = atfi.complex(a3r, a3i)

        ampl = atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
            atfi.const(0.0), atfi.const(0.0)
        )

        if switches[0]:
            ampl += a1 * bw1 * hel_ab
        if switches[1]:
            ampl += a2 * bw2 * hel_bc
        if switches[2]:
            ampl += a3 * bw3 * hel_ac
        if switches[3]:
            ampl += atfi.cast_complex(atfi.ones(m2ab)) * atfi.complex(
                atfi.const(5.0), atfi.const(0.0)
            )

        return ampl

    return _model


def Af(x):
    return model_dz_amp(x)(
        a1r=atfi.const(1.0),
        a1i=atfi.const(0.0),
        a2r=atfi.const(0.5),
        a2i=atfi.const(0.0),
        a3r=atfi.const(2.0),
        a3i=atfi.const(0.0),
        switches=4 * [1],
    )

def Afbar(x):
    return Af(x[:,::-1])


def mixing_model(x):
    # Calculate the amplitudes - cached
    ampl_dz = Af(c_phsp.data1(x))
    ampl_dzb = Afbar(c_phsp.data1(x))
    # Calculate the model from the mixing parameters and time evolution operators
    def _model(x_mix_par, y_mix_par, qop_par, qop_phase_par):
        t = c_phsp.phsp2.t(c_phsp.data2(x))
        tep = atfm.psip(t, y_mix_par, tdz)
        tem = atfm.psim(t, y_mix_par, tdz)
        tei = atfm.psii(t, x_mix_par, tdz)
        # calculate the amplitude density
        dens = atfm.mixing_density(ampl_dz, ampl_dzb, atfi.complex(qop_par * atfi.cos(qop_phase_par), qop_par * atfi.sin(qop_phase_par)), tep, tem, tei)
        return dens
    return _model


def toymc_model(x, switches=4 * [1]):
    return mixing_model(x)(
        x_mix_par=xmix,
        y_mix_par=ymix,
        qop_par=qop,
        qop_phase_par=qop_phase
    )


# TF graph for unbinned negalite log likelihood (the quantity to be minimised)
def nll(data, norm):
    data_model = mixing_model(data)
    norm_model = mixing_model(norm)

    @atfi.function
    def _nll(pars):
        return atfl.unbinned_nll(data_model(**pars), atfl.integral(norm_model(**pars)))

    return _nll


toy_sample = tft.run_toymc(
    toymc_model, c_phsp, ntoys, maximum=1.0e-20, chunk=1000000, components=False
)

print(toy_sample)

norm_sample = c_phsp.uniform_sample(nnorm)

print(norm_sample)

pars = [
    tfo.FitParameter("x_mix_par", 0.04, -0.2, 0.2),
    tfo.FitParameter("y_mix_par", 0.06, -0.2, 0.2),
    tfo.FitParameter("qop_par", 1.0, 0.8, 1.2),
    tfo.FitParameter("qop_phase_par", 0.0, -3.15, 3.15),
]
fixed_pars = ['qop_par','qop_phase_par']  # Fix qop and qop_phase for this example
for p in pars:
    if p.name in fixed_pars:
        p.fix()

# Run MINUIT minimisation of the neg. log likelihood
result = tfo.run_minuit(nll(toy_sample, norm_sample), pars)
print(result)

cov = result['covariance']

print(f"{result['time']/result['func_calls']} sec per function call")

fitted_pars = {p: atfi.const(v[0]) for p, v in result["params"].items()}


def fitted_model(x):
    return mixing_model(x)(**fitted_pars)


# ff = tfo.calculate_fit_fractions(fitted_model, norm_sample)
# print(ff)

fitted_sample = tft.run_toymc(
    fitted_model, c_phsp, nnorm, maximum=1.0e-20, chunk=1000000, components=True
)

# Plot results
import matplotlib.pyplot as plt

tfp.set_lhcb_style(size=12, usetex=False)  # Adjust plotting style for LHCb papers
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 6))  # Single subplot on the figure

# Plot 1D histogram from the toy MC sample
tfp.plot_distr2d(
    toy_sample[:, 0],
    toy_sample[:, 1],
    bins=(50, 50),
    ranges=((0.3, 3.1), (0.3, 3.1)),
    fig=fig,
    ax=ax[0, 0],
    labels=(r"$m^2(K_S^0\pi^+)$", r"$m^2(K_S^0\pi^-)$"),
    units=("MeV$^2$", "MeV$^2$"),
    log=True,
)

tfp.plot_distr1d_comparison(
    toy_sample[:, 0],
    fitted_sample[:, 0],
    #cweights=[fitted_sample[:, 3 + i] for i in range(4)],
    bins=50,
    range=(0.3, 3.1),
    ax=ax[0, 1],
    label=r"$m^2(K_S^0\pi^+)$",
    units="MeV$^2$",
    legend=False
)

tfp.plot_distr1d_comparison(
    toy_sample[:, 1],
    fitted_sample[:, 1],
    #cweights=[fitted_sample[:, 3 + i] for i in range(4)],
    bins=50,
    range=(0.3, 3.1),
    ax=ax[1, 0],
    label=r"$m^2(K_S^0\pi^-)$",
    units="MeV$^2$",
    legend=False
)

tfp.plot_distr1d_comparison(
    toy_sample[:, 2],
    fitted_sample[:, 2],
    #cweights=[fitted_sample[:, 3 + i] for i in range(4)],
    bins=50,
    range=(0., 5),
    ax=ax[0, 2],
    label=r"t",
    units=r"$\tau_{D^0}$",
    log=True,
    legend=False
)

tfp.plot_distr1d_comparison(
    c_phsp.phsp1.m2ac(c_phsp.data1(toy_sample)),
    c_phsp.phsp1.m2ac(c_phsp.data1(fitted_sample)),
    #cweights=[fitted_sample[:, 3 + i] for i in range(4)],
    bins=50,
    range=(0.05, 1.9),
    ax=ax[1, 1],
    label=r"$m^2(\pi^+\pi^-)$",
    units="MeV$^2$",
    legend_ax=ax[1,2]
)

# Show the plot
plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
plt.show()
