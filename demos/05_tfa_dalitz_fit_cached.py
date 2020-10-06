import amplitf.interface as atfi
import amplitf.kinematics as atfk
import amplitf.dynamics as atfd
import amplitf.likelihood as atfl
from amplitf.phasespace.dalitz_phasespace import DalitzPhaseSpace

import tfa.toymc as tft
import tfa.plotting as tfp
import tfa.optimisation as tfo

ntoys = 100000   # Number of points to generate 
nnorm = 1000000  # Number of normalisation points

mk  = atfi.const(0.498)
mpi = atfi.const(0.139)
md  = atfi.const(1.8646)

dd = atfi.const(5.0)
dr = atfi.const(1.5)

mkst = atfi.const(0.892)
wkst = atfi.const(0.050)
mrho = atfi.const(0.770)
wrho = atfi.const(0.150)

phsp = DalitzPhaseSpace(mpi, mk, mpi, md)

def model(x) : 

  m2ab = phsp.m2ab(x)
  m2bc = phsp.m2bc(x)
  m2ac = phsp.m2ac(x)

  hel_ab = atfd.helicity_amplitude(phsp.cos_helicity_ab(x), 1)
  hel_bc = atfd.helicity_amplitude(phsp.cos_helicity_bc(x), 1)
  hel_ac = atfd.helicity_amplitude(phsp.cos_helicity_ac(x), 1)

  bw1 = atfd.breit_wigner_lineshape(m2ab, mkst,  wkst,  mpi, mk, mpi, md, dr, dd, 1, 1)
  bw2 = atfd.breit_wigner_lineshape(m2bc, mkst,  wkst,  mpi, mk, mpi, md, dr, dd, 1, 1)
  bw3 = atfd.breit_wigner_lineshape(m2ac, mrho,  wrho,  mpi, mpi, mk, md, dr, dd, 1, 1)

  def _model(a1r, a1i, a2r, a2i, a3r, a3i, switches = 4*[1]) : 

    a1 = atfi.complex(a1r, a1i)
    a2 = atfi.complex(a2r, a2i)
    a3 = atfi.complex(a3r, a3i)

    ampl = atfi.complex(atfi.const(0.), atfi.const(0.))

    if switches[0] : 
      ampl += a1*bw1*hel_ab
    if switches[1] : 
      ampl += a2*bw2*hel_bc
    if switches[2] : 
      ampl += a3*bw3*hel_ac
    if switches[3] : 
      ampl += atfi.cast_complex(atfi.ones(m2ab))*atfi.complex(atfi.const(5.), atfi.const(0.))

    return atfd.density( ampl )

  return _model

def toymc_model(x, switches = 4*[1]) : 
  return model(x)(switches = switches, 
               a1r = atfi.const(1.0), a1i = atfi.const(0.), 
               a2r = atfi.const(0.5), a2i = atfi.const(0.), 
               a3r = atfi.const(2.), a3i = atfi.const(0.))


# TF graph for unbinned negalite log likelihood (the quantity to be minimised)
def nll(data, norm) : 
  data_model = model(data)
  norm_model = model(norm)

  @atfi.function
  def _nll(pars) : 
    return atfl.unbinned_nll(data_model(**pars), atfl.integral(norm_model(**pars)))

  return _nll

toy_sample = tft.run_toymc(toymc_model, phsp, ntoys, maximum = 1.e-20, chunk = 1000000, components = False)

print(toy_sample)

# Plot results
import matplotlib.pyplot as plt
tfp.set_lhcb_style(size = 12, usetex = False)   # Adjust plotting style for LHCb papers
fig, ax = plt.subplots(nrows = 1, ncols = 1, figsize = (4, 3) )  # Single subplot on the figure

# Plot 1D histogram from the toy MC sample
tfp.plot_distr2d(toy_sample[:,0], toy_sample[:,1], bins = (50, 50), 
                 ranges = ((0.3, 3.1), (0.3, 3.1)), 
                 fig = fig, ax = ax, labels = (r"$m^2(K_S^0\pi^+)$", r"$m^2(K_S^0\pi^+)$"), 
                 units = ("MeV$^2$", "MeV$^2$"), log = True)

# Show the plot
plt.tight_layout(pad=1., w_pad=1., h_pad=1.)
plt.show()

norm_sample = phsp.uniform_sample(nnorm)

print(norm_sample)

pars = [
  tfo.FitParameter("a1r", 1.0, -10., 10.), 
  tfo.FitParameter("a1i", 0.0, -10., 10.), 
  tfo.FitParameter("a2r", 0.5, -10., 10.), 
  tfo.FitParameter("a2i", 0.0, -10., 10.), 
  tfo.FitParameter("a3r", 2.0, -10., 10.), 
  tfo.FitParameter("a3i", 0.0, -10., 10.), 
]

# Run MINUIT minimisation of the neg. log likelihood
result = tfo.run_minuit(nll(toy_sample, norm_sample), pars)
print(result)

print(f"{result['time']/result['func_calls']} sec per function call")

fitted_pars = { p : atfi.const(v[0]) for p,v in result["params"].items() }

def fitted_model(x, switches = 4*[1]) :
  return model(x)(**fitted_pars, switches = switches)

ff = tfo.calculate_fit_fractions(fitted_model, norm_sample)
print(ff)