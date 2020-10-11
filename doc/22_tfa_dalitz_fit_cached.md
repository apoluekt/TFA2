# Dalitz plot fit with caching, plotting fit results

Example is available here: https://github.com/apoluekt/TFA2/blob/master/demos/05_tfa_dalitz_fit_cached.py

In this example, we will demonstrate an approach to the optimisation of calculations by caching certaing quantities which don't need to be recalculated at every step of minimisation. In addition, we will demonstrate plotting the projections of the fit result, including components of the fitted function. Otherwise, the problem that is solved by the script is the same: generation and fitting of the D0 -> Ks pi+ pi- decay. 

The difference with the previous fit is that we fix the masses and widths of the intermediate rho(770) and K* resonances, and only float the complex couplings. As a result, we don't need to calculate the Breit-Wigner amplitudes at each minimisation step, but only once at the very beginning. However, during toy MC generation, the full amplitude has to be evaluated each time. In this script, the caching of Breit-Wigned amplitudes is done by defining a __higher-order function__ that takes the input tensor `x`, precalculates the amplitude terms for each entry of the input tensor, and returns the function to calculate the PDF with the fit paramaters as arguments: 

```python
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

    ampl = atfi.cast_complex(atfi.ones(m2ab))*atfi.complex(atfi.const(0.), atfi.const(0.))

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
```

The derived functions for toy MC generation, fitting and generation of the fit result is obtained from `model`. 

In addition to what was done in the previous script, we also demonstrate how one can plot 1D projections of the fit result including plotting the individual components of the amplitude. First step is to generate a large toy MC data sample from the fitted model with the argument `components=True` that adds component weights to the dataset: 
```python
fitted_sample = tft.run_toymc(fitted_model, phsp, nnorm, maximum = 1.e-20, chunk = 1000000, components = True)
```
This command generates a 2D tensor where the second dimension, in addition to the 2 Dalitz plot variables, contains the weights of each of the 4 amplitude components (thus the shape of the output tensor is (1000000, 6): 1000000 events and 2+4 variables characterising each event, 2 Dalitz plot variables and 4 component weights). These weights can be passed to the plotting functions `tfp.plot_distr1d_comparison()` in the `cweights` argument that takes the list of vectors of component weights: 
```python
tfp.plot_distr1d_comparison(toy_sample[:,0], fitted_sample[:,0], 
                 cweights = [ fitted_sample[:,2+i] for i in range(4) ], 
                 bins = 50, range = (0.3, 3.1), 
                 ax = ax[0,1], label = r"$m^2(K_S^0\pi^+)$", 
                 units = "MeV$^2$")
```
