# Dalitz plot fit using TensorFlowAnalysis

Example is available here: https://github.com/apoluekt/TFA2/blob/master/demos/04_tfa_dalitz_fit.py

Now we will do some useful stuff: a real Dalitz plot fit. We will generate a sample of <img src="https://render.githubusercontent.com/render/math?math=D^{0}\to K_{S}^{0}\pi^{%2B}\pi^{-}"> decays according to a rather simple amplitude model with only 3 intermediate resonances, and then fit it back with the unbinned maximnum likelihood fit. 

The phase space for Dalitz plot (2-dimensional) is declared by `DalitzPhaseSpace` object, which takes the masses of final and initial states in the constructor: 
```python
phsp = DalitzPhaseSpace(mpi, mk, mpi, md)
```

Let's declare the function that will act as a model for both generation and fitting: 
```python
def model(x, mrho, wrho, mkst, wkst, a1r, a1i, a2r, a2i, a3r, a3i, switches) : 

    a1 = atfi.complex(a1r, a1i)
    a2 = atfi.complex(a2r, a2i)
    a3 = atfi.complex(a3r, a3i)

    m2ab = phsp.m2ab(x)
    m2bc = phsp.m2bc(x)
    m2ac = phsp.m2ac(x)

    hel_ab = phsp.cos_helicity_ab(x)
    hel_bc = phsp.cos_helicity_bc(x)
    hel_ac = phsp.cos_helicity_ac(x)

    ampl = atfi.complex(atfi.const(0.), atfi.const(0.))

    if switches[0] : 
      ampl += a1*atfd.breit_wigner_lineshape(m2ab, mkst,  wkst,  mpi, mk, mpi, md, dr, dd, 1, 1)*atfd.helicity_amplitude(hel_ab, 1)
    if switches[1] : 
      ampl += a2*atfd.breit_wigner_lineshape(m2bc, mkst,  wkst,  mpi, mk, mpi, md, dr, dd, 1, 1)*atfd.helicity_amplitude(hel_bc, 1)
    if switches[2] : 
      ampl += a3*atfd.breit_wigner_lineshape(m2ac, mrho,  wrho,  mpi, mpi, mk, md, dr, dd, 1, 1)*atfd.helicity_amplitude(hel_ac, 1)
    if switches[3] : 
      ampl += atfi.cast_complex(atfi.ones(m2ab))*atfi.complex(atfi.const(5.), atfi.const(0.))

    return atfd.density( ampl )
```
We have K* and rho resonances in each of the 3 channels, Ks pi+, Ks pi- and pi+ pi-. The structure of the function needs some explanations: 

   * Arguments of the function. `x` is, as before, the input data tensor. We expect it to have the shape `(N, 2)`, where `N` is the number of events, and `2` is the number of dimensions of the phase space. Then we have the set of named arguments (masses, widths and couplings of resonances) which will either be fixed at generation level or will serve as free parameters of the fit. The last parameter, `switches`, is a Boolean list that allows one to switch on/off components of the amplitude (in this case, the individual resonances and the flat non-resonant term). The switches are manipulated by the function that runs toy MC generation to obtain the weights of individual components, or by the function that calculates fit fractions (branching fractions) of components. 
   
   * We are using several `phsp` methods to obtain the vectors of some observables that are functions of the input dataset, such as invariant masses (`m2ab` and `mabc` are equivalent to `x[:,0]` and `x[:,1]`, but `m2ac` is a function of those) and cosines of helicity angles (`cos_helicity_XY(x)` which are more complicated functions of the Dalitz plot variables). 
   
   * The `atfd.breit_wigner_lineshape` we use here, unlike the `atfd.relativistic_breit_wigner()` used in the previous example, includes various corrections to the Breit-Wigner amplitude, such as Blatt-Weisskopf formfactors and mass-dependent width, thus the large number of parameters it needs. 
   
   * A rather weird construction, `atfi.cast_complex(atfi.ones(m2ab))*atfi.complex(atfi.const(5.), atfi.const(0.))` is needed to ensure that we always obtain the vector of `N` values, even if only one non-resonant component switch is turned on. Otherwise, if we used `atfi.complex(atfi.const(5.), atfi.const(0.))`, the function would return a scalar instead of vector. 

The function `model` is the most generic function which is then used in the derived functions that are called by the respective parts of `TFA`. The first one of those is the model for toy MC generation, which has only one argument, the input tensor `x` (we have already used this form in the previous example): 
```python 
def toymc_model(x) : 
  return model(x, switches = 4*[1], 
               mrho = atfi.const(0.770), wrho = atfi.const(0.150), 
               mkst = atfi.const(0.892), wkst = atfi.const(0.050), 
               a1r = atfi.const(1.0), a1i = atfi.const(0.), 
               a2r = atfi.const(0.5), a2i = atfi.const(0.), 
               a3r = atfi.const(2.), a3i = atfi.const(0.))
```
In this function, we fix the parameters of the model to their "true" values. All the switches are set to `1` (all components are enabled). 

The second derived function is the fit model: 
```python
@atfi.function
def fit_model(x, pars) : 
  return model(x, **pars, switches = 4*[1])
```
In addition to the input tensor `x`, it takes the second argument, the dictionary of fit parameters `pars` in the form `{'name' : value}`. Since the parameters in `model` are named, we can use Python [argument unpacking](https://www.geeksforgeeks.org/packing-and-unpacking-arguments-in-python/) syntax when calling it with `**par`. 

This function is used in the unbinned negative log likelihood function `nll`: 
```python
def nll(data, norm) : 
  @atfi.function
  def _nll(pars) : 
    return atfl.unbinned_nll(fit_model(data, pars), atfl.integral(fit_model(norm, pars)))
  return _nll
```
which returns the function with the only argument `par` (dictionary of fit parameters) to be minimised in the call `tfo.run_minuit()`. 
```python
result = tfo.run_minuit(nll(toy_sample, norm_sample), pars)
```
Unbinned likelihood above requires an integral of the PDF, that is calculated by summing the PDF values over the uniformly distributed random sample (MC integration) `norm` created with `phsp.uniform_sample()` call. 

The call to `tfo.run_minuit` needs a list of fit parameters which are defined by the object `tfo.FitParameter`
```python
pars = [
  tfo.FitParameter("mrho", 0.770, 0.7, 0.9), 
  tfo.FitParameter("wrho", 0.150, 0.05, 0.2), 
  ...
]
```
The arguments of the constructor are: parameter name (should be the same as in the definition of `func`), initial value, lower and upper limits. 

`run_minuit` uses `iminuit` libraray to perform minimisation. By default, it will use the TF ability to evaluate __analytical gradient__ of the minimised function, which greatly reduces the number of steps needed for the fit to convertge, although requires more memory. The optional argument `use_gradient` controls this behaviour

> __Excercise__: try adding `use_gradient = False` to the `run_minuit` call. 

The output of the `run_minuit` call is the Python dictionary with various infromation, including fit result and uncertainties, the minimised NLL value, number of calls, _etc_ (see the output of `print(result)`). 

At the end of the script, we are calculating the fit fractions of each component. This is finally where we will use component switches. First, we prepare a dictionary of fitted results in the form `{name : value}` from the `result` dictionary: 
```python
fitted_pars = { p : atfi.const(v[0]) for p,v in result["params"].items() }
```
Then we declare the third function derived from our `model`: 
```python
def fitted_model(x, switches = 4*[1]) :
  return model(x, **fitted_pars, switches = switches)
```
for which the two arguments are the data tensor `x` and the list of switches. We should provide the default value for the list of switches, which will be used by the function `calculate_fit_fraction` to determine how many components there are. 
