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

   * Arguments of the function. `x` is, as before, the input data tensor. We expect it to have the shape `(N, 2)`, where `N` is the number of events, and `2` is the number of dimensions of the phase space. Then we have the set of named arguments (masses, widths and couplings of resonances) which will either be fixed at generation level or will serve as free parameters of the fit. The last parameter, `switches`, is a Boolean list that allows one to switch on/off components of the amplitude (in this case, the individual resonances and the flat non-resonant term). The switches are manipulated by the functions that run toy MC generation to obtain the weights of individual components, or by the function that calculates fit fractions (branching fractions) of individual components. 
   
   * We are using several `phsp` methods to obtain the vectors of some observables that are functions of the input dataset, such as invariant masses (`m2ab` and `mabc` are equivalen to `x[:,0]` and `x[:,1]`, but `m2ac` is a function of those) and cosines of helicity angles (`cos_helicity_XY(x)` which are more complicated functions of the Dalitz plot variables). 
   
   * The `atfd.breit_wigner_lineshape` we use here, unlike the `atfd.relativistic_breit_wigner()` used in the previous example, includes various corrections to the Breit-Wigner amplitude, such as Blatt-Weisskopf formfactors and mass-dependent width, thus the large number of parameters it needs. 
   
   * A rather weird construction, `atfi.cast_complex(atfi.ones(m2ab))*atfi.complex(atfi.const(5.), atfi.const(0.))` is needed to ensure that we always obtain the vector of `N` values, even if only one non-resonant component switch is turned on. Otherwise, if we used `atfi.complex(atfi.const(5.), atfi.const(0.))`, the function would return a scalar instead of vector. 
