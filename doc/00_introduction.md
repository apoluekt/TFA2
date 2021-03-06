# Introduction

Modern physics analyses in HEP, and in flavour physics in particular, are often dealing with large data sets and complex theoretical models, and thus need efficient computational tools. It appears that many computational challenges specific to HEP can be tackled with modern machine learning frameworks. Here, two software packages, `AmpliTF` and `TensorFlowAnalysis`, are presented, which extend the TensorFlow framework developed by Google (<http://tensorflow.org>) to be used as an efficient compute engine in flavour physics calculations. 

## Vectorised calculations in HEP

Calculations typically performed in HEP analyses mostly have statistical nature, _e.g._, instead of dealing with each event individually, we perform bulk operations on a large dataset as a whole. As a result, most of the calculations can be efficiently __vectorised__ (one can apply the same function element-wise on the full dataset). Typical use cases, such as maximum likelihood fits and parameter scans, Monte-Carlo techniques, etc., all fall in this category. Vectorised code can then run on different architectures where hardware parallelisation is available, such as multicore/multithreaded CPU, graphical processors (GPU), tensor units or even FPGAs. 

## Machine learning frameworks

The kinds of problems that are typically solved in HEP analyses resemble very much those seen in machine learning (ML). For instance, in a typical maximum likelihood fit, one minimises a certain figure of merit (negative log likelihood), which can be a rather complex function (theoretical model) calculated on a given (often large) data set, against a number of tunable parameters. This is pretty similar to a typical machine learning task, where the figure of merit is called "_cost function_" that is calculated on a training data set, and the fitted model is expressed as an artificial neural network with optimisable parameters. 

Machine learning community (which is admittedly much broader than that of HEP) has developed a number of computational frameworks to efficiently deal with their 
problems using various hardware. Given the similarity of the two fields, it seems logical to try to reuse ML products for HEP analyses. There is a number of ML frameworks currently on the market, but not all of them fit sufficiently to physics calculations. The best match for the moment seems to be the __TensorFlow__ library by Google. Its features that are critical for HEP are: 

   * Good support of complex numbers and a rich library of mathematical functions.
   * Support by `sympy` library (Python library for symbolic calculations) in cases when certain math functions are not available out of the box.
   * Large user community, open source.

TensorFlow can run on many architectures, including CPU, NVidia GPU (including multi-GPU configurations), calculations can be distributed to several machines over the network. 

## Imperative, declarative and functional programming

Writing efficient vectorised code (in particular, using TensorFlow) requires a certain change in programming style. The library uses the so-called 
[__declarative programming__](https://en.wikipedia.org/wiki/Declarative_programming) style. Instead of exactly ordering the computer _how_ to perform calculations step-by-step ([__imperative programming__](https://en.wikipedia.org/wiki/Imperative_programming)), the user of TensorFlow describes _what_ has to be solved in terms of specific TensorFlow building blocks, and then gives the control to the framework which will decide how to best solve the problem using available hardware. 

Another concept useful to develop parallelisable programs is [__functional programming__](https://en.wikipedia.org/wiki/Functional_programming). Functional style, in particular, includes writing the code using __pure functions__, i.e. functions, which don't have any internal modifiable state or __side effects__, with the output that is completely determined by the input parameters. When combined with declarative style, this allows the computer to easily identify the dependencies between different parts of the program and decide which parts can be calculated in parallel. 

While TensorFlow does not strictly follow the functional approach (e.g. there are objects like __variables__ which have internal state), the interface of `TensorFlowAnalysis` library is designed with the functional approach in mind, mostly because of the possible extension to ML libraries other than TensorFlow, such as [JAX](https://jax.readthedocs.io/en/latest/) which has a more functional interface. 

## Libraries to use TensorFlow in flavour analyses

While TensorFlow can be used as a general computation engine, it would be useful to have a library of functions that are widely used in HEP calculations, such as operations with 3-vectors and Lorentz vectors, helicity amplitudes, dynamical functions such as Breit-Wigner amplitudes, _etc_. These components are included in the `AmpliTF` library: https://github.com/apoluekt/AmpliTF. This library is a collection of simple functions and classes to help build programs to perform HEP calculations. It is designed in such a way as to make it easier in the future to change TensorFlow to other libraries with `numpy`-based syntax, e.g. JAX, or even make it possible to choose vectorised computation backends. 

Another library, `TensorFlowAnalysis v2`, or `TFA2`, is more tightly bound to TensorFlow and provides the interface between TensorFlow and `iminuit` for minimisation, routines for toy MC generation, plotting with `matplotlib` and simple helper functions for ROOT input/output using `uproot`. It is also home for several demo programs, and this tutorial. 
