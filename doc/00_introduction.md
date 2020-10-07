# Introduction

Modern physics analyses in HEP, and in flavour physics in particular, are often dealing with large data sets and complex theoretical models, and thus need efficient computational tools. It appears that many computational challenges specific to HEP can be tackled with modern machine learning frameworks. Here, two software packages, AmpliTF and TensorFlowAnalysis, are presented, which extend the TensorFlow framework developed by Google (http://tensorflow.org) to be used as an efficient compute engine in flavour physics analyses. 

## Vectorised calculations in HEP

Calculations typically performed in HEP analyses mostly have statistical nature, e.g. instead of dealing with each even individually, we perform bulk claculations on a large dataset as a whole. As a result, most of the claculations can be efficiently vectorised (one can apply the same function element-wise on the full dataset). Typical use cases, such as maximum likelihood fits and parameter scans, Monte-Carlo techniques, etc., all fall in this category. Vectorised code can then 
run on different architectures where hardware parallelisation is available, such as multicore/multithreaded CPU, graphical processors (GPU), tensor units or even FPGAs. 

## Machine learning frameworks

The kinds of problems that are typically solved in HEP analyses resemble very much those seen in machine learning (ML). For instance, in a typical maximum likelihood fit, one minimises a certain figure of merit (negative log likelihood), which can be a rather complex function (theoretical model) calculated on a given (often large) data set, against a number of tunable parameters. This is pretty similar to a typical machine learning task, where the figure of merit is called "cost function" that is calculated on a training data set, and the fitted model is expressed as an artificial neural network with optimisable parameters. 

Machine learning community (which is admittedly much broader than that of HEP) has developed a number of computational frameworks to efficiently deal with their 
problems using various hardware. Given the similarity of the two fields, it seems logical to try to reuse ML products for HEP analyses. 

## Imperative, declarative and functioal programming

Writing efficient vectorised code requires a certain change in programming style. 

## AmpliTF library

## TFA2 library

