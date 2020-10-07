# Introduction

Modern physics analyses in HEP, and in flavour physics in particular, are often dealing with large data sets and complex theoretical models, and thus need efficient computational tools. It appears that many computational challenges specific to HEP can be tackled with modern machine learning frameworks. Here, two software packages, AmpliTF and TensorFlowAnalysis, are presented, which extend the TensorFlow framework developed by Google (http://tensorflow.org) to be used as an efficient compute engine in flavour physics analyses. 

## Vectorised calculations in HEP

Calculations typically performed in HEP analyses mostly have statistical nature, e.g. instead of dealing with each even individually, we perform bulk claculations on a large dataset as a whole. As a result, most of the claculations can be efficiently vectorised (one can apply the same function element-wise on the full dataset). Typical use cases, such as maximum likelihood fits and parameter scans, Monte-Carlo techniques, etc., all fall in this category. 

## Imperative, declarative and functioal programming. 

## Machine learning frameworks

## AmpliTF library

## TFA2 library

