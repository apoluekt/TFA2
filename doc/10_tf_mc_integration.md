# Monte-Carlo integration using pure TensorFlow

We will start with a very simple example of a script, where we will perform Monte-Carlo integration of a 1D Gaussian distribution using pure TensorFlow. The example is available here: https://github.com/apoluekt/TFA2/blob/master/demos/01_mc_integration.py

The structure of the script is rather simple, but let's note a few important points. 

## Loading TF modules

TensorFlow modules are included in python by calling 
```python
import tensorflow as tf
```

## TF functions and dataflow graphs

These lines define the function that we will be integrating: 
```python
@tf.function
def func(x, sigma) : 
   return 1. / (sigma*math.sqrt(2.*math.pi)) * tf.exp( -x**2 / (2.*sigma**2) )
```
This looks like a normal Python (or numpy) function, except that it isn't. There are a couple of notable differences: 
   * Instead of using `math` or `numpy` library functions, we are using TF building blocks (tf.exp in this case). Most of TF mathematical functions are identical or similar in syntax to `math` or `numpy` module functions, so converting code from existing python should be not too difficult. However, rather than working with floating point numbers or numpy arrays, these functions work with _TF tensors_. 

     Note that we don't need to call, e.g. tf.sqrt() to calculate square root in this example because we just need a floating-point constant. 

   * Note the ```@tf.function``` in front of the function definition. It is a _decorator_, which is a pythonic way to use a function that takes a function as a parameter and returns a function as a result (remember, we are dealing with elements of _functional programming_?). Here, ```@tf.function``` is a function defined somewhere in the TF framework, that takes the function `func` we have declared here and returns a _TF dataflow graph_ corresponding to it. The graph defines _what_ the computed should do with the data, without actually doing the corresponding operations (remember, we are using _declarative style_?). The graph can then be compiled and called when necessary (e.g. when we call the `func` function with some concrete input data). 
   
     The behaviour when the program does not do actual calculations before it becomes necessary is called _lazy evaluation_. In principle, the same code could also work without using the `tf.function` decorator, but in this case the computer would run each operation immediately (_eager evaluation_). The difference between eager and lazy modes becomes visible when we need to run `func` multiple times. In eager mode, the TF would run compilation of the code every time the function is called, which makes it slower. In lazy mode, the graph is compiled only once, and is called multiple times without recompilation. 
     
     > __Exercise:__ Try commenting out `@tf.function` in front of `func` or `integral` defintions and see what happens. 

## TF tensors, shapes, indexing

This line creates a unformly distributed random vector of length `npoints`: 
```python
x = tf.random.uniform( (npoints, ), minval = -5., maxval = 5. )
```
The output of this command is a _TF tensor_ (see the output of `print(x)`). This tensor has a representation as a numpy array, which can be obtained 
by calling `x.numpy()`. 

Like in `numpy`, TF tensors are multidimensional rectangular arrays. The _shape_ of the tensor (the number of dimensions and number of elements of each dimension) in the `tf.random.uniform` function is defined by a python _tuple_ passed as the first argument. In our case, it is a tuple of one element (comma is needed to still make it a tuple rather than scalar integer), thus we create a 1D array. 

TensorFlow uses the same style for indexing the multidimensional arrays as `numpy`. A few examples as a reminder: 
   * `array[a,b]` : single element of a 2D array
   * `array[a,:]` : 1D subarray, row `a` of a 2D array
   * `array[:,b]` : 1D subarray, column `b` of a 2D array
   * `array[a1:a2,:]` : 2D slice (rows from a1 to a2) of a 2D array

## Graph tracing and retracing

> __Exercise__: Try using `sigma` instead of `tf.constant(sigma)`. 

