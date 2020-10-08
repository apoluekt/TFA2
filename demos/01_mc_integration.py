# An example of MC integration of a 1D Gaussian distribution using pure TensorFlow v2. 

import math
import tensorflow as tf
from timeit import default_timer as timer

npoints = 100000000  # Number of random points for MC integration

@tf.function
def func(x, sigma) : 
  """
    The function being integrated. 
    Normalised Gaussian distribution with zero mean and RMS sigma
  """
  return 1. / (sigma*math.sqrt(2.*math.pi)) * tf.exp( -x**2 / (2.*sigma**2) )


@tf.function
def integral(x) : 
  """
    Integral (basically, the normalised sum) of values of a vector x
  """
  return tf.reduce_sum(x)/npoints*10.

# Set initial random seed
tf.random.set_seed(1)

# Generate random sample (1D vector) with the uniformly distributed 
# values from -5 to 5
x = tf.random.uniform( (npoints, ), minval = -5., maxval = 5. )
print(x)

# Calculate integral with 10 different values of sigma
for i in range(1, 10) : 
  sigma = 0.1*i   # sigma will change from 0.1 to 0.9

  start = timer()  # Start time

  # Do the actual integration
  y = integral(func(x, tf.constant(sigma) ))

  stop = timer()   # End time

  print(f"sigma={sigma}, integral={y.numpy()}, time={stop-start}")
