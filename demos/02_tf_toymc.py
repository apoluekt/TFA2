# An example of toy MC generation of a 1D Breit-Wigner distribution using pure TensorFlow v2

import tensorflow as tf

npoints = 100000000  # Number of random points

def bw(m, m0, gamma) : 
  """
    Breit-Wigner distribution
      m     : the variable to be generated
      m0    : resonance mass
      gamma : resonance width
  """
  ampl = tf.complex(m0*m0 - m*m, -m0*gamma)
  return tf.abs(ampl)**2

# Set initial random seed
tf.random.set_seed(1)

# Generate random sample (1D vector) with the uniformly distributed values
m = tf.random.uniform( (npoints, ), minval = 0., maxval = 1500. )

# Calculate the vector of 1D Breit-Wigner densities for a generated dataset
#   Use rho0 resonance parameters
y = bw(m, tf.constant(770.), tf.constant(150.) )

# Calculate the maximum of the PDF 
ymax = tf.reduce_max(y)

# Generate the uniform random sample from 0 to maximum
r = tf.random.uniform( (npoints, ), minval = 0., maxval = ymax )

# Filter only the points with the random variable less than PDF value
mgen = m[r<y]

print(f"Length of the generated vector: {len(mgen)}")
print(f"Generated vector: {mgen}")
