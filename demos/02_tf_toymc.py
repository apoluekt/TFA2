import tensorflow as tf

npoints = 100000000  # Number of random points

def bw(m, m0, gamma) : 
  """
    Breit-Wigner distribution
  """
  ampl = tf.complex(m0*m0 - m*m, -m0*gamma)
  return tf.abs(ampl)**2

# Set initial random seed
tf.random.set_seed(1)

# Generate random sample (1D vector) with the uniformly distributed values
m = tf.random.uniform( (npoints, ), minval = 0., maxval = 1500. )

y = bw(m, tf.constant(770.), tf.constant(150.) )

ymax = tf.reduce_max(y)

r = tf.random.uniform( (npoints, ), minval = 0., maxval = ymax )

mgen = m[y>r]

print(mgen)
