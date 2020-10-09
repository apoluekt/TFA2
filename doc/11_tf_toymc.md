# Toy MC with pure TensorFlow

Example available here: https://github.com/apoluekt/TFA2/blob/master/demos/02_tf_toymc.py

Again, we start with the function that defines the PDF we will use, this time, for MC generation using [__rejection sampling__](https://en.wikipedia.org/wiki/Rejection_sampling): 

```python
def bw(m, m0, gamma) : 
  ampl = 1./tf.complex(m0*m0 - m*m, -m0*gamma)
  return tf.abs(ampl)**2
```

We are relying on the TF ability to do complex arithmetics. There is no `@tf.function` decorator in front of the definition of the function, meaning that TF will work in _eager mode_. This does not make any difference in this script, because the function is called only once. 

We then generate a random vector of mass values as we already did before
```python
m = tf.random.uniform( (npoints, ), minval = 0., maxval = 1500., dtype = tf.float64 )
```
and calculate the values of Breit-Wigner PDF at each point for fixed values of resonance mass and width: 
```python
y = bw(m, tf.constant(770., dtype = tf.float64), tf.constant(150., dtype = tf.float64) )
```
Note the arguments `dtype = tf.float64` whenever we create tensors. In this example, we are using __double precision__ to make calculations, which is usually a good idea in physics analyses. By default, TF always works with __single precision__ floating point arythmetics (`float32`). There is currently no way to set the default floating point precision, so this has to be done explicitly for every created tensor (we will fix this in the `AmpliTF` package). 

Moreover, TF operations cannot mix precision, e.g. you cannot add a tensor of double and a tensor of single precision. Thus, if mixed precision is really necessary, one has to explicitly convert the tensor with the dedicated operation `tf.cast(x)`
> __Exercise__: Try removing one of the `dtype = tf.float64` instances. 

TF supports the same syntax for [__fancy indexing__](https://numpy.org/doc/stable/user/basics.indexing.html) as `numpy`. This is used to filter only the entries in the generated mass vector, for which the random points fall under the PDF curve: 
```python
mgen = m[r<y]
```
Equivalently, this could be reached using the Boolean operations `tf.less()` and `tf.boolean_mask()`: 
```python
mgen = tf.boolean_mask(m, tf.less(r, y) )
```
