# Toy MC with pure TensorFlow

Example available here: https://github.com/apoluekt/TFA2/blob/master/demos/02_tf_toymc.py

Again, we start by defining the function that defined the PDF we will use, this time, for MC generation using [rejection sampling](https://en.wikipedia.org/wiki/Rejection_sampling): 

```python
def bw(m, m0, gamma) : 
  ampl = tf.complex(m0*m0 - m*m, -m0*gamma)
  return tf.abs(ampl)**2
```

We are relying on the TF ability to do complex arithmetics. There is no `@tf.function` decorator in front of the definition of the function, meaning that TF will work in _eager mode_. This does not make any difference in this case, because the function is called only once. 
