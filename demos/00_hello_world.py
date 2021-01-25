import tensorflow as tf

a = tf.Variable(1.0, trainable=True)
w = tf.Variable(1.0, trainable=True)
p = tf.Variable(0.0, trainable=True)


@tf.function
def f(x):
    return a * tf.sin(w * x + p)


x = tf.constant([0.0, 1.0, 2.0, 3.0, 4.0])
print(f(x))

y = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0])


@tf.function
def chi2():
    return tf.reduce_sum((f(x) - y) ** 2)


with tf.GradientTape() as gt:
    grad = gt.gradient(chi2(), [a, w, p])
print(grad)

from tensorflow.python.training import gradient_descent

opt = gradient_descent.GradientDescentOptimizer(0.001)

for _ in range(1000):
    print(a.numpy(), w.numpy(), p.numpy(), chi2().numpy())
    opt.minimize(chi2)
