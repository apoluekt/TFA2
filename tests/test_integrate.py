import sys, math
import tensorflow as tf

sys.path.append("../")

import amplitf.interface as atfi

atfi.backend_tf()

npoints = 100000000


@atfi.function
def func(x):
    return 1.0 / math.sqrt(math.pi) * atfi.exp(-(x ** 2))


@atfi.function
def integral(x):
    return atfi.reduce_sum(func(x)) / npoints * 10.0


atfi.set_seed(1)
x = atfi.random_uniform((npoints, 1), -5.0, 5.0)

strategy = tf.distribute.MirroredStrategy()

global_batch_size = x.shape[0]
tf_dataset = tf.data.Dataset.from_tensor_slices(x).batch(global_batch_size)
print(tf_dataset)
dist_dataset = strategy.experimental_distribute_dataset(tf_dataset)
print(dist_dataset)
dist_values = iter(dist_dataset).get_next()
print(dist_values)


@tf.function
def distributed_integral(x):
    integral_per_replica = strategy.experimental_run_v2(integral, args=(x,))
    print(integral_per_replica)
    return strategy.reduce("SUM", integral_per_replica, axis=None)


y = distributed_integral(x)

print(y)
