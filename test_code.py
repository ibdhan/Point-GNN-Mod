import os
from zlib import Z_BEST_COMPRESSION
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# from tensorflow.python.client import device_lib
# print(device_lib.list_local_devices())

import tensorflow as tf
import numpy as np

x = tf.Variable(np.array([[1, 1, 1],[2, 1, 3], [4, 3, 1]]), dtype = tf.float32)
xs = tf.Variable(np.array([[2, 2, 2],[2, 2, 5], [4, 1, 3]]), dtype = tf.float32)

#x and y are 2 dims
def euclideanDistance_ori(x, y):
    dist = tf.sqrt(tf.reduce_sum(tf.square(x - y), 1, keepdims = True))
    return dist

def euclideanDistance(x, y):
    x = tf.Print(x, [x], "\nx:", summarize=40)
    y = tf.Print(y, [y], "\ny:", summarize=40)
    z = x - y
    # z = tf.maximum(z, 1e-9) #Prevent NaN
    z = tf.Print(z, [z], "\nx-y:", summarize=40)

    z = tf.square(z)
    z = tf.Print(z, [z], "\nsquare[z]:", summarize=40)
    
    z = tf.reduce_sum(z, 1, keepdims = True)
    z = tf.Print(z, [z], "\nreduce_sum[z]:", summarize=40)

    z = tf.sqrt(z)
    z = tf.Print(z, [z], "\nsqrt[z]:", summarize=40)
    return z

d = euclideanDistance(x, xs)
init = tf.global_variables_initializer() 
init_local = tf.local_variables_initializer()
with tf.Session() as sess:
    sess.run([init, init_local])
    print(sess.run(d))