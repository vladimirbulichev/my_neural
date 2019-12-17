import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

x = tf.constant([[1., 2., 0.]])
print(x)
negMatrix = tf.negative(x)
print(negMatrix)

with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True)) as sess:
    result = sess.run(negMatrix)

print(result)
