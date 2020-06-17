import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist=input_data.read_data_sets("./data_set",one_hot=True)
saver  = tf.train.Saver()
with tf.Session() as sess:
    model_file = tf.train.latest_checkpoint("./ckpt/")
    saver.restore(sess,model_file)



