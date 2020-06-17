import tensorflow as tf

a = tf.placeholder(tf.float32,3.0)
b = tf.placeholder(tf.float32,4.0)
b = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y_,labels=y)
a = tf.cond(a<b,lambda:b, lambda:a)
print(a)