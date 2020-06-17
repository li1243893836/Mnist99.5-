"""
编写多层卷积神经网络来识别mnist数据集
采用了Batch Normalization对输入数据进行处理
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data_set/", one_hot=True)


def batch_norm(wx_plus_b, n_output):
    """
    进行BN的函数
    :param wx_plus_b: 经历激活函数之前的数据
    :param n_output: 输出的神经元个数
    :return: 经过BN之后的数据，传递给激活函数
    """
    fc_mean, fc_var = tf.nn.moments(
        wx_plus_b,
        axes=[0],
    )
    scale = tf.Variable(tf.random_normal([n_output]))
    shift = tf.Variable(tf.random_normal([n_output]))
    epsilon = 0.001
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([fc_mean, fc_var])
        with tf.control_dependencies([ema_apply_op]):
            return tf.identity(fc_mean), tf.identity(fc_var)
    mean, var = mean_var_with_update()
    wx_plus_b_out = tf.nn.batch_normalization(wx_plus_b, mean, var, shift, scale, epsilon)
    return wx_plus_b_out


x_data = tf.placeholder(tf.float32, [None, 784])
y_data = tf.placeholder(tf.float32, [None, 10])
x_data_ = batch_norm(x_data, 784)  # 对输入数据也进行了BN
x_image = tf.reshape(x_data_, [-1, 28, 28, 1])  # 将数据转变为卷积需要的格式


# 定义权重函数
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


# 定义偏置项
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


# 定义卷积核
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding="SAME")


# 定义池化矩阵
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")


# 设置第一层的初始权重和偏置
w_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
# 第一层的卷积输出
h_conv1 = tf.nn.relu(tf.add(conv2d(x_image, w_conv1), b_conv1))
# 第一层的池化输出
h_pool1 = max_pool_2x2(h_conv1)

# 设置第二层的初始权重和偏置
w_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
# 第二层的卷积输出，注意使用Batch Normalization
h_conv2 = tf.nn.relu(batch_norm(tf.add(conv2d(h_pool1, w_conv2), b_conv2), 64))
# 第二层的池化输出
h_pool2 = max_pool_2x2(h_conv2)

# 设置全连接层1的初始权重和偏置
w_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])

# 将第二层的池化输出压缩为一维向量，用于全连接层1的输入，输入时注意BN
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(batch_norm(tf.add(tf.matmul(h_pool2_flat, w_fc1), b_fc1), 1024))

# 设置dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# 设置全连接层2的初始权重和偏置
w_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
# 全连接层2的输出，注意BN
y_model = tf.nn.softmax(batch_norm(tf.add(tf.matmul(h_fc1_drop, w_fc2), b_fc2), 10))

# 损失函数
loss = -tf.reduce_sum(y_data * tf.log(y_model))
# 训练
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
# 准确率
correct_prediction = tf.equal(tf.argmax(y_model, 1), tf.argmax(y_data, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

# 启动训练
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch = mnist.train.next_batch(50)
    sess.run(train_step, feed_dict={x_data: batch[0], y_data: batch[1], keep_prob: 0.5})

    if i % 5 == 0:
        train_acc = sess.run(accuracy, feed_dict={x_data: batch[0], y_data: batch[1], keep_prob: 1.0})
        test_acc = sess.run(accuracy, feed_dict={x_data: mnist.test.images,
                                                 y_data: mnist.test.labels,
                                                 keep_prob: 1.0})
        print("step %d, \ntraining accuracy %g, testing accuracy %g" % (i, train_acc, test_acc))

sess.close()
