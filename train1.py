import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# 读入数据  ‘MNIST_data’ 是我保存数据的文件夹的名称
mnist = input_data.read_data_sets('data_set', one_hot=True)

# 各种图片数据以及标签 images是图像数据  labels 是正确的结果
trainimg = mnist.train.images
trainlabels = mnist.train.labels
testimg = mnist.test.images
testlabels = mnist.test.labels

# 输入的数据 每张图片的大小是 28 * 28，在提供的数据集中已经被展平乘了 1 * 784（28 * 28）的向量
# 方便矩阵乘法处理
x = tf.placeholder(tf.float32, [None, 784])
# 输出的结果是对于每一张图输出的是 1*10 的向量，例如 [1, 0, 0, 0...]
# 只有一个数字是1 所在的索引表示预测数据
y = tf.placeholder(tf.float32, [None, 10])

# 模型参数
# 对于这样的全连接方式 某一层的参数矩阵的行数是输入数据的数量 ，列数是这一层的神经元个数
# 这一点用线性代数的思想考虑会比较好理解
W = tf.Variable(tf.zeros([784, 10]))
# 偏置
b = tf.Variable(tf.zeros([10]))

# 建立模型 并使用softmax（）函数对输出的数据进行处理
# softmax（） 函数比较重要 后面写
# 这里注意理解一下 模型输出的actv的shape 后边会有用（n * 10, n时输入的数据的数量）
actv = tf.nn.softmax(tf.matmul(x, W) + b)

# 损失函数 使用交叉熵的方式  softmax（）函数与交叉熵一般都会结合使用
# clip_by_value()函数可以将数组整理在一个范围内，后面会具体解释
cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(tf.clip_by_value(actv, 1e-10, 1.0)), reduction_indices=1))

# 使用梯度下降的方法进行参数优化
learning_rate = 0.01
optm = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# 判断是否预测结果与正确结果是否一致
# 注意这里使用的函数的 argmax（）也就是比较的是索引 索引才体现了预测的是哪个数字
# 并且 softmax（）函数的输出不是[1, 0, 0...] 类似的数组 不会与正确的label相同
# pred 数组的输出是  [True, False, True...] 类似的
pred = tf.equal(tf.argmax(actv, 1), tf.argmax(y, 1))

# 计算正确率
# 上面看到pred数组的形式 使用cast转化为浮点数 则 True会被转化为 1.0, False 0.0
# 所以对这些数据求均值 就是正确率了（这个均值表示所有数据中有多少个1 -> True的数量 ->正确个数）
accr = tf.reduce_mean(tf.cast(pred, tf.float32))

init_op = tf.global_variables_initializer()

# 接下来要使用的一些常量 可能会自己根据情况调整所以都定义在这里
training_epochs = 50  # 一共要训练的轮数
batch_size = 10  # 每一批训练数据的数量

with tf.Session() as sess:
    sess.run(init_op)
    # 对于每一轮训练
    for epoch in range(training_epochs):
        avg_cost = 0.
        max = 0.
        # 计算训练数据可以划分多少个batch大小的组
        num_batch = int(mnist.train.num_examples / batch_size)

        # 每一组每一组地训练
        for i in range(num_batch):
            # 这里地 mnist.train.next_batch()作用是：
            # 第一次取1-10数据 第二次取 11-20 ... 类似这样
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # 运行模型进行训练
            sess.run(optm, feed_dict={x: batch_xs, y: batch_ys})
            # 如果觉得上面 feed_dict 的不方便 也可以提前写在外边
            feeds = {x: batch_xs, y: batch_ys}
            # 累计计算总的损失值
            avg_cost += sess.run(cost, feed_dict=feeds) / num_batch
           
			
            
        # 输出一些数据
        # 为了输出在训练集上的正确率本来应该使用全部的train数据 这里为了快一点就只用了部分数据
        feed_train = {x: trainimg[1: 100], y: trainlabels[1: 100]}
            # 在测试集上运行模型
        feedt_test = {x: mnist.test.images, y: mnist.test.labels}
        train_acc = sess.run(accr, feed_dict=feed_train)
        test_acc = sess.run(accr, feed_dict=feedt_test)
        #计算出最大的准确率
        if max < test_acc:
            max = test_acc
        print("轮: %d 损失值: %.9f 训练准确率: %.9f 测试准确率: %.9f" %
                (epoch, avg_cost, train_acc, test_acc))
max = max * 100
print("准确率为：%f%%" %(max))
