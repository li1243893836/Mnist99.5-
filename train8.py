import tensorflow as tf
import numpy as np
#Tensorflow提供了一个类来处理MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
CKPT_DIR = 'ckpt'
#载入数据集
mnist=input_data.read_data_sets("F:/Mnist/data_set",one_hot=True)
#设置批次的大小
batch_size=256
#计算一共有多少个批次
n_batch=mnist.train.num_examples//batch_size
#mnist.train.images是一个55000 * 784维的矩阵, mnist.train.labels是一个55000 * 10维的矩阵
trX, trY, teX, teY = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
trX = trX.reshape(-1, 28, 28, 1)  # 将每张图片用一个28x28的矩阵表示,(55000,28,28,1)
teX = teX.reshape(-1, 28, 28, 1)  # 将每张图片用一个28x28的矩阵表示,(1000,28,28,1)
global_acc = 0.
save_interval = 5 #每5轮保存模型 
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

#定义初始化权值函数
def weight_variable(shape):
    initial=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(initial)
#定义初始化偏置函数
def bias_variable(shape):
    initial=tf.constant(0.1,shape=shape)
    return tf.Variable(initial)
#卷积层
def conv2d(input,filter):
    return tf.nn.conv2d(input,filter,strides=[1,1,1,1],padding='SAME')
#池化层
def max_pool_2x2(value):
    return tf.nn.max_pool(value,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
 
#keep_prob=tf.placeholder(tf.float32) 
p_keep_conv = tf.placeholder(tf.float32) # 卷积层的dropout概率
p_keep_hidden = tf.placeholder(tf.float32)# 全连接层的dropout概率
#输入层
#定义两个placeholder
x=tf.placeholder(tf.float32,[None, 28, 28, 1]) #28*28
y=tf.placeholder(tf.float32,[None,10])
#改变x的格式转为4维的向量[batch,in_hight,in_width,in_channels]
x_image=tf.reshape(x,[-1,28,28,1])
w0 = weight_variable([625,10]) #FC 128 inputs,10 outputs
w4 = weight_variable([128*4*4,625]) # FC 128 * 4 * 4 inputs, 625 outputs
w1 = weight_variable([625,128]) # FC 625 inputs, 128 outputs (labels)
#卷积、激励、池化操作
#初始化第一个卷积层的权值和偏置
W_conv1=weight_variable([3,3,1,32]) #3*3的采样窗口，32个卷积核从1个平面抽取特征
b_conv1=bias_variable([32])
#把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1))
h_pool1=max_pool_2x2(h_conv1)  #进行max_pooling 池化层
h_drop1=tf.nn.dropout(h_pool1,p_keep_conv) 
#初始化第二个卷积层的权值和偏置
W_conv2=weight_variable([3,3,32,64]) #3*3的采样窗口，64个卷积核从32个平面抽取特征
b_conv2=bias_variable([64])
#把第一个池化层结果和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv2=tf.nn.relu(batch_norm((conv2d(h_pool1,W_conv2)),64))
h_pool2=max_pool_2x2(h_conv2)  #池化层
h_drop2=tf.nn.dropout(h_pool2,p_keep_conv) 
#初始化第三个卷积层的权值和偏置
W_conv3=weight_variable([3,3,64,128]) #3*3的采样窗口，128个卷积核从64个平面抽取特征
b_conv3=bias_variable([128])
#把第二个池化层结果和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv3=tf.nn.relu(batch_norm(conv2d(h_pool2,W_conv3),128))
h_pool3=max_pool_2x2(h_conv3)
h_drop3=tf.nn.dropout(h_pool3,p_keep_conv) 

#合并所有的feature map
h_feature = tf.reshape(h_drop3,[-1,w4.get_shape().as_list()[0]])
h_f = tf.nn.dropout(h_feature,p_keep_conv)
#全连接层
h_fc1 = tf.nn.relu(batch_norm(tf.matmul(h_f,w4),625))
h_fc2 = tf.nn.dropout(h_fc1,p_keep_hidden)
#h_fc3 = tf.nn.relu(tf.matmul(h_fc2,w1))
#h_fc4 = tf.nn.dropout(h_fc3,p_keep_hidden)
pyx = tf.matmul(h_fc2,w0)
#全连接层
#初始化第一个全连接层的权值
#W_fc1=weight_variable([3*3*128,2048])#经过池化层后有7*7*64个神经元，全连接层有1024个神经元
#b_fc1 = bias_variable([2048])#1024个节点
#把池化层2的输出扁平化为1维
#h_pool2_flat = tf.reshape(h_pool2,[-1,3*3*128])
#求第一个全连接层的输出
#h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
 
#keep_prob用来表示神经元的输出概率

#h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
 
#初始化第二个全连接层
#W_fc2=weight_variable([2048,10])
#b_fc2=bias_variable([10])
 
#输出层
#计算输出
#y = softmax(x*w+b)
#prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
 
#交叉熵代价函数
#cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pyx))
#使用AdamOptimizer进行优化
#train_step =tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
#结果存放在一个布尔列表中(argmax函数返回一维张量中最大的值所在的位置)
#correct_prediction=tf.equal(tf.argmax(pyx,1),tf.argmax(y,1))
#求准确率(tf.cast将布尔值转换为float型)
#predict_op = tf.argmax(pyx, 1)
#accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
predict_op = tf.argmax(pyx, 1)
 
#创建会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #初始化变量
    for epoch in range(41):
        training_batch = zip(range(0, len(trX), batch_size),
                             range(batch_size, len(trX)+1, batch_size))
        for start, end in training_batch:
            sess.run(train_step, feed_dict={x: trX[start:end], y: trY[start:end],
                                        p_keep_conv: 0.8, p_keep_hidden: 0.5})
        test_batch = zip(range(0, len(teX), 1),
                             range(1, len(teX)+1, 1))
        
        #测试
        test_indices = np.arange(len(teX)) 
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:]

        acc = np.mean(np.argmax(teY[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={x: teX[test_indices],
                                                         y: teY[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0}))
        if acc > global_acc:
            global_acc =  acc
            tf.train.Saver().save(sess,CKPT_DIR + "/model")
        #print(type(acc))                             
    #for epoch in range(31):  
       # for batch in range(n_batch):
           # batch_xs,batch_ys=mnist.train.next_batch(batch_size)
           # sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7}) #进行迭代训练
        #测试数据计算出准确率
        #acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0}       
        print("\r","进度百分比:{0}%".format(round((epoch+1)*100/41)) + "当前轮次：" + str(epoch) + ",当前测试准确率：" + str(acc),flush=True)
       # print("当前轮次："+str(epoch)+",当前测试准确率："+str(acc))  
    global_acc = global_acc * 100
    print("测试准确率：%f%%"%(global_acc)) #输出运行时间
