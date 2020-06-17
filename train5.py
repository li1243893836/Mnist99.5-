import tensorflow as tf
import numpy as np
#Tensorflow提供了一个类来处理MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
CKPT_DIR = 'ckpt'
#载入数据集
mnist=input_data.read_data_sets("F:/Mnist/data_set",one_hot=True)

#一批数量
batch_size=700
#mnist.train.images是一个55000 * 784维的矩阵, mnist.train.labels是一个55000 * 10维的矩阵
train_X, train_Y, test_X, test_Y = mnist.train.images, mnist.train.labels, mnist.test.images, mnist.test.labels
train_X = train_X.reshape(-1, 28, 28, 1)  # 将每张图片用一个28x28的矩阵表示,(55000,28,28,1)
test_X = test_X.reshape(-1, 28, 28, 1)  # 将每张图片用一个28x28的矩阵表示,(1000,28,28,1)
global_acc = 0.
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
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2))
h_pool2=max_pool_2x2(h_conv2)  #池化层
h_drop2=tf.nn.dropout(h_pool2,p_keep_conv) 
#初始化第三个卷积层的权值和偏置
W_conv3=weight_variable([3,3,64,128]) #3*3的采样窗口，128个卷积核从64个平面抽取特征
b_conv3=bias_variable([128])
#把第二个池化层结果和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv3=tf.nn.relu(conv2d(h_pool2,W_conv3))
h_pool3=max_pool_2x2(h_conv3)
h_drop3=tf.nn.dropout(h_pool3,p_keep_conv) 

#合并所有的feature map
h_feature = tf.reshape(h_drop3,[-1,w4.get_shape().as_list()[0]])
h_f = tf.nn.dropout(h_feature,p_keep_conv)
#全连接层
h_fc1 = tf.nn.relu(tf.matmul(h_f,w4))
h_fc2 = tf.nn.dropout(h_fc1,p_keep_hidden)
#h_fc3 = tf.nn.relu(tf.matmul(h_fc2,w1))
#h_fc4 = tf.nn.dropout(h_fc3,p_keep_hidden)
pyx = tf.matmul(h_fc2,w0)

 
#交叉熵代价函数
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pyx))
#使用AdamOptimizer进行优化
#train_step =tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)

predict_op = tf.argmax(pyx, 1)
 
#创建会话
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #初始化变量
    for epoch in range(41):
      
        training_batch = zip(range(0, len(train_X), batch_size),
                            range(batch_size, len(train_X)+1, batch_size))
        for start,end in training_batch:
            
            sess.run(train_step, feed_dict={x: train_X[start:end], y: train_Y[start:end],
                                        p_keep_conv: 0.8, p_keep_hidden: 0.5})
        test_batch = zip(range(0, len(test_X), 1),
                             range(1, len(test_X)+1, 1))
        
        #测试数据计算出准确率 
        test_indices = np.arange(len(test_X)) 
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:]

        acc = np.mean(np.argmax(test_Y[test_indices], axis=1) ==
                         sess.run(predict_op, feed_dict={x: test_X[test_indices],
                                                         y: test_Y[test_indices],
                                                         p_keep_conv: 1.0,
                                                         p_keep_hidden: 1.0}))
        if acc > global_acc:
            global_acc =  acc
            tf.train.Saver().save(sess,CKPT_DIR + "/model")
        
        print("\r","进度百分比:{0}%".format(round((epoch+1)*100/41)) + "当前轮次：" + str(epoch) + ",当前测试准确率：" + str(acc),flush=True)
       # print("当前轮次："+str(epoch)+",当前测试准确率："+str(acc))  
    global_acc = global_acc * 100
    print("测试准确率：%f%%"%(global_acc)) #输出运行时间
