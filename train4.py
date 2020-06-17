import tensorflow as tf
#Tensorflow提供了一个类来处理MNIST数据
from tensorflow.examples.tutorials.mnist import input_data
import time
 
#载入数据集
mnist=input_data.read_data_sets("F:/Mnist/data_set",one_hot=True)
#设置批次的大小
batch_size=100
#计算一共有多少个批次
n_batch=mnist.train.num_examples//batch_size
 
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
 
 
#输入层
#定义两个placeholder
x=tf.placeholder(tf.float32,[None,784]) #28*28
y=tf.placeholder(tf.float32,[None,10])
#改变x的格式转为4维的向量[batch,in_hight,in_width,in_channels]
x_image=tf.reshape(x,[-1,28,28,1])
 
 
#卷积、激励、池化操作
#初始化第一个卷积层的权值和偏置
W_conv1=weight_variable([5,5,1,32]) #5*5的采样窗口，32个卷积核从1个平面抽取特征
b_conv1=bias_variable([32]) #每一个卷积核一个偏置值
#把x_image和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv1=tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1=max_pool_2x2(h_conv1)  #进行max_pooling 池化层
 
#初始化第二个卷积层的权值和偏置
W_conv2=weight_variable([5,5,32,64]) #5*5的采样窗口，64个卷积核从32个平面抽取特征
b_conv2=bias_variable([64])
#把第一个池化层结果和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2=max_pool_2x2(h_conv2)  #池化层
  
#初始化第三个卷积层的权值和偏置
W_conv3=weight_variable([5,5,64,128]) #5*5的采样窗口，128个卷积核从64个平面抽取特征
b_conv3=bias_variable([128])
#把第二个池化层结果和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
h_conv3=tf.nn.relu(conv2d(h_pool2,W_conv3)+b_conv3)

 
 
#全连接层
#初始化第一个全连接层的权值
W_fc1=weight_variable([7*7*128,1024])#经过池化层后有7*7*128个神经元，全连接层有1024个神经元
b_fc1 = bias_variable([1024])#1024个节点
#把卷积层3的输出扁平化为1维
h_pool2_flat = tf.reshape(h_conv3,[-1,7*7*128])
#求第一个全连接层的输出
h_fc1=tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
 
#keep_prob用来表示神经元的输出概率
keep_prob=tf.placeholder(tf.float32)
h_fc1_drop=tf.nn.dropout(h_fc1,keep_prob)
 
#初始化第二个全连接层
W_fc2=weight_variable([1024,10])
b_fc2=bias_variable([10])
 
#输出层
#计算输出
#y = softmax(x*w+b)
prediction=tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
 
#交叉熵代价函数
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))
#使用AdamOptimizer进行优化
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#结果存放在一个布尔列表中(argmax函数返回一维张量中最大的值所在的位置)
correct_prediction=tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
#求准确率(tf.cast将布尔值转换为float型)
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
 
#创建会话
with tf.Session() as sess:
    start_time=time.clock()
    sess.run(tf.global_variables_initializer()) #初始化变量
    for epoch in range(26):  
        for batch in range(n_batch):
            batch_xs,batch_ys=mnist.train.next_batch(batch_size)
            sess.run(train_step,feed_dict={x:batch_xs,y:batch_ys,keep_prob:0.7}) #进行迭代训练
        #测试数据计算出准确率
        acc=sess.run(accuracy,feed_dict={x:mnist.test.images,y:mnist.test.labels,keep_prob:1.0})
        max = 0.
        if(max < float(acc)):
            max = float(acc)
        print("\r","进度百分比:{0}%".format(round((epoch+1)*100/26)) + "当前轮次：" + str(epoch) + ",当前测试准确率：" + str(acc),end="",flush=True)
       # print("当前轮次："+str(epoch)+",当前测试准确率："+str(acc))
    end_time=time.clock()
    max = max * 100
    print("系统运行时间:%s 秒,测试准确率：%f%%"%(end_time-start_time,max)) #输出运行时间