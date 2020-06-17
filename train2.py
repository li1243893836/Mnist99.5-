import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#权值初始化
def weight_value(shape):
    init=tf.truncated_normal(shape,mean=0.0,stddev=0.1)
    return tf.Variable(init)
#偏置初始化
def biase_value(shape):
    init=tf.constant(0.1,shape=shape)
    return tf.Variable(init)
#卷积
def conv2d(Inputs,Weight):
    return tf.nn.conv2d(Inputs,Weight,strides=[1,1,1,1],padding='SAME')
#池化
def maxpooling(Inputs):
    return tf.nn.max_pool(Inputs,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
 
mnist=input_data.read_data_sets("MNIST_data",one_hot=True)
 
xs=tf.placeholder(tf.float32,[None,784])#读入图片
ys=tf.placeholder(tf.float32,[None,10])#读入标签
keep_prob=tf.placeholder(tf.float32)#dropout的失活率
max=tf.placeholder(tf.float32) #最大的准确度
test_suss=tf.placeholder(tf.float32)#每次测试的准确度 
Image=tf.reshape(xs,[-1,28,28,1])
#第一个卷积层的定义
conv1_W=weight_value([5,5,1,32])
conv1_b=biase_value([32])
conv1_h=tf.nn.relu(conv2d(Image,conv1_W)+conv1_b)
conv1_out=maxpooling(conv1_h)
#第二个卷积层的定义
conv2_W=weight_value([5,5,32,64])
conv2_b=biase_value([64])
conv2_h=tf.nn.relu(conv2d(conv1_out,conv2_W)+conv2_b)
conv2_out=maxpooling(conv2_h)
#全连接层的定义
fcnn_in=tf.reshape(conv2_out,[-1,49*64])
fcnn1_W=weight_value([49*64,49*64])
fcnn1_b=biase_value([49*64])
fcnn1_out=tf.nn.relu(tf.matmul(fcnn_in,fcnn1_W)+fcnn1_b)
fcnn1_dropout=tf.nn.dropout(fcnn1_out,keep_prob)
 
fcnn2_W=weight_value([49*64,10])
fcnn2_b=biase_value([10])
prediction=tf.nn.softmax(tf.matmul(fcnn1_dropout,fcnn2_W)+fcnn2_b)
#采用交叉熵做目标函数
cross_entropy=-tf.reduce_sum(ys*tf.log(prediction))
#精确度计算
num=tf.equal(tf.argmax(prediction,1),tf.argmax(ys,1))
accurate=tf.reduce_mean(tf.cast(num,tf.float32))
#训练模型
train_step=tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
 
sess=tf.Session()
sess.run(tf.initialize_all_variables())
 
for step in range(2500):
    batch_x,batch_y=mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_x,ys:batch_y,keep_prob:0.5})
    #print(test_suss)
    #比较大小
    #tf.less()
    if step % 50 == 0:
        test_suss=sess.run(accurate,feed_dict={xs:mnist.test.images,ys:mnist.test.labels,keep_prob:1.0})
        test_suss_float = tf.to_float(test_suss)
        print("迭代次数：%d   准确率：" %(step),end="")
        print(test_suss)
#max = max * 100
#print("准确度为：%f%%" %(max))
        
