#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys, os
import numpy as np
import tensorflow as tf
from PIL import Image, ImageQt


from qt.layout import Ui_MainWindow
from qt.paintboard import PaintBoard

from PyQt5.QtWidgets import QMainWindow, QDesktopWidget, QApplication
from PyQt5.QtWidgets import QLabel, QMessageBox, QPushButton, QFrame
from PyQt5.QtGui import QPainter, QPen, QPixmap, QColor, QImage
from PyQt5.QtCore import Qt, QPoint, QSize










# 初始化网络
model_file = tf.train.latest_checkpoint("./ckpt0.54/")



class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self):
        super(MainWindow,self).__init__()
    
        # 初始化参数
        self.result = [0, 0]

        # 初始化UI
        self.setupUi(self)
        self.center()

        # 初始化画板
        self.paintBoard = PaintBoard(self, Size = QSize(224, 224), Fill = QColor(0,0,0,255))
        self.paintBoard.setPenColor(QColor(255,255,255,255))
        self.dArea_Layout.addWidget(self.paintBoard)

        self.clearDataArea()

    # 窗口居中
    def center(self):
        # 获得窗口
        framePos = self.frameGeometry()
        # 获得屏幕中心点
        scPos = QDesktopWidget().availableGeometry().center()
        # 显示到屏幕中心
        framePos.moveCenter(scPos)
        self.move(framePos.topLeft())
    
    
    # 窗口关闭事件
    def closeEvent(self, event):
        reply = QMessageBox.question(self, 'Message',
            "你确定要退出吗?", QMessageBox.Yes | 
            QMessageBox.No, QMessageBox.Yes)

        if reply == QMessageBox.Yes:
            event.accept()
        else:
            event.ignore()   
    
    # 清除数据待输入区
    def clearDataArea(self):
        self.paintBoard.Clear()
        self.lbDataArea.clear()
        self.lbResult.clear()
        self.lbCofidence.clear()
        self.result = [0, 0]

    
        
        #mnist.train.images是一个55000 * 784维的矩阵, mnist.train.labels是一个55000 * 10维的矩阵

    
       


    # 数据清除
    def pbtClear_Callback(self):
        self.clearDataArea()
        Gui = MainWindow()
        Gui.show()
 

    # 识别
    def pbtPredict_Callback(self):
        img, img_array =[],[]      # 将图像统一从qimage->pil image -> np.array [1, 1, 28, 28]
        
        # 获取qimage格式图像
        
        img = self.paintBoard.getContentAsQImage()

        # 转换成pil image类型处理
        pil_img = ImageQt.fromqimage(img)
        pil_img = pil_img.resize((28, 28), Image.ANTIALIAS)
        
        pil_img.save('test.png')

        pil_img = pil_img.convert('L')
        tv = list(pil_img.getdata())
        tva = [(255-x)*1.0/255.0 for x in tv]

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
        
        keep_prob=tf.placeholder(tf.float32) 
        #p_keep_conv = tf.placeholder(tf.float32) # 卷积层的dropout概率
        #p_keep_hidden = tf.placeholder(tf.float32)# 全连接层的dropout概率
        #输入层
        #定义两个placeholder
        x=tf.placeholder(tf.float32,[None,784]) #28*28
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
        h_drop1=tf.nn.dropout(h_pool1,keep_prob) 
        #初始化第二个卷积层的权值和偏置
        W_conv2=weight_variable([3,3,32,64]) #3*3的采样窗口，64个卷积核从32个平面抽取特征
        b_conv2=bias_variable([64])
        #把第一个池化层结果和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
        h_conv2=tf.nn.relu(conv2d(h_pool1,W_conv2))
        h_pool2=max_pool_2x2(h_conv2)  #池化层
        h_drop2=tf.nn.dropout(h_pool2,keep_prob) 
        #初始化第三个卷积层的权值和偏置
        W_conv3=weight_variable([3,3,64,128]) #3*3的采样窗口，128个卷积核从64个平面抽取特征
        b_conv3=bias_variable([128])
        #把第二个池化层结果和权值向量进行卷积，再加上偏置值，然后应用于relu激活函数
        h_conv3=tf.nn.relu(conv2d(h_pool2,W_conv3))
        h_pool3=max_pool_2x2(h_conv3)
        h_drop3=tf.nn.dropout(h_pool3,keep_prob) 

        #合并所有的feature map
        h_feature = tf.reshape(h_drop3,[-1,w4.get_shape().as_list()[0]])
        h_f = tf.nn.dropout(h_feature,keep_prob)
        #全连接层
        h_fc1 = tf.nn.relu(tf.matmul(h_f,w4))
        h_fc2 = tf.nn.dropout(h_fc1,keep_prob)
        pyx = tf.matmul(h_fc2,w0)

        #交叉熵代价函数

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=pyx))
        #使用AdamOptimizer进行优化

        train_step = tf.train.RMSPropOptimizer(0.001, 0.9).minimize(cross_entropy)
        #结果放在bool集中
        correct_prediction=tf.equal(tf.argmax(pyx,1),tf.argmax(y,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        saver = tf.train.Saver() 
        #创建会话
        with tf.Session() as sess:
            
            sess.run(tf.global_variables_initializer()) #初始化变量
            saver.restore(sess,model_file)
            #测试
            prediction = tf.argmax(pyx,1)
            predint=prediction.eval(feed_dict={x: [tva],keep_prob: 1.0}, session=sess)
            print(predint[0])
            #acc=sess.run(accuracy,feed_dict={x:[tva],keep_prob:1.0})
        
        # img_array = np.where(img_array>0.5, 1, 0)
        # reshape成网络输入类型 
             # shape:[1, 10]
        # print (__result)

        # 将预测结果使用softmax输出
       
       
        self.result[0] = predint[0]          # 预测的数字
        #self.result[1] = __result[0, self.result[0]]     # 置信度

        self.lbResult.setText("%d" % (self.result[0]))
        #self.lbCofidence.setText("%.8f" % (self.result[1]))





if __name__ == "__main__":
    app = QApplication(sys.argv)
    Gui = MainWindow()
    Gui.show()

    sys.exit(app.exec_())