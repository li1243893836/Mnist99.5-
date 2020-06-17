# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'layout.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        
        
        self.pbtClear = QtWidgets.QPushButton(MainWindow)
        self.pbtClear.setGeometry(QtCore.QRect(80, 440, 120, 30))
        self.pbtClear.setStyleSheet("")
        self.pbtClear.setCheckable(False)
        self.pbtClear.setChecked(False)
        self.pbtClear.setObjectName("pbtClear")
        
        self.pbtPredict = QtWidgets.QPushButton(MainWindow)
        self.pbtPredict.setGeometry(QtCore.QRect(80, 500, 120, 30))
        self.pbtPredict.setStyleSheet("")
        self.pbtPredict.setObjectName("pbtPredict")
        self.lbDataArea = QtWidgets.QLabel(MainWindow)
        self.lbDataArea.setGeometry(QtCore.QRect(540, 350, 224, 224))
        self.lbDataArea.setMouseTracking(False)
        self.lbDataArea.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.lbDataArea.setFrameShape(QtWidgets.QFrame.Box)
        self.lbDataArea.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.lbDataArea.setLineWidth(4)
        self.lbDataArea.setMidLineWidth(0)
        self.lbDataArea.setText("")
        self.lbDataArea.setObjectName("lbDataArea")
        self.label_3 = QtWidgets.QLabel(MainWindow)
        self.label_3.setGeometry(QtCore.QRect(260, 340, 91, 181))
        self.label_3.setObjectName("label_3")
        self.label_4 = QtWidgets.QLabel(MainWindow)
        self.label_4.setGeometry(QtCore.QRect(540, 320, 131, 20))
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(MainWindow)
        self.label_5.setGeometry(QtCore.QRect(20, 10, 711, 241))
        self.label_5.setObjectName("label_5")
        self.verticalLayoutWidget = QtWidgets.QWidget(MainWindow)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(540, 350, 221, 221))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.dArea_Layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.dArea_Layout.setContentsMargins(0, 0, 0, 0)
        self.dArea_Layout.setSpacing(0)
        self.dArea_Layout.setObjectName("dArea_Layout")
        self.lbResult = QtWidgets.QLabel(MainWindow)
        self.lbResult.setGeometry(QtCore.QRect(380, 350, 91, 131))
        font = QtGui.QFont()
        font.setPointSize(48)
        self.lbResult.setFont(font)
        self.lbResult.setObjectName("lbResult")
        self.lbCofidence = QtWidgets.QLabel(MainWindow)
        self.lbCofidence.setGeometry(QtCore.QRect(360, 500, 151, 21))
        font = QtGui.QFont()
        font.setPointSize(12)
        self.lbCofidence.setFont(font)
        self.lbCofidence.setObjectName("lbCofidence")

        self.retranslateUi(MainWindow)
        
        self.pbtClear.clicked.connect(MainWindow.pbtClear_Callback)
        self.pbtPredict.clicked.connect(MainWindow.pbtPredict_Callback)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "手写数字识别"))
        self.pbtClear.setText(_translate("MainWindow", "清除数据"))
        self.pbtPredict.setText(_translate("MainWindow", "识别"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:600;\">识别结果：</p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "数据输入区域"))
        self.lbResult.setText(_translate("MainWindow", "9"))
        self.lbCofidence.setText(_translate("MainWindow", "0.99999999"))
