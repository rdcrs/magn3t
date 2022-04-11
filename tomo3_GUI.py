from PyQt5.QtGui import QImage, QPixmap, QPalette, QPainter
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMainWindow, QMenu, QAction, \
	qApp, QFileDialog
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5 import QtGui,QtCore,QtWidgets
#~ from qrangeslider import QRangeSlider
from PyQt5.QtCore import QObject, QThread, pyqtSignal
import math
from PIL import Image
from collections import deque,OrderedDict
#from cv2 import imread,imwrite
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas 
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar 
import matplotlib.pyplot as plt 
from src import tomo
import numpy as np
import random 
import matplotlib as mpl
import vtk
from vtk.util import numpy_support
import vtk.qt
vtk.qt.QVTKRWIBase = "QGLWidget"
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from multiprocessing import Process
import os
import time
import os, psutil

import multiprocessing as mp


class RangeSlider(QSlider):
	valueChanged=pyqtSignal()
	
	def __init__(self, parent=None):
		super().__init__(parent)
		
		
		self.dis=0
		self.first_position = 0
		self.second_position = 999

		self.opt = QStyleOptionSlider()
		self.opt.minimum = 0
		self.opt.maximum = 999

		self.setTickPosition(QSlider.TicksAbove)
		self.setTickInterval(100)

		self.setSizePolicy(
			QSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed, QSizePolicy.Slider)
		)

	def setRangeLimit(self, minimum: int, maximum: int):
		self.opt.minimum = minimum
		self.opt.maximum = maximum

	def setRange(self, start: int, end: int):
		self.first_position = start
		self.second_position = end

	def getRange(self):
		return (self.first_position, self.second_position)

	def setTickPosition(self, position: QSlider.TickPosition):
		self.opt.tickPosition = position

	def setTickInterval(self, ti: int):
		self.opt.tickInterval = ti

	def paintEvent(self, event: QPaintEvent):

		painter = QPainter(self)

		# Draw rule
		self.opt.initFrom(self)
		self.opt.rect = self.rect()
		self.opt.sliderPosition = 0
		self.opt.subControls = QStyle.SC_SliderGroove | QStyle.SC_SliderTickmarks

		#   Draw GROOVE
		self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)

		#  Draw INTERVAL

		color = self.palette().color(QPalette.Highlight)
		color.setAlpha(160)
		painter.setBrush(QBrush(color))
		painter.setPen(Qt.NoPen)

		self.opt.sliderPosition = self.first_position
		x_left_handle = (
			self.style()
			.subControlRect(QStyle.CC_Slider, self.opt, QStyle.SC_SliderHandle)
			.right()
		)

		self.opt.sliderPosition = self.second_position
		x_right_handle = (
			self.style()
			.subControlRect(QStyle.CC_Slider, self.opt, QStyle.SC_SliderHandle)
			.left()
		)

		groove_rect = self.style().subControlRect(
			QStyle.CC_Slider, self.opt, QStyle.SC_SliderGroove
		)

		selection = QRect(
			x_left_handle,
			groove_rect.y(),
			x_right_handle - x_left_handle,
			groove_rect.height(),
		).adjusted(-1, 1, 1, -1)

		painter.drawRect(selection)

		# Draw first handle

		self.opt.subControls = QStyle.SC_SliderHandle
		self.opt.sliderPosition = self.first_position
		self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)

		# Draw second handle
		self.opt.sliderPosition = self.second_position
		self.style().drawComplexControl(QStyle.CC_Slider, self.opt, painter)

	def mousePressEvent(self, event: QMouseEvent):

		self.opt.sliderPosition = self.first_position
		self._first_sc = self.style().hitTestComplexControl(
			QStyle.CC_Slider, self.opt, event.pos(), self
		)
		
		self.opt.sliderPosition = self.second_position
		self._second_sc = self.style().hitTestComplexControl(
			QStyle.CC_Slider, self.opt, event.pos(), self
		)
		
		#~ print(self.style().hitTestComplexControl(
			#~ QStyle.CC_Slider, self.opt, event.pos(), self
		#~ ))
		distance = self.opt.maximum - self.opt.minimum
		pos = self.style().sliderValueFromPosition(
			0, distance, event.pos().x(), self.rect().width()
		)
		
		self.dis=pos-self.first_position

	def mouseMoveEvent(self, event: QMouseEvent):

		distance = self.opt.maximum - self.opt.minimum

		pos = self.style().sliderValueFromPosition(
			0, distance, event.pos().x(), self.rect().width()
		)

		dis=self.second_position-self.first_position
		
		if self._first_sc == QStyle.SC_SliderHandle:
			if pos <= self.second_position:
				self.first_position = pos
				self.update()
				self.valueChanged.emit()
				
		
		if self._second_sc == QStyle.SC_SliderHandle:
			if pos >= self.first_position:
				self.second_position = pos
				self.update()
				self.valueChanged.emit()
				
		if self._second_sc==1 and self._first_sc==1:
			if pos >= self.first_position and pos <= self.second_position:
				print(123)
				self.first_position=max(pos-self.dis,self.opt.minimum)
				self.second_position=min(pos-self.dis+dis,self.opt.maximum)
				self.update()
				self.valueChanged.emit()
				self.dis=pos-self.first_position
				
				
				
	def sizeHint(self):
		""" override """
		SliderLength = 84
		TickSpace = 5

		w = SliderLength
		h = self.style().pixelMetric(QStyle.PM_SliderThickness, self.opt, self)

		if (
			self.opt.tickPosition & QSlider.TicksAbove
			or self.opt.tickPosition & QSlider.TicksBelow
		):
			h += TickSpace

		return (
			self.style()
			.sizeFromContents(QStyle.CT_Slider, self.opt, QSize(w, h), self)
			.expandedTo(QApplication.globalStrut())
		)

class CustomDialog(QDialog):
	def __init__(self,values):
		self.values=values
		super(CustomDialog, self).__init__()
		self.setWindowTitle("Auto")
		self.layout=QGridLayout()
		
		
		text1=QtWidgets.QLabel()
		text1.setText('Threshold value')		
		self.value1=QtWidgets.QLineEdit()
		self.value1.setText(str(values['th']))
		self.layout.addWidget(text1,0,0)
		self.layout.addWidget(self.value1,0,1)
		
		text2=QtWidgets.QLabel()
		text2.setText('Erode')		
		self.value2=QtWidgets.QLineEdit()
		self.value2.setText(str(values['erode']))
		self.layout.addWidget(text2,1,0)
		self.layout.addWidget(self.value2,1,1)
		
		text3=QtWidgets.QLabel()
		text3.setText('Dilate')		
		self.value3=QtWidgets.QLineEdit()
		self.value3.setText(str(values['dilate']))
		self.layout.addWidget(text3,2,0)
		self.layout.addWidget(self.value3,2,1)
		
		text4=QtWidgets.QLabel()
		text4.setText('Peak height')		
		self.value4=QtWidgets.QLineEdit()
		self.value4.setText(str(values['peak']))
		self.layout.addWidget(text4,3,0)
		self.layout.addWidget(self.value4,3,1)
		
		#~ text5=QtWidgets.QLabel()
		#~ text5.setText('Font Size')		
		#~ value5=QtWidgets.QLineEdit()
		#~ value5.setText(str(values['fontSize']))
		#~ self.layout.addWidget(text5,4,0)
		#~ self.layout.addWidget(value5,4,1)
		QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
		self.buttonBox = QDialogButtonBox(QBtn)	
		self.buttonBox.accepted.connect(self.accept)
		self.buttonBox.rejected.connect(self.reject)
		self.layout.addWidget(self.buttonBox,4,0,4,2)
		self.setLayout(self.layout)
	#~ def accept(self):
		#~ pass
		#~ #self.accept()
	#~ def reject(self):
		#~ #self.reject()
		#~ pass
	def getResults(self):
		if self.exec_() == QDialog.Accepted:
			self.values['th']=float(self.value1.text())
			self.values['erode']=int(self.value2.text())
			self.values['dilate']=int(self.value3.text())
			self.values['peak']=float(self.value4.text())
			return self.values
		else:
			return None

class CustomDialog1(QDialog):
	def __init__(self,lista):
		self.lista=lista
		super(CustomDialog, self).__init__()
		self.setWindowTitle("Select filled maxima")
		self.layout=QGridLayout()
		
		
		text1=QtWidgets.QLabel()
		text1.setText('Threshold value')		
		self.value1=QtWidgets.QLineEdit()
		self.value1.setText(str(values['th']))
		self.layout.addWidget(text1,0,0)
		self.layout.addWidget(self.value1,0,1)
		
		text2=QtWidgets.QLabel()
		text2.setText('Erode')		
		self.value2=QtWidgets.QLineEdit()
		self.value2.setText(str(values['erode']))
		self.layout.addWidget(text2,1,0)
		self.layout.addWidget(self.value2,1,1)
		
		text3=QtWidgets.QLabel()
		text3.setText('Dilate')		
		self.value3=QtWidgets.QLineEdit()
		self.value3.setText(str(values['dilate']))
		self.layout.addWidget(text3,2,0)
		self.layout.addWidget(self.value3,2,1)
		
		text4=QtWidgets.QLabel()
		text4.setText('Peak height')		
		self.value4=QtWidgets.QLineEdit()
		self.value4.setText(str(values['peak']))
		self.layout.addWidget(text4,3,0)
		self.layout.addWidget(self.value4,3,1)
		
		#~ text5=QtWidgets.QLabel()
		#~ text5.setText('Font Size')		
		#~ value5=QtWidgets.QLineEdit()
		#~ value5.setText(str(values['fontSize']))
		#~ self.layout.addWidget(text5,4,0)
		#~ self.layout.addWidget(value5,4,1)
		QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
		self.buttonBox = QDialogButtonBox(QBtn)	
		self.buttonBox.accepted.connect(self.accept)
		self.buttonBox.rejected.connect(self.reject)
		self.layout.addWidget(self.buttonBox,4,0,4,2)
		self.setLayout(self.layout)
	#~ def accept(self):
		#~ pass
		#~ #self.accept()
	#~ def reject(self):
		#~ #self.reject()
		#~ pass
	def getResults(self):
		if self.exec_() == QDialog.Accepted:
			self.values['th']=float(self.value1.text())
			self.values['erode']=int(self.value2.text())
			self.values['dilate']=int(self.value3.text())
			self.values['peak']=float(self.value4.text())
			return self.values
		else:
			return None
			
class CustomDialog2(QDialog):
	def __init__(self,parent=None):
		super(CustomDialog2, self).__init__(parent)
		self.setWindowTitle("Select filled maxima")
		self.layout=QVBoxLayout()
		self.parent=parent
		
		self.listW=QtWidgets.QListWidget()
		self.listW.setSelectionMode(1)#4 
		#self.listW.itemSelectionChanged.connect(self.selectionChanged)
		self.updateLista()
		
		#inputLay=QHBoxLayout()
		#text1=QtWidgets.QLabel()
		#text1.setText('Valoare')		
		#self.value1=QtWidgets.QLineEdit()
		#self.value1.setText(str(1))
		#inputLay.addWidget(text1)
		#inputLay.addWidget(self.value1)
		
		self.layout.addWidget(self.listW)
		#self.layout.addLayout(inputLay)
		
		
		QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
		self.buttonBox = QDialogButtonBox(QBtn)	
		self.buttonBox.accepted.connect(self.accept)
		self.buttonBox.rejected.connect(self.reject)
		self.layout.addWidget(self.buttonBox)
		self.setLayout(self.layout)

	def updateLista(self):
		self.listW.clear()
		for i in range(len(self.parent.listaNumeMRC)):
			self.listW.addItem(str(self.parent.listaNumeMRC[i]))
		
		self.listW.setCurrentRow(len(self.listW)-1)
		self.listW.scrollToBottom()
		
	#def selectionChanged(self):
		#self.value1.setText(str(self.listW.currentRow()))

	def getResults(self):
		x=self.exec_()
		if  x== QDialog.Accepted:
			return self.listW.currentRow()
		else:
			return None

class CustomDialog3(QDialog):
	def __init__(self,parent=None):
		super(CustomDialog3, self).__init__(parent)
		self.setWindowTitle("Select distance map")
		self.layout=QVBoxLayout()
		self.parent=parent
		
		self.listW=QtWidgets.QListWidget()
		self.listW.setSelectionMode(1)#4 
		#self.listW.itemSelectionChanged.connect(self.selectionChanged)
		self.updateLista()
		
		#inputLay=QHBoxLayout()
		#text1=QtWidgets.QLabel()
		#text1.setText('Valoare')		
		#self.value1=QtWidgets.QLineEdit()
		#self.value1.setText(str(1))
		#inputLay.addWidget(text1)
		#inputLay.addWidget(self.value1)
		
		self.layout.addWidget(self.listW)
		#self.layout.addLayout(inputLay)
		
		
		QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
		self.buttonBox = QDialogButtonBox(QBtn)	
		self.buttonBox.accepted.connect(self.accept)
		self.buttonBox.rejected.connect(self.reject)
		self.layout.addWidget(self.buttonBox)
		self.setLayout(self.layout)

	def updateLista(self):
		self.listW.clear()
		for i in range(len(self.parent.listaNumeMRC)):
			self.listW.addItem(str(self.parent.listaNumeMRC[i]))
		
		self.listW.setCurrentRow(len(self.listW)-1)
		self.listW.scrollToBottom()
		
	#def selectionChanged(self):
		#self.value1.setText(str(self.listW.currentRow()))

	def getResults(self):
		x=self.exec_()
		if  x== QDialog.Accepted:
			return self.listW.currentRow()
		else:
			return None

class CustomDialog4(QDialog):
	def __init__(self,parent=None):
		super(CustomDialog4, self).__init__(parent)
		self.setWindowTitle("Scan")
		self.layout=QVBoxLayout()
		self.parent=parent
		
		self.figure0 = plt.figure() 
		self.canvas0 = FigureCanvas(self.figure0) 
		self.toolbar0 = NavigationToolbar(self.canvas0, self) 
		
		inputLay=QHBoxLayout()
		text1=QtWidgets.QLabel()
		text1.setText('Start')		
		self.value1=QtWidgets.QLineEdit()
		self.value1.setText(str(0))
		
		inputLay=QHBoxLayout()
		text2=QtWidgets.QLabel()
		text2.setText('Stop')		
		self.value2=QtWidgets.QLineEdit()
		self.value2.setText(str(1))
		
		inputLay=QHBoxLayout()
		text3=QtWidgets.QLabel()
		text3.setText('No. of Steps')	
		self.value3=QtWidgets.QLineEdit()
		self.value3.setText(str(10))
		
		button=QtWidgets.QPushButton("Run")
		button.clicked.connect(self.compute)
		
		
		
		inputLay.addWidget(text1)
		inputLay.addWidget(self.value1)
		inputLay.addWidget(text2)
		inputLay.addWidget(self.value2)
		inputLay.addWidget(text3)
		inputLay.addWidget(self.value3)
		inputLay.addWidget(button)
		
		self.layout.addLayout(inputLay)
		self.layout.addWidget(self.canvas0)
		self.layout.addWidget(self.toolbar0)
		#inputLay.addWidget(self.value1)
		
		#self.layout.addLayout(inputLay)
		
		
		QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
		self.buttonBox = QDialogButtonBox(QBtn)	
		self.buttonBox.accepted.connect(self.accept)
		self.buttonBox.rejected.connect(self.reject)
		self.layout.addWidget(self.buttonBox)
		self.setLayout(self.layout)
	def compute(self):
		if(len(self.parent.listaMRC)==0):
			return
		
		ax0 = self.figure0.add_subplot(111)
		ax0.set_xlim(float(self.value1.text()),float(self.value2.text()))
		x=self.parent.listaMRC[self.parent.currentItem]
		
		th_vec=[]
		surf_vec=[]
		for i in range(int(self.value3.text())):
			x1=x.copy()
			step=(float(self.value2.text())-float(self.value1.text()))/(int(self.value3.text())-1)
			th=float(self.value1.text())+i*step
			
			t1=time.time()
			tomo.applyThreshold_linear(x1,th,10)
			surf=tomo.surface(x1,10)
			volume=tomo.volume(x1,10)
			t2=time.time()
			print(t2-t1)
			
			th_vec.append(th)
			surf_vec.append(surf/(0.1+volume))
			
		ax0.scatter(th_vec,surf_vec)	
		self.canvas0.draw()
			
	def getResults(self):
		x=self.exec_()
		if  x== QDialog.Accepted:
			return self.listW.currentRow()
		else:
			return None
			
class CustomDialog5(QDialog):
	def __init__(self,parent=None):
		super(CustomDialog5, self).__init__(parent)
		self.setWindowTitle("Otsu Thresholding")
		self.layout=QVBoxLayout()
		self.parent=parent
		
		self.figure0 = plt.figure() 
		self.canvas0 = FigureCanvas(self.figure0) 
		self.toolbar0 = NavigationToolbar(self.canvas0, self) 
		
		inputLay=QHBoxLayout()
		text1=QtWidgets.QLabel()
		text1.setText('Start')		
		self.value1=QtWidgets.QLineEdit()
		self.value1.setText(str(0))
		
		inputLay=QHBoxLayout()
		text2=QtWidgets.QLabel()
		text2.setText('Stop')		
		self.value2=QtWidgets.QLineEdit()
		self.value2.setText(str(1))
		
		inputLay=QHBoxLayout()
		text3=QtWidgets.QLabel()
		text3.setText('No. of Steps')	
		self.value3=QtWidgets.QLineEdit()
		self.value3.setText(str(10))
		
		button=QtWidgets.QPushButton("Run")
		button.clicked.connect(self.compute)
		
		
		
		inputLay.addWidget(text1)
		inputLay.addWidget(self.value1)
		inputLay.addWidget(text2)
		inputLay.addWidget(self.value2)
		inputLay.addWidget(text3)
		inputLay.addWidget(self.value3)
		inputLay.addWidget(button)
		
		self.layout.addLayout(inputLay)
		self.layout.addWidget(self.canvas0)
		self.layout.addWidget(self.toolbar0)
		#inputLay.addWidget(self.value1)
		
		#self.layout.addLayout(inputLay)
		
		
		QBtn = QDialogButtonBox.Ok | QDialogButtonBox.Cancel
		self.buttonBox = QDialogButtonBox(QBtn)	
		self.buttonBox.accepted.connect(self.accept)
		self.buttonBox.rejected.connect(self.reject)
		self.layout.addWidget(self.buttonBox)
		self.setLayout(self.layout)
	def compute(self):
		self.figure0.clear()
		if(len(self.parent.listaMRC)==0):
			return
		
		ax0 = self.figure0.add_subplot(111)
		ax0.set_xlim(float(self.value1.text()),float(self.value2.text()))
		x=self.parent.listaMRC[self.parent.currentItem]
		
		xMin=float(self.value1.text())
		xMax=float(self.value2.text())
		n=int(self.value3.text())
		th_vec=[]
		surf_vec=[]
		for i in range(n):
			step=(xMax-xMin)/(n-1)
			th=xMin+i*step
			th_vec.append(th)
		
		otsu=np.array(tomo.otsu(x,xMin,xMax,n))
		print(type(tomo.otsu(x,xMin,xMax,n)))
		print(len(th_vec),len(otsu))
		ax0.set_yscale('log')
		ax0.plot(th_vec[1:-1],otsu,"-")	
		self.canvas0.draw()
			
	def getResults(self):
		x=self.exec_()
		if  x== QDialog.Accepted:
			return self.listW.currentRow()
		else:
			return None
class Worker(QObject):
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def run(self):
        """Long-running task."""
        i=0
        while True:
            time.sleep(1)
            self.progress.emit(i)
            i+=1
        self.finished.emit()

class Worker(QRunnable):
	'''
	Worker thread

	Inherits from QRunnable to handler worker thread setup, signals and wrap-up.

	:param callback: The function callback to run on this worker thread. Supplied args and
					 kwargs will be passed through to the runner.
	:type callback: function
	:param args: Arguments to pass to the callback function
	:param kwargs: Keywords to pass to the callback function

	'''

	def __init__(self, fn, *args, **kwargs):
		super(Worker, self).__init__()
		# Store constructor arguments (re-used for processing)
		self.fn = fn
		self.args = args
		self.kwargs = kwargs

	@pyqtSlot()
	def run(self):
		'''
		Initialise the runner function with passed args, kwargs.
		'''
		self.fn(*self.args, **self.kwargs)

class MainWindow2(QtWidgets.QMainWindow):
	def __init__(self):
		super(MainWindow2,self).__init__()
		self.setGeometry(0,0,1000,700)
		self.defaultValues={
			"th":1.,
			"erode":0,
			"dilate":0,
			"peak":0.02			
		}
		#self.showMaximized()
		#~ try:
			#~ self.configFile=eval(open("Configure.txt","r").read())
		#~ except:
			#~ f=open("Configure.txt",'w')
			#~ s="{\n'default':True,\n'fontSize':30,\n'lineWidth':2,\n'circleRadius':10,\n'lineWidth':2,\n'antialiasing':False,\n'scrollSpeed':0.3,\n'dequeMaxlen':25\n}"
			#~ f.write(s)
			#~ f.close()
			#~ self.configFile=eval(s)
		self.setWindowTitle("TOMO")		
		self.setWindowOpacity(1)
		# ~ self.setWindowFlags(self.windowFlags() | QtCore.Qt.FramelessWindowHint)
		self.createActions()
		self.createMenus()
		
		self.mainWidget=QtWidgets.QWidget(self)
		self.setCentralWidget(self.mainWidget)
		
		
		MainLayout=QtWidgets.QHBoxLayout()
		MainLayout.setAlignment(QtCore.Qt.AlignLeft)
		#MainLayout.setAlignment(QtCore.Qt.AlignTop)


		
		leftWidget=QtWidgets.QWidget(self)
		leftWidget.setFixedWidth(300)
		leftLayout=QGridLayout(leftWidget)
		#setColumnStretch de vazut alta data
		
		#self.scrollArea=QtWidgets.QScrollArea()		
		#self.scrollArea.setWidget(self.mainWidget)
		#self.scrollArea.setWidgetResizable(True)
		#self.scrollArea.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
		#self.scrollArea.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
		#self.mainWidget.setFixedSize(1800,1000)
		#self.setCentralWidget(self.scrollArea)
		
		
		
		
		self.distanceLabel=QtWidgets.QLabel()	
		
		self.memLabel=QtWidgets.QLabel()
		self.memLabel.setText(" - ")	
			
		self.statusLabel=QtWidgets.QLabel()
		self.statusLabel.setText("Welcome")		
		self.listW=QtWidgets.QListWidget()
		self.listW.setSelectionMode(1)#4 - multiple selection
		self.listW.itemDoubleClicked.connect(self.view_act3)

		self.deleteButton = QPushButton('Delete', self.mainWidget)
		self.deleteButton.clicked.connect(self.deleteItem)
		
		self.loadButton = QPushButton('Load', self.mainWidget)
		self.loadButton.clicked.connect(self.open_act)
		
		self.viewButton = QPushButton('View', self.mainWidget)
		self.viewButton.clicked.connect(self.view_act3)
		
		self.saveButton = QPushButton('Save', self.mainWidget)
		self.saveButton.clicked.connect(self.save_mrc)
		
		layButton=QtWidgets.QHBoxLayout()
		layButton.addWidget(self.loadButton)
		layButton.addWidget(self.viewButton)
		layButton.addWidget(self.deleteButton)
		layButton.addWidget(self.saveButton)

		
		#~ self.editPixInfo = QtWidgets.QLineEdit(self)
		#~ self.editPixInfo.setReadOnly(True)
		#~ self.editPixInfo.setText("Ioana")#setText('%d, %d' % (pos.x(), pos.y()))
		
		layThresh=QtWidgets.QHBoxLayout()
		self.threshButton=QPushButton("Threshold", self.mainWidget)
		self.threshButton.clicked.connect(self.applyThreshold)
		self.threshValue=QtWidgets.QLineEdit()
		self.threshValue.returnPressed.connect(self.applyThreshold)
		self.threshValue.setText("1")
		layThresh.addWidget(self.threshButton)
		layThresh.addWidget(self.threshValue)
		
		layED=QtWidgets.QHBoxLayout()
		self.erodeButton=QPushButton("Erode", self.mainWidget)
		self.erodeButton.clicked.connect(self.erode)
		self.erodeValue=QtWidgets.QLineEdit()
		self.erodeValue.returnPressed.connect(self.erode)
		self.erodeValue.setText("1")
		
		self.dilateButton=QPushButton("Dilate", self.mainWidget)
		self.dilateButton.clicked.connect(self.dilate)
		self.dilateValue=QtWidgets.QLineEdit()
		self.dilateValue.returnPressed.connect(self.dilate)
		self.dilateValue.setText("1")
		layED.addWidget(self.erodeButton)
		layED.addWidget(self.erodeValue)
		layED.addWidget(self.dilateButton)
		layED.addWidget(self.dilateValue)
		
		layDM=QtWidgets.QHBoxLayout()
		self.DMButton=QPushButton("Distance transform", self.mainWidget)
		self.DMButton.clicked.connect(self.distanceMap)
		#~ self.threshValue=QtWidgets.QLineEdit()
		#~ self.threshValue.setText("1")
		layDM.addWidget(self.DMButton)
		#~ layDM.addWidget(self.threshValue)
		
		layMaxima=QtWidgets.QHBoxLayout()
		self.maximaButton=QPushButton("Find local maxima", self.mainWidget)
		self.maximaButton.clicked.connect(self.findLocalMaxima)
		self.maximaValue=QtWidgets.QLineEdit()
		self.maximaValue.returnPressed.connect(self.findLocalMaxima)
		self.maximaValue.setText("0.02")
		layMaxima.addWidget(self.maximaButton)
		layMaxima.addWidget(self.maximaValue)
		
		self.fillButton=QPushButton("Fill Particles", self.mainWidget)
		self.fillButton.clicked.connect(self.fillParticles)
		
		layAuto=QtWidgets.QHBoxLayout()
		self.autoButton=QPushButton("Auto", self.mainWidget)
		self.autoButton.clicked.connect(self.autoComutation)
		layAuto.addWidget(self.autoButton)
		
		self.testButton=QPushButton("Segmentation", self.mainWidget)
		self.testButton.clicked.connect(self.showDialog4)
		
		# ~ self.colorButton=QPushButton("Color Button", self.mainWidget)
		# ~ self.colorButton.clicked.connect(self.changeColor)
		
		
		scanLayer=QHBoxLayout()
		self.scanButton=QPushButton("Scan", self.mainWidget)
		self.scanButton.clicked.connect(self.scan)
			
		self.otsuButton=QPushButton("Otsu", self.mainWidget)
		self.otsuButton.clicked.connect(self.otsu)
		scanLayer.addWidget(self.scanButton)
		scanLayer.addWidget(self.otsuButton)
		
		leftLayout.addWidget(self.memLabel,0,0)
		leftLayout.addWidget(self.statusLabel,1,0)
		leftLayout.addWidget(self.listW,2,0)

		leftLayout.addLayout(layButton,4,0)
		leftLayout.addLayout(scanLayer,5,0)
		leftLayout.addLayout(layThresh,6,0)
		leftLayout.addLayout(layED,7,0)
		leftLayout.addLayout(layDM,8,0)
		leftLayout.addLayout(layMaxima,9,0)
		leftLayout.addWidget(self.fillButton,10,0)
		leftLayout.addWidget(self.testButton,11,0)
		leftLayout.addLayout(layAuto,12,0)
		# ~ leftLayout.addWidget(self.colorButton,13,0)
		
		
		#~ MainLayout.addWidget()
		
		#~ self.rightWidget=QtWidgets.QWidget(self)
		#~ self.rightLayout=QtWidgets.QVBoxLayout(self.rightWidget)
		#~ self.rightLayout.setSpacing(2)
		
		#~ self.buttonsWidget=QtWidgets.QWidget()
		#~ self.buttonsWidget.setFixedWidth(100)
		#~ self.buttonsWidget.setFixedHeight(40)
		#~ self.buttonLayout=QtWidgets.QHBoxLayout(self.buttonsWidget)
		#~ self.button1=QtWidgets.QPushButton('3D')
		#~ self.button1.setCheckable(True)
		#~ self.button1.setChecked(True)
		#~ self.button1.clicked.connect(self.on_click_b1)
		
		#~ self.button2=QtWidgets.QPushButton('2D')
		#~ self.button2.setCheckable(True) 
		#self.button2.setChecked(False)
		#~ self.button2.clicked.connect(self.on_click_b2)
		
		#~ self.buttonLayout.addWidget(self.button1)
		#~ self.buttonLayout.addWidget(self.button2)
		
		#~ self.buttonGroup=QtWidgets.QButtonGroup()
		#~ self.buttonGroup.setExclusive(True)
		#~ self.buttonGroup.addButton(self.button1)
		#~ self.buttonGroup.addButton(self.button2)
		
		self.vtkFrame = QtWidgets.QFrame()
		self.vtk_widget = QVTKRenderWindowInteractor(self.vtkFrame)
		
		#pentru histograma
		self.figure0 = plt.figure() 
		self.canvas0 = FigureCanvas(self.figure0) 
		# ~ self.toolbar0 = NavigationToolbar(self.canvas0, self) 
		
		self.rangeSlider=RangeSlider(self)
		self.rangeSlider.valueChanged.connect(self.rangeSliderValueChange)
		
		splitter1 = QSplitter(Qt.Vertical)
		
		splitter1.addWidget(self.vtk_widget)
		splitter1.addWidget(self.canvas0)
		splitter1.addWidget(self.rangeSlider)
		# ~ splitter1.setSizes([10, 3])
		#~ self.vtkFrame2 = QtWidgets.QFrame()
		#~ self.vtk_widget2 = QVTKRenderWindowInteractor(self.vtkFrame2)
		#~ self.vtk_widget2.hide()
		self.setVTKWidget()
		
		
		
		
		self.Widget2d=QtWidgets.QWidget()
		lay2d=QtWidgets.QVBoxLayout(self.Widget2d)
		
		laySlider=QHBoxLayout()
		self.slider = QSlider()
		self.slider.setOrientation(Qt.Horizontal)
		self.slider.valueChanged.connect(self.sliderValueChange)
		self.slider.setSingleStep(100)
		
		self.sliderText=QtWidgets.QLineEdit()
		self.sliderText.setText(str(self.slider.value()))
		self.sliderText.setFixedWidth(30)
		
		self.sliderText.returnPressed.connect(self.sliderTextPressed)

		laySlider.addWidget(self.slider)
		laySlider.addWidget(self.sliderText)
		
		
		self.figure = plt.figure() 
		self.canvas = FigureCanvas(self.figure) 
		self.toolbar = NavigationToolbar(self.canvas, self) 
		#~ self.Widget2d.hide()
		
		layoutRadio = QHBoxLayout()
		self.b1 = QRadioButton("XY")
		self.b1.setChecked(True)
		self.b1.toggled.connect(lambda:self.btnstate(self.b1))		
		self.b2 = QRadioButton("YZ")
		self.b2.toggled.connect(lambda:self.btnstate(self.b2))
		self.b3 = QRadioButton("XZ")
		self.b3.toggled.connect(lambda:self.btnstate(self.b3))
		layoutRadio.addWidget(self.b1)
		layoutRadio.addWidget(self.b2)
		layoutRadio.addWidget(self.b3)
		layoutRadio.addStretch()
		
		self.buttonGroup = QButtonGroup()
		self.buttonGroup.addButton(self.b1,0)
		self.buttonGroup.addButton(self.b2,1)
		self.buttonGroup.addButton(self.b3,2)
		self.buttonGroup.buttonClicked.connect(self.changeAxis)
		
		lay2d.addLayout(layoutRadio)
		lay2d.addLayout(laySlider)
		lay2d.addWidget(self.canvas)
		lay2d.addWidget(self.toolbar)
		
        # Just some button connected to 'plot' method 
		#~ self.button = QPushButton('Plot') 
		#~ self.button.clicked.connect(self.plot) 
   
        # creating a Vertical Box layout 
        #~ layout = QVBoxLayout() 
        #~ layout.addWidget(self.toolbar) 
        #~ layout.addWidget(self.canvas) 
        #~ layout.addWidget(self.button) 
		#~ self.setLayout(layout) 

		
		#~ self.dispWidget=QtWidgets.QWidget()
		#~ self.dispLayout=QtWidgets.QHBoxLayout(self.dispWidget)		
		#~ self.dispLayout.addWidget(self.vtk_widget)
		#~ self.dispLayout.addWidget(self.Widget2d)
		
		self.tabs = QTabWidget()
		#~ self.tab1 = QWidget()
		#~ self.tab2 = QWidget()
		
		#self.tabs.addTab(self.vtk_widget,"3D")
		self.tabs.addTab(splitter1,"3D")
		self.tabs.addTab(self.Widget2d,"2D")
		
		#~ self.rightLayout.addWidget(self.buttonsWidget)
		#~ self.rightLayout.addWidget(self.dispWidget)
		
		MainLayout.addWidget(leftWidget)
		MainLayout.addWidget(self.tabs)
		self.mainWidget.setLayout(MainLayout)
		
		self.setAcceptDrops(True)
		
		QApplication.clipboard()
		
		self.shortcut1 = QShortcut(QKeySequence("Ctrl+C"), self)
		self.shortcut1.activated.connect(self.on_copy)
		self.cb=QApplication.clipboard()
		
		self.shortcut2 = QShortcut(QKeySequence("Ctrl+V"), self)
		self.shortcut2.activated.connect(self.view_act3)

		
		self.shortcut3 = QShortcut(QKeySequence("Ctrl+M"), self)
		self.shortcut3.activated.connect(self.seeMem)

		self.currentFolder=""
		self.listaMRC=[]
		self.listaNumpy=[]
		self.listaNumeMRC=[]
		self.currentItem=-1
		#self.plot()
		self.listW.itemSelectionChanged.connect(self.selectionChanged)
		self.minViz=0
		self.maxViz=0.99
		self.show()
		
		
		self.statusBar().showMessage('Message in statusbar.')
		self.progressBar = QProgressBar(self)
		self.progressBar.setFixedSize(200,10)
		
		self.statusBar().addPermanentWidget(self.progressBar)
		self.statusBar().addPermanentWidget(self.statusLabel)
        # This is simply to show the bar
		
		self.progressBar.setValue(100)
		
		
		
		self.threadpool = QThreadPool()
		print("Multithreading with maximum %d threads" % self.threadpool.maxThreadCount())
		#self.startThread()
		
		self.jobs=[]
		# ~ self.startThread2()
		
		
		# ~ self.startThread3()
		
	def startThread(self):
		self.thread = QThread()
		# Step 3: Create a worker object
		self.worker = Worker()
		# Step 4: Move worker to the thread
		self.worker.moveToThread(self.thread)
		# Step 5: Connect signals and slots
		self.thread.started.connect(self.worker.run)
		#self.worker.finished.connect(self.thread.quit)
		#self.worker.finished.connect(self.worker.deleteLater)
		#self.thread.finished.connect(self.thread.deleteLater)
		self.worker.progress.connect(self.seeMem)
		# Step 6: Start the thread
		self.thread.start()
		
	
	
	def seeMem(self):
		process = psutil.Process(os.getpid())
		# ~ print(psutil.virtual_memory().available/1024/1024)
		# ~ print(process.memory_info().rss/1024/1024)
		self.memLabel.setText(str(round(process.memory_info().rss/1024/1024,1))+" MB")
		self.statusBar().showMessage(str(round(process.memory_info().rss/1024/1024,1))+" MB")
		# ~ print(self.threadpool.activeThreadCount())
	
	def execute_this_fn(self):
		i=0
		while True:
		# ~ for i in range(100):
			self.progressBar.setValue(i)
			# ~ time.sleep(0.3)
			self.seeMem()
			print(self.jobs)
			i+=1
		self.statusBar().showMessage('Done')
		self.progressBar.setValue(100)
		

		
	def startThread2(self):
    # Pass the function to execute
		# ~ worker = Worker(self.execute_this_fn) # Any other args, kwargs are passed to the run function

    # Execute
		# ~ self.threadpool.start(worker)
		
		# ~ p=mp.Process(target=self.execute_this_fn)
		# ~ self.jobs.append(p)
		# ~ p.start()
		# ~ while 1:
			# ~ print("si eu merg")
		pass
	def startThread3(self):
		p=mp.Process(target=self.execute_this_fn)
		self.jobs.append(p)
		p.start()
		pass
	
	
	def addListaNumpy(self,c):
		x=np.array(c.getVal()).reshape([c.Nx,c.Ny,c.Nz],order='F')
		self.listaNumpy.append(x)
	
	def plotHistogram(self):
		if len(self.listaMRC)==0:
			self.figure0.clear()
			self.figure.clear()
			return
		
		
		#~ c=self.listaMRC[self.currentItem]
		x=self.listaNumpy[self.currentItem]
		self.figure0.clear()
		#~ volume=tomo_load.volume.copy();
		values=x.ravel()
		hyst=np.histogram(values,bins=100)
		counts=hyst[0]
		bins=hyst[1]
		nbins=np.shape(bins)[0]
		bincents=np.zeros(nbins-1)
		
		ax0 = self.figure0.add_subplot(111)
		
		for i in range (0,nbins-1):
			bincents[i]=bins[i]+(bins[i+1]-bins[i])/2.0
   
		X=bincents[0:int(nbins)]
		Y=counts[0:int(nbins)]
   
		ax0.set_yscale('log')
		ax0.set_ylim(bottom=np.min(counts)*0+0.1, top=np.max(counts)*1.2)
		
		#~ line, = ax0.fill(bincents[0:int(nbins)],counts[0:int(nbins)], facecolor='none')
		img_data = bincents[0:int(nbins)].reshape(1, bincents[0:int(nbins)].size)
		norm = mpl.colors.Normalize(vmin=np.min(values), vmax=np.max(values))
		cmap=plt.cm.plasma
		cmap_data=img_data
		
		dx=X[1]-X[0]
		N=float(X.size)
		
		ax0.plot(bincents[0:int(nbins)],counts[0:int(nbins)],'-',color='k',lw=0)
		for n, (x,y) in enumerate(zip(X,Y)):
			color = cmap(n/N)
			polygon = plt.Rectangle((x,0),dx,y,color=color)
			ax0.add_patch(polygon)
		
		self.canvas0.draw()
		return
		extent=(np.min(values), np.max(values),0,np.max(counts))
		ax0.imshow(cmap_data, extent=extent,aspect='auto', cmap=cmap)
		ax0.plot(bincents[0:int(nbins)],counts[0:int(nbins)],'-',color='k',lw=0)
		#ax0.set_clip_path(line)
		#~ plt.show()
		ax0.fill_between(bincents[0:int(nbins)],counts[0:int(nbins)],2*np.max(counts),color="w")
		self.canvas0.draw()
	
	
	def plot(self): 
		if len(self.listaMRC)==0:
			# ~ self.figure0.clear()
			self.figure.clear()
			return
		
		
		#~ c=self.listaMRC[self.currentItem]
		x=self.listaNumpy[self.currentItem]
		self.figure.clear() 
		#~ self.canvas.clear()
		ax = self.figure.add_subplot(111) 
		
		if self.buttonGroup.checkedId()==0:
			ax.imshow(x[:,:,self.slider.value()],cmap="plasma") 
		if self.buttonGroup.checkedId()==1:
			ax.imshow(x[self.slider.value(),:,:].T,cmap="plasma")
		if self.buttonGroup.checkedId()==2:
			ax.imshow(x[:,self.slider.value(),:].T,cmap="plasma")
		#~ x[:,:,self.slider.value()]
		self.canvas.draw()
		

	def updateLista(self):
		self.listW.clear()
		for i in range(len(self.listaNumeMRC)):
			self.listW.addItem(str(self.listaNumeMRC[i]))
		
		self.listW.setCurrentRow(len(self.listW)-1)
		self.currentItem=len(self.listW)-1
		
		self.listW.scrollToBottom()
					
		self.view_act3()
		self.plotHistogram()
		self.plot()
	
	def selectionChanged(self):
		print(len(self.listW.selectedItems()))
		#for item in self.listW.selectedItems():
			#print(item.text())
			#print(self.listW.currentRow())
		self.currentItem=self.listW.currentRow()
		#print(self.currentFolder+"/"+self.listaNumeMRC[self.currentItem])
		

			
	def save_mrc(self):
		if len(self.listaMRC)==0:
			return
		c=self.listaMRC[self.currentItem]
		defaultName=self.listaNumeMRC[self.currentItem]
		options = QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		fileName, _ = QFileDialog.getSaveFileName(self,"Save file",defaultName[0:-4],"All Files (*);;MRC Files (*.mrc)", options=options)
		if fileName:
			tomo.writeMRC(c,fileName)
			
			
	def on_copy(self):
		s=""
		for i in self.listW.selectedItems():
			s+=i.text()+"\n"
		self.cb.setText(s)
		return s
		
	def lista_string(self):
		s=""
		for i in range(self.listW.count()-1):
			s+=self.listW.item(i).text()+"\n"
		return s
		
	def on_click(self):
		self.listW.setCurrentRow(len(self.listW)-1)
		
	def createActions(self):
		self.openAct = QtWidgets.QAction("&Open...", self, shortcut="Ctrl+O", triggered=self.open_act)
		self.helpAct = QAction("&Help...", self, shortcut="Ctrl+H", enabled=True, triggered=self.help_act)
		self.saveAct=QAction("&Save...", self, shortcut="Ctrl+S", enabled=True, triggered=self.save_mrc)
		self.exitAct = QtWidgets.QAction("&Exit", self, shortcut="Ctrl+Q", triggered=self.close)
		self.applyConfigAct=QtWidgets.QAction("&Configure from file",self,shortcut="Ctrl+1", triggered=self.apply_config)
		self.applyConfigAct2=QtWidgets.QAction("&Configure", self, shortcut="Ctrl+2",triggered=self.apply_config2)
	
	def createMenus(self):
		self.fileMenu = QtWidgets.QMenu("&File", self)
		self.fileMenu.addAction(self.openAct)
		self.fileMenu.addAction(self.saveAct)
		self.fileMenu.addSeparator()
		self.fileMenu.addAction(self.exitAct)
		
		
		self.helpMenu = QtWidgets.QMenu("&Help", self)
		self.helpMenu.addAction(self.helpAct)
		
		self.editMenu = QtWidgets.QMenu("&Edit", self)
		self.editMenu.addAction(self.applyConfigAct)
		self.editMenu.addAction(self.applyConfigAct2)
		self.menuBar().addMenu(self.fileMenu)
		self.menuBar().addMenu(self.editMenu)
		self.menuBar().addMenu(self.helpMenu)
		
	def open_act(self):
		
		self.statusLabel.setText("Working")
		options = QtWidgets.QFileDialog.Options()
		options |= QFileDialog.DontUseNativeDialog
		# fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
		fileName, _ = QtWidgets.QFileDialog.getOpenFileName(self, 'Open Image', '12345','mrc (*)', options=options)
		#self.viewer2 = PhotoViewer2(self.mainWidget)
		if fileName:
			self.fileName=fileName
			self.listaNumeMRC.append(fileName.split("/")[-1])
			self.currentFolder="/".join(fileName.split("/")[0:-1])
			
			nume=self.currentFolder+"/"+self.listaNumeMRC[-1]
			c=tomo.cubee(nume)
			self.listaMRC.append(c)
			self.addListaNumpy(c)
					
		self.updateLista()
		self.statusLabel.setText("Done1")
		print(123)
	def save_act(self):
		pass
		
		
	def help_act(self):
		
		helpDialog = QtWidgets.QDialog(self) # Added
		helpDialog.setAttribute(QtCore.Qt.WA_DeleteOnClose) # Added
		browser = QtWidgets.QTextBrowser()
		browser.append('Open Image: File -> Open or Ctrl+O or Drag & Drop')
		browser.append('Zoom: Z+Scroll')
		browser.append('Line: RightClick+D')
		browser.append('Point: RightClick+X\n ')
		browser.append("Have fun :)")
		
		layout = QtWidgets.QVBoxLayout()
		layout.addWidget(browser)
		helpDialog.setLayout(layout) # Added
		helpDialog.setWindowTitle("Instructions") # Added for neatness
		helpDialog.show()
				
	def apply_config(self):
		try:
			self.configFile=eval(open("Configure.txt","r").read())
		except:
			print("Something went wrong")
		if(self.configFile["default"]==False):
			self.viewer2.applyConfig(self.configFile)
						
	def apply_config2(self):
		w = CustomDialog(self.configFile)
		values = w.getResults()
		if(values):
			self.viewer2.applyConfig(values)
			self.configFile=values
		return 
		
	def keyPressEvent(self, e):
		if e.key() == Qt.Key_Delete:
			self.deleteItem()
		if e.key() == Qt.Key_Escape:
			self.close()
			
	def reset_act(self):
		if self.fileName:
			self.viewer2.open(self.fileName)
			
	def view_act(self):
		pass
	
	
	
		
	def setVTKWidget(self):
				
		self.ren1 = vtk.vtkRenderer()

		self.vtk_widget.GetRenderWindow().AddRenderer(self.ren1)
		self.vtk_widget.show()

		#~ self.renWin=self.vtk_widget.GetRenderWindow()

		self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()
		self.iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
		self.iren.SetRenderWindow(self.vtk_widget.GetRenderWindow())
		self.iren.LightFollowCameraOn()		
		
		self.ren1.SetBackground(0.18,0.18,0.18)
		self.ren1.GetActiveCamera().Azimuth(45)
		self.ren1.GetActiveCamera().Elevation(30)
		self.ren1.ResetCameraClippingRange()
		self.ren1.ResetCamera()
		
		self.vtk_widget.GetRenderWindow().Render()
		self.iren.Initialize()
		self.iren.Start()

		

	
	def resetRender(self):
		self.ren1.RemoveAllViewProps()
		#~ self.ren1.ResetCamera()
		self.vtk_widget.GetRenderWindow().Render()
	
	

	def view_act3(self):
		if len(self.listaMRC)==0:
			print("0 elements in list")
			self.resetRender()
			return
		
		self.resetRender()
		# ~ reader = vtk.vtkMRCReader()
		# ~ reader.SetFileName("sfere_filled2.mrc")
		
		c=self.listaMRC[self.currentItem]
		
		t1=time.time()
		x=self.listaNumpy[self.currentItem].ravel('F')#reshape([c.Nx,c.Ny,c.Nz],order='F').ravel()
		t2=time.time()
		print(t2-t1)
		self.setSliderMinMax()
		self.plot()
		self.plotHistogram()
		
		
		dataImporter = vtk.vtkImageImport()
		#data_string = x.tostring()
		#dataImporter.CopyImportVoidPointer(data_string, len(data_string))
		dataImporter.SetImportVoidPointer(x)
		dataImporter.SetDataScalarTypeToDouble()
		dataImporter.SetNumberOfScalarComponents(1)
		dataImporter.SetDataExtent(0,c.Nx-1,0,c.Ny-1,0,c.Nz-1)
		dataImporter.SetWholeExtent(0,c.Nx-1,0,c.Ny-1,0,c.Nz-1)
		
		opacityTransferFunction = vtk.vtkPiecewiseFunction()
		#opacityTransferFunction.AddPoint(np.min(x)+(0.02+self.minViz*(np.max(x)-np.min(x)), 0.0)
		# ~ opacityTransferFunction.AddPoint(0.5*np.max(x), 0.4)
		#opacityTransferFunction.AddPoint(self.maxViz*np.max(x), 1)

		opacityTransferFunction.AddPoint(np.min(x)+0.02*(np.max(x)-np.min(x)), 0.0)
		opacityTransferFunction.AddPoint(np.min(x)+(0.02+self.minViz)*(np.max(x)-np.min(x)), 0.0)
		opacityTransferFunction.AddPoint(self.maxViz*np.max(x), 1)
		opacityTransferFunction.AddPoint(1*np.max(x), 1)

		funcOpacityGradient = vtk.vtkPiecewiseFunction()

		funcOpacityGradient.AddPoint(1,   0.0)
		funcOpacityGradient.AddPoint(5,   0.1)
		funcOpacityGradient.AddPoint(100,   1.0)

		# Create transfer mapping scalar value to color.
		colorTransferFunction = vtk.vtkColorTransferFunction()
		colorTransferFunction.AddRGBPoint(0.0, 47/255,0/255, 135/255)
		colorTransferFunction.AddRGBPoint(0.14*np.max(x), 98/255,0/255, 164/255)
		colorTransferFunction.AddRGBPoint(0.29*np.max(x), 146/255,0/255, 166/255)
		colorTransferFunction.AddRGBPoint(0.43*np.max(x), 186/255,47/255, 138/255)
		colorTransferFunction.AddRGBPoint(0.57*np.max(x), 216/255,91/255, 105/255)
		colorTransferFunction.AddRGBPoint(0.71*np.max(x), 238/255,137/255, 73/255)
		colorTransferFunction.AddRGBPoint(0.86*np.max(x), 246/255,189/255, 39/255)
		colorTransferFunction.AddRGBPoint(1*np.max(x), 228/255,250/255, 21/255)
		
		# The property describes how the data will look.
		self.volumeProperty = vtk.vtkVolumeProperty()
		self.volumeProperty.SetColor(colorTransferFunction)
		self.volumeProperty.SetScalarOpacity(opacityTransferFunction)
		#~ volumeProperty.SetGradientOpacity(funcOpacityGradient)
		self.volumeProperty.ShadeOff()
		self.volumeProperty.SetInterpolationTypeToLinear()
		#~ volumeProperty.setScalarOpacityUnitDistance( 4.5)
		#~ volumeProperty.SetShade(0,0)
		#volumeProperty.SetAmbient(0,0)
		#volumeProperty.SetDiffuse(1,2)
		#volumeProperty.SetSpecular(0,0)
		#volumeProperty.SetSpecularPower(0,20)
			 
		volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()#vtk.vtkFixedPointVolumeRayCastMapper()
		volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
		volumeMapper.SetBlendModeToComposite();	   
		volumeMapper.UseJitteringOff()
		
		
		dodecahedron = MakeDodecahedron()

		# Visualize
		# ~ mapper = vtk.vtkPolyDataMapper()
		# ~ mapper.SetInputData(dodecahedron.GetPolyData())
		
		mapper1 = vtk.vtkImageSliceMapper()
		mapper1.SetInputConnection(dataImporter.GetOutputPort())
		# ~ actor = vtk.vtkActor()
		mapper1.SetOrientationToX()
		mapper1.SetSliceNumber(c.Nx//2)
		
		mapper2 = vtk.vtkImageSliceMapper()
		mapper2.SetInputConnection(dataImporter.GetOutputPort())
		# ~ actor = vtk.vtkActor()
		mapper2.SetOrientationToY()
		mapper2.SetSliceNumber(c.Ny//2)
		
		mapper3 = vtk.vtkImageSliceMapper()
		mapper3.SetInputConnection(dataImporter.GetOutputPort())
		# ~ actor = vtk.vtkActor()
		mapper3.SetOrientationToZ()
		mapper3.SetSliceNumber(c.Nz//2)
		
		
		actor1 = vtk.vtkImageActor()
		actor1.SetMapper(mapper1)
		actor2 = vtk.vtkImageActor()
		actor2.SetMapper(mapper2)
		actor3 = vtk.vtkImageActor()
		actor3.SetMapper(mapper3)
		actor1.GetProperty().SetOpacity(0.7)
		actor2.GetProperty().SetOpacity(0.7)
		actor3.GetProperty().SetOpacity(0.7)
		
		#~ axes = vtk.vtkAxesActor()
		#~ axes.SetScale(1000)
#  The axes are positioned with a user transform
		#~ axes.SetUserTransform(transform)
		#~ axes.GetXAxisCaptionActor2D().GetCaptionTextProperty().SetColor(1,0,0);
 # the actual text of the axis label can be changed:
		#~ axes.SetXAxisLabelText("test");
 		#~ self.ren1.AddActor(axes)
 
		
		# The volume holds the mapper and the property and
		# can be used to position/orient the volume.
		volume = vtk.vtkVolume()
		volume.SetMapper(volumeMapper)
		volume.SetProperty(self.volumeProperty)
		#~ volume.GetProperty().setScalarOpacityUnitDistance(0, 4.5);
		
		#-------------------------------------------------------
		
		
		self.slicer = vtk.vtkImageReslice()
		# ~ self.slicer.SetInput(volume)
		self.slicer.SetInputConnection(dataImporter.GetOutputPort() )
		self.slicer.SetOutputDimensionality(2)

		# the next filter provides a mechanism for slice selection
		# ~ self.selector = SliceSelector(self.volume)
		# ~ self.slicer.SetResliceAxes( self.selector.GetDirectionCosines() )
		# ~ self.slicer.SetResliceAxesOrigin( self.selector.GetAxesOrigin() )
		self.slicer.SetResliceAxesDirectionCosines( [1,0,0],[0,1,0],[0,0,1])
		self.slicer.SetResliceAxesOrigin( [50,50,60] )
		# setup link for adjusting the contrast of the image
		# ~ r = volume.GetScalarRange()
		# ~ self.lutBuilder = LUTBuilder(r[0],r[1],1)
		# ~ lut = self.lutBuilder.Build()

		self.colors = vtk.vtkImageMapToColors()
		self.colors.SetInputConnection(  mapper1.GetOutputPort() )
		self.colors.SetLookupTable(colorTransferFunction)
		# ~ self.colors.SetLookupTable( lut )

		self.actor = vtk.vtkImageActor()
		self.actor.GetMapper().SetInputConnection( self.colors.GetOutputPort() )
        
		#-------------------------------------------------------
		
		
		
		
		
		self.ren1.AddActor(self.actor)
		# ~ self.ren1.AddActor(actor2)
		# ~ self.ren1.AddActor(actor3)
		self.ren1.AddVolume(volume)
		self.ren1.SetBackground(0.18,0.18,0.18)#colors.GetColor3d("Wheat")
		self.ren1.GetActiveCamera().Azimuth(45)
		self.ren1.GetActiveCamera().Elevation(30)
		self.ren1.ResetCameraClippingRange()
		self.ren1.ResetCamera()

		#renWin.SetSize(600, 600)
		#renWin.Render()
		self.vtk_widget.GetRenderWindow().Render()

	def on_click_b1(self):
		self.vtk_widget.show()
		self.vtk_widget2.hide()
		self.Widget2d.hide()
		if self.button1.isChecked():
			
			pass
		else:
			print(1)
			#self.vtkFrame.show()
			#self.vtkFrame2.hide()

	def on_click_b2(self):
		self.vtk_widget.hide()
		self.vtk_widget2.show()	
		self.Widget2d.show()
		if self.button2.isChecked():
			pass
		else:
			1
			#self.vtkFrame.hide()
			#self.vtkFrame2.show()
			
			
	def dragEnterEvent(self, e):
		if e.mimeData().hasUrls:
			e.accept()
			print("enter")
		else:
			e.ignore()
			print("enter2")


	def dragMoveEvent(self, e):
		if e.mimeData().hasUrls:
			e.accept()
		else:
			e.ignore()
	def dropEvent(self, e):
		"""
		Drop files directly onto the widget
		File locations are stored in fname
		:param e:
		:return:
		"""
		
		#files=[unicode(u.toLocalFile()) for u in e.mimeData().urls()]
		
		listaNume=[e.mimeData().urls()[i].toLocalFile() for i in range(len(e.mimeData().urls()))]
		#name=e.mimeData().urls()[0].toLocalFile()
		print(listaNume)
		for name in listaNume:		
			self.listaNumeMRC.append(name.split("/")[-1])
			self.currentFolder="/".join(name.split("/")[0:-1])
			
		
		for i in range(len(self.listaNumeMRC)):
			nume=self.currentFolder+"/"+self.listaNumeMRC[i]
			c=tomo.cubee(nume)
			#~ del c
			self.listaMRC.append(c)
			self.addListaNumpy(c)
		self.updateLista()
		#self.open(name)
		return 
	def deleteItem(self):
		
		if len(self.listaMRC)==0:
			return
		del self.listaMRC[self.currentItem]
		del self.listaNumeMRC[self.currentItem]
		del self.listaNumpy[self.currentItem]
		self.updateLista()
		
	def applyThreshold(self):
		self.statusLabel.setText("Working...")
		self.statusLabel.repaint()
		c=self.listaMRC[self.currentItem].copy()
		tomo.applyThreshold_linear(c,float(self.threshValue.text()))
		self.listaMRC.append(c)
		self.addListaNumpy(c)
		self.listaNumeMRC.append(self.listaNumeMRC[self.currentItem]+"TH")
		self.updateLista()
		self.statusLabel.setText("Done")
	
	
	def erode(self):
		self.statusLabel.setText("Working...")
		self.statusLabel.repaint()
		c=self.listaMRC[self.currentItem].copy()
		tomo.erodeGrayscale(c,int(self.erodeValue.text()))
		self.listaMRC.append(c)
		self.addListaNumpy(c)
		self.listaNumeMRC.append(self.listaNumeMRC[self.currentItem]+"Erode")
		self.updateLista()
		self.statusLabel.setText("Done")
	def dilate(self):
		self.statusLabel.setText("Working...")
		self.statusLabel.repaint()
		c=self.listaMRC[self.currentItem].copy()
		tomo.dilateGrayscale(c,int(self.dilateValue.text()))
		self.listaMRC.append(c)
		self.addListaNumpy(c)
		self.listaNumeMRC.append(self.listaNumeMRC[self.currentItem]+"Dilate")
		self.updateLista()
		self.statusLabel.setText("Done")
	def distanceMap(self):
		self.statusLabel.setText("Working...")
		self.statusLabel.repaint()
		c=self.listaMRC[self.currentItem].copy()
		c1=tomo.distanceMapGeneralEfficient(c,10)
		c1.normalize()
		self.listaMRC.append(c1)
		self.addListaNumpy(c1)
		self.listaNumeMRC.append(self.listaNumeMRC[self.currentItem]+"DM")
		self.updateLista()
		self.statusLabel.setText("Done")
	def sliderValueChange(self):
		txt = str(self.slider.value())
		#self.label3.setText(txt)
		self.sliderText.setText(txt)
		print("bla"+txt)
		if(self.currentItem>=0):
			self.plot()
	
	def sliderTextPressed(self):
		try:
			if int(self.sliderText.text())<=self.slider.maximum() and  int(self.sliderText.text())>=self.slider.minimum():
				self.slider.setValue(int(self.sliderText.text()))
			else:
				self.sliderText.setText(str(self.slider.value()))
		except:
			self.sliderText.setText(str(self.slider.value()))
			
	def btnstate(self,b):
		if b.text() == "Button1":
			if b.isChecked() == True:
				print(b.text()+" is selected")
			else:
				print(b.text()+" is deselected")
				
		if b.text() == "Button2":
			if b.isChecked() == True:
				print(b.text()+" is selected")
			else:
				print(b.text()+" is deselected")
	
	def changeAxis(self):
		self.setSliderMinMax()
		self.plot()
	def setSliderMinMax(self):
		if len(self.listaMRC)==0:
			return
		c=self.listaMRC[self.currentItem]
		if self.buttonGroup.checkedId()==0:
			minim=0
			maxim=c.Nz-1
			self.slider.setMinimum(minim)
			self.slider.setMaximum(maxim)
			self.slider.setValue(int(0.5*(minim+maxim)))
		if self.buttonGroup.checkedId()==1:
			minim=0
			maxim=c.Nx-1
			self.slider.setMinimum(minim)
			self.slider.setMaximum(maxim)
			self.slider.setValue(int(0.5*(minim+maxim)))
		if self.buttonGroup.checkedId()==2:
			minim=0
			maxim=c.Ny-1
			self.slider.setMinimum(minim)
			self.slider.setMaximum(maxim)
			self.slider.setValue(int(0.5*(minim+maxim)))
	def findLocalMaxima(self):
		self.statusLabel.setText("Working")
		self.statusLabel.repaint()
		c=self.listaMRC[self.currentItem].copy()
		c.normalize()
		marker=c-float(self.maximaValue.text())
		maxime=c-tomo.morphologicalReconstructionHybrid(marker,c)
		tomo.applyThreshold_linear(maxime,0.01,10);
		x=tomo.fillParticlesRandom_2(maxime,10,10+1);
		
		self.listaMRC.append(maxime)
		self.addListaNumpy(maxime)
		self.listaNumeMRC.append(self.listaNumeMRC[self.currentItem]+"Max")
		self.updateLista()
		self.statusLabel.setText("Done")
		
	def autoComutation(self):
		self.statusLabel.setText("Working...")
		self.statusLabel.repaint()
		if len(self.listaMRC)==0:
			print("0 elements in list")
			return
		c=self.listaMRC[self.currentItem]
		
		d=CustomDialog(self.defaultValues)
		vals=d.getResults()
		print(vals)
		
		print(1)
		if not vals:
			return
		print(type(vals["th"]))
		tomo.applyThreshold_linear(c,vals["th"],10)
		c1=tomo.distanceMapGeneralEfficientMic(c,10)
		tomo.blurGrayscale(c1,1)
		tomo.writeMRC(c1,"tomotDM.mrc")
		print(2)
		#~ marker=c1-
		#~ maxime=c1-tomo.morphologicalReconstructionHybrid(marker,c1)

		#~ maxime1=maxime.copy()
		print(3)
		c1.normalize()
		marker=c1-vals["peak"]
		maxime=c1-tomo.morphologicalReconstructionHybrid(marker,c1)
		maxime1=maxime.copy()
		print(4)
		tomo.applyThreshold_linear(maxime,0.01);
		tomo.writeMRC(maxime,"tomotMax.mrc")
		if vals["erode"]>0:
			tomo.erodeGrayscale(maxime,vals["erode"])
		if vals["dilate"]>0:
			tomo.dilateGrayscale(maxime,vals["dilate"])
		
		x=tomo.fillParticlesRandom_2(maxime,10,11);
		tomo.writeMRC(maxime,"tomotMaxFilled.mrc")
		print(5)
		print(x)
		flo1=tomo.priorityFlood(c1,maxime);
		tomo.writeMRC(flo1,"tomot.mrc")
		
		self.listaMRC.append(flo1)
		self.addListaNumpy(flo1)
		self.listaNumeMRC.append(self.listaNumeMRC[self.currentItem]+"flo1")
		self.updateLista()
		print(6)
		
	def showDialog(self,n):
		print(1)
		sender = self.sender()
		text , ok = QInputDialog.getText(self,"Modify name!","Please enter your name:")
		if sender == self.testButton:
			text , ok = QInputDialog.getText(self,"Modify name!","Please enter your name:")
			if ok:
				self.lb12.setText(text)
	def showDialog2(self):
		p = Process(target=self.showDialog, args=(1,))
		p.start()
		p.join()
	def showDialog3(self):
		dialog=QDialog(self)
		edit = QLineEdit("Write my name here..")
		edit.setText(str(self.currentItem))
		button = QPushButton("Show Greetings")
		layout = QVBoxLayout()
		layout.addWidget(edit)
		layout.addWidget(button)
# Set dialog layout
		dialog.setLayout(layout)
		dialog.show()
		
	def showDialog4(self):
		maxDialog=CustomDialog2(self)
		
		markerIndex=maxDialog.getResults()
		if not markerIndex:
			return
		
		dmDialog=CustomDialog3(self)
		dmIndex=dmDialog.getResults()
		if not dmIndex:
			return
			
		c=self.listaMRC[dmIndex]
		maximeFilled=self.listaMRC[markerIndex]
		flo1=tomo.priorityFlood(c,maximeFilled);
		
		self.listaMRC.append(flo1)
		self.addListaNumpy(flo1)
		self.listaNumeMRC.append(self.listaNumeMRC[self.currentItem]+"flood")
		self.updateLista()
	
	def fillParticles(self):
		self.statusLabel.setText("Working...")
		self.statusLabel.repaint()
		c=self.listaMRC[self.currentItem].copy()
		x=tomo.fillParticlesRandom_2(c,10,10+1);
		
		self.listaMRC.append(c)
		self.addListaNumpy(c)
		self.listaNumeMRC.append(self.listaNumeMRC[self.currentItem]+"Max")
		self.updateLista()
		self.statusLabel.setText("Done")
		
	def rangeSliderValueChange(self):
		#~ print(1)
		self.minViz=(self.rangeSlider.getRange()[0])/1000
		self.maxViz=(self.rangeSlider.getRange()[1])/1000
		self.changeColor()
		print(self.minViz,self.maxViz)

		'''
	def changeColor(self):
		x=self.listaNumpy[self.currentItem]
		opacityTransferFunction = vtk.vtkPiecewiseFunction()
		
		opacityTransferFunction.AddPoint(np.min(x)+0.02*(np.max(x)-np.min(x)), 0.0)
		opacityTransferFunction.AddPoint(np.min(x)+(0.02+self.minViz)*(np.max(x)-np.min(x)), 0.0)
		# ~ opacityTransferFunction.AddPoint(0.5*np.max(x), 0.4)
		opacityTransferFunction.AddPoint(self.maxViz*np.max(x), 1)
		opacityTransferFunction.AddPoint((self.maxViz+0.001)*np.max(x), 0)
		opacityTransferFunction.AddPoint(1*np.max(x), 0)
		# ~ colorTransferFunction = vtk.vtkColorTransferFunction()
		# ~ r=[72,79,67,52,34,61,147,243]
		# ~ g=[0,48,91,127,162,195,219,233]
		# ~ b=[84,127,141,142,135,108,53,28]
		# ~ colorTransferFunction.AddRGBPoint(0.0           , r[0]/255, g[0]/255, b[0]/255)
		# ~ colorTransferFunction.AddRGBPoint(0.14*np.max(x), r[1]/255, g[1]/255, b[1]/255)
		# ~ colorTransferFunction.AddRGBPoint(0.29*np.max(x), r[2]/255, g[2]/255, b[2]/255)
		# ~ colorTransferFunction.AddRGBPoint(0.43*np.max(x), r[3]/255, g[3]/255, b[3]/255)
		# ~ colorTransferFunction.AddRGBPoint(0.57*np.max(x), r[4]/255, g[4]/255, b[4]/255)
		# ~ colorTransferFunction.AddRGBPoint(0.71*np.max(x), r[5]/255, g[5]/255, b[5]/255)
		# ~ colorTransferFunction.AddRGBPoint(0.86*np.max(x), r[6]/255, g[6]/255, b[6]/255)
		# ~ colorTransferFunction.AddRGBPoint(1*np.max(x),    r[7]/255, g[7]/255, b[7]/255)
		# ~ self.volumeProperty.SetColor(colorTransferFunction)
		self.volumeProperty.SetScalarOpacity(opacityTransferFunction)
		self.vtk_widget.GetRenderWindow().Render()
'''
	def scan(self):
		dmDialog=CustomDialog4(self)
		dmDialog.show()
	def otsu(self):
		dmDialog=CustomDialog5(self)
		dmDialog.show()
	
app=QtWidgets.QApplication([])
GUI=MainWindow2()
#GUI=QImageViewer()
app.exec_()




'''
def view_act2(self):
		x=np.array([0,0,0,0,0,0,0,0,0,0,0,0,0,1.,0,0,0,0,0,0,0,0,0,0,0,0,0]).reshape([3,3,3])
		vimage=vtk.vtkImageData()
		vtype=numpy_support.get_vtk_array_type(x.dtype)
		#vimage.SetDimensions()
		VTK_data = numpy_support.numpy_to_vtk(num_array=x.ravel(), deep=True, array_type=vtype)
		colors = vtk.vtkNamedColors()
		
		
		img_vtk = vtk.vtkImageData()
		img_vtk.SetDimensions(x.shape)
		img_vtk.GetPointData().SetScalars(VTK_data)


		implicit_volume = vtk.vtkImplicitVolume()  # I want a vtkImplicitDataSet, whose input is a vtkDataSet
		implicit_volume.SetVolume(img_vtk)


		print(type(VTK_data))
		# This is a simple volume rendering example that
		# uses a vtkFixedPointVolumeRayCastMapper

		# Create the standard renderer, render window
		# and interactor.
		ren1 = vtk.vtkRenderer()

		renWin = vtk.vtkRenderWindow()
		renWin.SetWindowName("TOMO 3D")
		renWin.AddRenderer(ren1)

		#~ self.vtk_widget.GetRenderWindow().AddRenderer(ren1)
		#~ iren = self.vtk_widget.GetRenderWindow().GetInteractor()
		self.vtk_widget.show()


		iren = vtk.vtkRenderWindowInteractor()
		iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
		iren.SetRenderWindow(renWin)
		#~ iren.SetRenderWindow(self.vtk_widget.GetRenderWindow())
		iren.LightFollowCameraOn()
		# Create the reader for the data.
		reader = vtk.vtkMRCReader()
		reader.SetFileName("sfere_filled2.mrc")
		
		c=self.listaMRC[self.currentItem]
		x=np.array(c.getVal())#.reshape([c.Nx,c.Ny,c.Nz],order='C')
		#~ y=np.asfortranarray(x)
		print(x.dtype)
		#~ vimage=vtk.vtkImageData()
		#~ vtype=numpy_support.get_vtk_array_type(x.dtype)
    #~ #vimage.SetDimensions()
		#~ VTK_data = numpy_support.numpy_to_vtk(num_array=x.ravel(), deep=True, array_type=vtype)
		colors = vtk.vtkNamedColors()
	# imports raw data and stores it.
		
		dataImporter = vtk.vtkImageImport()
# The previously created array is converted to a string of chars and imported.
		data_string = x.tostring()
 
		#dataImporter.CopyImportVoidPointer(data_string, len(data_string))
		dataImporter.SetImportVoidPointer(x)
# The type of the newly imported data is set to unsigned char (uint8)
		dataImporter.SetDataScalarTypeToDouble()
# Because the data that is imported only contains an intensity value (it isn't RGB-coded or something similar), the importer
# must be told this is the case.
		dataImporter.SetNumberOfScalarComponents(1)
		dataImporter.SetDataExtent(0, c.Nx-1, 0, c.Ny-1, 0, c.Nz-1)
		dataImporter.SetWholeExtent(0, c.Nx-1, 0, c.Ny-1,0, c.Nz-1)
		
		

		# Create transfer mapping scalar value to opacity.
		opacityTransferFunction = vtk.vtkPiecewiseFunction()
		opacityTransferFunction.AddPoint(10, 0.0)
		opacityTransferFunction.AddPoint(0.5*360, 0.05)
		opacityTransferFunction.AddPoint(1*360, 0.2)

		funcOpacityGradient = vtk.vtkPiecewiseFunction()

		funcOpacityGradient.AddPoint(1,   0.0)
		funcOpacityGradient.AddPoint(5,   0.1)
		funcOpacityGradient.AddPoint(100,   1.0)

		# Create transfer mapping scalar value to color.
		colorTransferFunction = vtk.vtkColorTransferFunction()
		colorTransferFunction.AddRGBPoint(0.0, 47/255,0/255, 135/255)
		colorTransferFunction.AddRGBPoint(0.14*255, 98/255,0/255, 164/255)
		colorTransferFunction.AddRGBPoint(0.29*255, 146/255,0/255, 166/255)
		colorTransferFunction.AddRGBPoint(0.43*255, 186/255,47/255, 138/255)
		colorTransferFunction.AddRGBPoint(0.57*255, 216/255,91/255, 105/255)
		colorTransferFunction.AddRGBPoint(0.71*255, 238/255,137/255, 73/255)
		colorTransferFunction.AddRGBPoint(0.86*255, 246/255,189/255, 39/255)
		colorTransferFunction.AddRGBPoint(1*255, 228/255,250/255, 21/255)
		
		# The property describes how the data will look.
		volumeProperty = vtk.vtkVolumeProperty()
		volumeProperty.SetColor(colorTransferFunction)
		volumeProperty.SetScalarOpacity(opacityTransferFunction)
		#~ volumeProperty.SetGradientOpacity(funcOpacityGradient)
		volumeProperty.ShadeOff()
		volumeProperty.SetInterpolationTypeToLinear()
		#~ volumeProperty.setScalarOpacityUnitDistance( 4.5)
		#~ volumeProperty.SetShade(0,0)
		#volumeProperty.SetAmbient(0,0)
		#volumeProperty.SetDiffuse(1,2)
		#volumeProperty.SetSpecular(0,0)
		#volumeProperty.SetSpecularPower(0,20)
			 
		volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()#vtk.vtkFixedPointVolumeRayCastMapper()
		volumeMapper.SetInputConnection(dataImporter.GetOutputPort())
		volumeMapper.SetBlendModeToComposite();
		
	   
		volumeMapper.UseJitteringOff()
		
		
		dodecahedron = MakeDodecahedron()

		# Visualize
		mapper = vtk.vtkPolyDataMapper()
		mapper.SetInputData(dodecahedron.GetPolyData())

		actor = vtk.vtkActor()
		actor.SetMapper(mapper)
		
		# The volume holds the mapper and the property and
		# can be used to position/orient the volume.
		volume = vtk.vtkVolume()
		volume.SetMapper(volumeMapper)
		volume.SetProperty(volumeProperty)
		#~ volume.GetProperty().setScalarOpacityUnitDistance(0, 4.5);
		
		#~ ren1.AddActor(actor)
		ren1.AddVolume(volume)
		ren1.SetBackground(0.18,0.18,0.18)#colors.GetColor3d("Wheat")
		ren1.GetActiveCamera().Azimuth(45)
		ren1.GetActiveCamera().Elevation(30)
		ren1.ResetCameraClippingRange()
		ren1.ResetCamera()

		renWin.SetSize(600, 600)
		#renWin.Render()
		iren.Initialize()
		iren.Start()
		'''
		
			# ~ def setVtkDataImporter(self):
		# ~ c=self.listaMRC[self.currentItem]
		# ~ x=self.listaNumpy[self.currentItem].ravel()	
			
		# ~ self.dataImporter = vtk.vtkImageImport()
		# ~ #data_string = x.tostring()
		# ~ #dataImporter.CopyImportVoidPointer(data_string, len(data_string))
		# ~ self.dataImporter.SetImportVoidPointer(x)
		# ~ self.dataImporter.SetDataScalarTypeToDouble()
		# ~ self.dataImporter.SetNumberOfScalarComponents(1)
		# ~ self.dataImporter.SetDataExtent(0, c.Nx-1, 0, c.Ny-1, 0, c.Nz-1)
		# ~ self.dataImporter.SetWholeExtent(0, c.Nx-1, 0, c.Ny-1,0, c.Nz-1)
		
	# ~ def setOpacityTransferFunction(self):
		# ~ x=self.listaNumpy[self.currentItem]
		# ~ self.opacityTransferFunction = vtk.vtkPiecewiseFunction()
		# ~ self.opacityTransferFunction.AddPoint(np.min(x)+0.02*(np.max(x)-np.min(x)), 0.0)
		# ~ self.opacityTransferFunction.AddPoint(0.5*np.max(x), 0.4)
		# ~ self.opacityTransferFunction.AddPoint(1*np.max(x), 0.9)
		
	# ~ def setColorTransferFunction(self):
		# ~ x=self.listaNumpy[self.currentItem]
		# ~ self.funcOpacityGradient = vtk.vtkPiecewiseFunction()

		# ~ self.funcOpacityGradient.AddPoint(1,   0.0)
		# ~ self.funcOpacityGradient.AddPoint(5,   0.1)
		# ~ self.funcOpacityGradient.AddPoint(100,   1.0)

		# ~ # Create transfer mapping scalar value to color.
		# ~ self.colorTransferFunction = vtk.vtkColorTransferFunction()
		# ~ self.colorTransferFunction.AddRGBPoint(0.0, 47/255,0/255, 135/255)
		# ~ self.colorTransferFunction.AddRGBPoint(0.14*np.max(x), 98/255,0/255, 164/255)
		# ~ self.colorTransferFunction.AddRGBPoint(0.29*np.max(x), 146/255,0/255, 166/255)
		# ~ self.colorTransferFunction.AddRGBPoint(0.43*np.max(x), 186/255,47/255, 138/255)
		# ~ self.colorTransferFunction.AddRGBPoint(0.57*np.max(x), 216/255,91/255, 105/255)
		# ~ self.colorTransferFunction.AddRGBPoint(0.71*np.max(x), 238/255,137/255, 73/255)
		# ~ self.colorTransferFunction.AddRGBPoint(0.86*np.max(x), 246/255,189/255, 39/255)
		# ~ self.colorTransferFunction.AddRGBPoint(1*np.max(x), 228/255,250/255, 21/255)
		
	# ~ def setVolumeProperty(self):
		# ~ self.volumeProperty = vtk.vtkVolumeProperty()
		# ~ self.volumeProperty.SetColor(self.colorTransferFunction)
		# ~ self.volumeProperty.SetScalarOpacity(self.opacityTransferFunction)
		# ~ #~ volumeProperty.SetGradientOpacity(self.funcOpacityGradient)
		# ~ self.volumeProperty.ShadeOff()
		# ~ self.volumeProperty.SetInterpolationTypeToLinear()
		
	# ~ def setVolumeMapper(self):
		# ~ volumeMapper = vtk.vtkOpenGLGPUVolumeRayCastMapper()#vtk.vtkFixedPointVolumeRayCastMapper()
		# ~ volumeMapper.SetInputConnection(self.dataImporter.GetOutputPort())
		# ~ volumeMapper.SetBlendModeToComposite();	   
		# ~ volumeMapper.UseJitteringOff()
		
		# ~ self.volume = vtk.vtkVolume()
		# ~ self.volume.SetMapper(volumeMapper)
		# ~ self.volume.SetProperty(self.volumeProperty)
	
	# ~ def setRender(self):
		# ~ self.ren1.AddVolume(self.volume)
		# ~ self.ren1.SetBackground(0.18,0.18,0.18)#colors.GetColor3d("Wheat")
		# ~ self.ren1.GetActiveCamera().Azimuth(45)
		# ~ self.ren1.GetActiveCamera().Elevation(30)
		# ~ self.ren1.ResetCameraClippingRange()
		# ~ self.ren1.ResetCamera()

		# ~ #renWin.SetSize(600, 600)
		# ~ #renWin.Render()
		
	# ~ def view_act3(self):
		# ~ if len(self.listaMRC)==0:
			# ~ print("0 elements in list")
			# ~ self.resetRender()
			# ~ return
		
		# ~ self.resetRender()
		# ~ self.setSliderMinMax()
		# ~ self.plot()
		# ~ self.plotHistogram()
		# ~ self.setVtkDataImporter()
		# ~ self.setOpacityTransferFunction()
		# ~ self.setColorTransferFunction()
		# ~ self.setVolumeProperty()
		# ~ self.setVolumeMapper()
		# ~ self.setRender()
		# ~ self.vtk_widget.GetRenderWindow().Render()
		
