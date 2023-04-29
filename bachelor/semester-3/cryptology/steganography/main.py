#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 23:07:02 2020

@author: root
"""

import sys
from textwrap import wrap
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt

# GUI
class App(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(App, self).__init__()
        uic.loadUi('mainwindow.ui', self)
        self.show()
        
        self.pushButtonSave = self.findChild(QtWidgets.QPushButton, 'pushButtonSave')
        self.pushButtonSave.clicked.connect(self.save)
        self.pushButtonLoad = self.findChild(QtWidgets.QPushButton, 'pushButtonLoad')
        self.pushButtonLoad.clicked.connect(self.load)
        self.pushButtonRD = self.findChild(QtWidgets.QPushButton, 'pushButtonRD')
        self.pushButtonRD.clicked.connect(self.readData)
        self.pushButtonWD = self.findChild(QtWidgets.QPushButton, 'pushButtonWD')
        self.pushButtonWD.clicked.connect(self.writeData)
        
        self.textEditDI = self.findChild(QtWidgets.QTextEdit, 'textEditDI')
        self.textEditDO = self.findChild(QtWidgets.QTextEdit, 'textEditDO')
        self.labelImage = self.findChild(QtWidgets.QLabel, 'labelImage') 
    
    def save(self):
        if self.pixMap != None:
            file = QFileDialog.getSaveFileName(self, 'Save image', '', '*.png')
            self.pixMap.save(file[0], "PNG")
    
    def writeData(self):
        self.writeDataToImg(self.textEditDI.toPlainText())
        
    def readData(self):
        self.readDataFromImg()
    
    def load(self):
        dlg = QFileDialog()
        dlg.setFileMode(QFileDialog.AnyFile)
        if dlg.exec_():
            self.pixMap = QPixmap(dlg.selectedFiles()[0])
            self.displayPixMap(self.pixMap)
            
    def displayPixMap(self, pixmap):
        w = self.labelImage.width()
        h = self.labelImage.height()
        self.labelImage.setPixmap(pixmap.scaled(w, h, Qt.KeepAspectRatio))

    def writeDataToImg(self, data):
        data += '\0'
        data = ''.join('0' * (9 - len(bits)) + bits for bits in list(bin(ord(c))[2:] for c in data))
        if self.pixMap != None:
            img = self.pixMap.toImage()
            for y in range(img.height()):
                for x in range(img.width()):
                    if x + y * img.height() >= len(data): break
                    color = img.pixelColor(x, y)
                    color.setBlue(int(bin(color.blue())[:-1] + data[x + y * img.height()], 2)) 
                    img.setPixelColor(x, y, color)      
            self.pixMap = QPixmap.fromImage(img)
            self.displayPixMap(self.pixMap)
       
    def readDataFromImg(self):
        if self.pixMap != None:
            data = ""
            img = self.pixMap.toImage()
            zeroCnt = 0
            for y in range(img.height()):
                for x in range(img.width()):
                    bit = bin(img.pixelColor(x, y).blue())[-1]
                    data += bit
                    if bit == '0': 
                        zeroCnt += 1
                        if zeroCnt == 9: break
                    else:
                        zeroCnt = 0    
                if zeroCnt == 9: break
            txt = ''.join(chr(int(bits, 2)) for bits in wrap(data, 9))    
            self.textEditDO.setText(txt)           
                
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = App() 
    app.exec_()