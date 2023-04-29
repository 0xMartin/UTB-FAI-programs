# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 23:57:39 2020

@author: Martin Krcma
"""

import sys
import math
import unicodedata
from PyQt5 import QtWidgets, uic
from textwrap import wrap

class Rep:
    def __init__(self, a, b):
        self.A = a
        self.B = b

# list specialnich (nesifrovatelnych) znaku a k nim 
# jejich prislusny retezec (sifrovatelny) 
rep = {
    Rep(" ", "XMEZERAX"),   Rep("0", "XZEROX"),     Rep("1", "XONEX"),
    Rep("2", "XTWOX"),      Rep("3", "XTHREEX"),    Rep("4", "XFOURX"),
    Rep("5", "XFIVEX"),     Rep("6", "XSIXX"),      Rep("7", "XSEVENX"),
    Rep("8", "XEIGHTX"),    Rep("9", "XNINEX")
}

# nalezne inverzni prvek
# a - cislo pro ktere hledame inverzi
def inverse(a):
    for i in range(0, 26):
        if a * i % 26 == 1:
            return i
    return 0

# vstupni text zasifruje afinni sifrou
# txt - vstupni text
# a - klic A, gcd(A, 26) = 1
# b - klic B
def encrypt(txt, a, b):
    return ''.join(map(lambda c: chr((a * (ord(c) - 65) + b) % 26 + 65), txt))

# vstupni text zasifrovany afinni sifrou desifruje
# txt - vstupni text
# a - klic A, gcd(A, 26) = 1
# b - klic B
def decrypt(txt, a, b):
    return ''.join(map(lambda c: chr((ord(c) - 65 - b) * inverse(a) % 26 + 65), txt))

# GUI aplikace
class App(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(App, self).__init__()
        uic.loadUi('mainwindow.ui', self)
        self.show()
        self.buttonEncrypt = self.findChild (QtWidgets.QPushButton, 'buttonEncrypt')
        self.buttonEncrypt.clicked.connect(self.encrypt)
        self.buttonDecrypt = self.findChild(QtWidgets.QPushButton, 'buttonDecrypt')
        self.buttonDecrypt.clicked.connect(self.decrypt)
        self.spinBoxA = self.findChild(QtWidgets.QSpinBox, 'spinBoxA')
        self.spinBoxB = self.findChild(QtWidgets.QSpinBox, 'spinBoxB')
        self.textOT = self.findChild(QtWidgets.QTextEdit, 'textOT')
        self.textET = self.findChild(QtWidgets.QTextEdit, 'textET')
        self.textOut = self.findChild(QtWidgets.QTextEdit, 'textOut')
  
    # filtrace vstupniho textu
    # prevede vsechny znaky ne velke, odstrani diakritiku
    def inputTextFilter(self, txt):
        txt = txt.upper()
        txt = str(unicodedata.normalize('NFKD', txt).encode('ASCII', 'ignore'))[2:-1]
        out = ""
        for c in txt:
            if (c >= 'A' and c <= 'Z') or c == ' ' or (c >= '0' and c <= '9'):
                out += c            
        return out  
  
    # kontrola platnosti klice A
    # pokud je klic neplatny opozorni na chybu zvyrazneni prislusneho line editu
    # return: True = klic je platny
    def checkA(self, a):
        if math.gcd(a, 26) != 1:
            self.spinBoxA.setStyleSheet("color: white; background-color: red")
            return False
        else:
            self.spinBoxA.setStyleSheet("color: white; background-color: rgb(70, 70, 80)")
            return True

    #vypis abecedy a sifrovane abecedy
    def printAlphabet(self, a):  
        d1 = ""
        d2 = ""
        for c in range(ord('A'), ord('Z') + 1):
            d1 += chr(c) + " "
            d2 += encrypt(chr(c), a, int(self.spinBoxB.value())) + " "
        print(d1)
        print(d2)

    # zasifruje vstupni text
    def encrypt(self):
        a = int(self.spinBoxA.value())      
        if self.checkA(a): 
            txt = self.inputTextFilter(self.textOT.toPlainText())        
            print(txt)
        
            #nahradi mezery a cisly prislusnymi retezci
            for r in rep:
                txt = txt.replace(r.A, r.B)
                 
            self.printAlphabet(a)
            
            txt = encrypt(txt, a, int(self.spinBoxB.value()))
            
            self.textOut.setText(' '.join(wrap(txt, 5)))
        
    # desifruje vstupnit text
    def decrypt(self):
        a = int(self.spinBoxA.value())     
        if self.checkA(a):   
            txt = self.inputTextFilter(self.textET.toPlainText())
            txt = txt.replace(' ', '')
            print(txt)
        
            self.printAlphabet(a)
        
            txt = decrypt(txt, a, int(self.spinBoxB.value()))
        
            #retezce nahradi jejich prislusnymi znaky (0-9 + mezera)
            for r in rep:
                txt = txt.replace(r.B, r.A)
            
            self.textOut.setText(txt)


# main
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    app.exec_()