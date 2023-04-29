# -*- coding: utf-8 -*-
"""
Created on Wed Oct 14 00:17:45 2020

@author: Krcma
"""

from textwrap import wrap
import sys, random, math
from PyQt5 import QtWidgets, uic

def millerRabin(n, k):
    if n == 2:
        return True
    if n % 2 == 0:
        return False

    r, s = 0, n - 1
    while s % 2 == 0:
        r += 1
        s //= 2
    for i in range(k):
        a = random.randrange(2, n - 1)
        x = pow(a, s, n)
        if x == 1 or x == n - 1:
            continue
        for j in range(r - 1):
            x = pow(x, 2, n)
            if x == n - 1:
                break
        else:
            return False
    return True


def generateLargePrime(_min, _max):
    while True:
        n = random.randrange(_min, _max)
        if millerRabin(n, 10000):
            return n
    return None


def egcd(a, b):
    if a == 0:
        return b, 0, 1
    else:
        g, y, x = egcd(b % a, a)
        return g, x - (b // a) * y, y


def modinv(a, m):
    g, x, y = egcd(a, m)
    if g == 1: 
        return x % m


"""
    Generate public in private key 
"""
def generateKeys():
    p, q = generateLargePrime(1e18, 1e19), generateLargePrime(1e18, 1e19)
    n = p * q

    phi = (p - 1) * (q - 1)
    while True:
        e = random.randrange(2, phi)
        if math.gcd(e, phi) == 1:
            break
        
    return [[n, e], [n, modinv(e, phi)]]

"""
    [RSA]
    data - input data (enc=True: string consist ASCII, enc=False: string consist from from number separate by whitespace)
    key - RSA key
    enc - True: encrypting, False: decrypting
"""
def RSA(data, key, enc):
    if enc:
        # convert ASCII string to bit string
        data = ''.join((9 - len(w)) * '0' + w for w in list(bin(ord(c))[2:] for c in data))
        # long bit string split on block (72b or less per each) and then each block encrypt independently
        return list(pow(int(block, 2), key[1], key[0]) for block in wrap(data, 72))
    else:
        # each block decrypt independently (number = block)
        data = list(bin(pow(int(num), key[1], key[0]))[2:] for num in data.split(" "))
        # from all block create long bit string, if length of block is lower than 9 then add some 0 bits
        return ''.join(chr(int(char, 2)) for block in data for char in wrap((9 - len(block) % 9) * '0' + block, 9))

# GUI
class App(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(App, self).__init__()
        uic.loadUi('mainwindow.ui', self)
        self.show()
        
        self.textEditIn = self.findChild(QtWidgets.QTextEdit, 'textEditIn')
        self.textEditOut = self.findChild(QtWidgets.QTextEdit, 'textEditOut')
        
        self.lineEditPrvN = self.findChild(QtWidgets.QLineEdit, 'lineEditPrvN')
        self.lineEditPrvD = self.findChild(QtWidgets.QLineEdit, 'lineEditPrvD')
        
        self.lineEditPubN = self.findChild(QtWidgets.QLineEdit, 'lineEditPubN')
        self.lineEditPubE = self.findChild(QtWidgets.QLineEdit, 'lineEditPubE')
        
        self.lineEditPrvN.textChanged.connect(lambda: self.checkKey(self.lineEditPrvN, True))
        self.lineEditPrvD.textChanged.connect(lambda: self.checkKey(self.lineEditPrvD, True))
        self.lineEditPubN.textChanged.connect(lambda: self.checkKey(self.lineEditPubN, False))
        self.lineEditPubE.textChanged.connect(lambda: self.checkKey(self.lineEditPubE, False))

        self.pushButtonEnc = self.findChild (QtWidgets.QPushButton, 'pushButtonEnc')
        self.pushButtonEnc.clicked.connect(self.encrypt)
        self.pushButtonDec = self.findChild (QtWidgets.QPushButton, 'pushButtonDec')
        self.pushButtonDec.clicked.connect(self.decrypt)
        self.pushButtonGK = self.findChild (QtWidgets.QPushButton, 'pushButtonGK')
        self.pushButtonGK.clicked.connect(self.generateKey)
        self.pushButtonClr = self.findChild (QtWidgets.QPushButton, 'pushButtonClr')
        self.pushButtonClr.clicked.connect(self.clear)
        
        self.generateKey()

    def encrypt(self):
        buffer = self.textEditIn.toPlainText()
        if len(buffer) == 0: return
        key = [int(self.lineEditPubN.text()), int(self.lineEditPubE.text())]
        buffer = RSA(buffer, key, True)
        self.textEditOut.setText(' '.join(str(num) for num in buffer))
    
    def decrypt(self):
        buffer = self.textEditIn.toPlainText()
        if len(buffer) == 0: return
        key = [int(self.lineEditPrvN.text()), int(self.lineEditPrvD.text())]
        self.textEditOut.setText(RSA(buffer, key, False))
    
    def generateKey(self):
        keys = generateKeys()
        self.lineEditPrvN.setText(str(keys[1][0]))
        self.lineEditPrvD.setText(str(keys[1][1]))
        self.lineEditPubN.setText(str(keys[0][0]))
        self.lineEditPubE.setText(str(keys[0][1]))
        
    def clear(self):
        self.lineEditPrvN.setText("")
        self.lineEditPrvD.setText("")
        self.lineEditPubN.setText("")
        self.lineEditPubE.setText("")
        self.textEditIn.setText("")
        self.textEditOut.setText("")
        
    def checkKey(self, lineEdit, private):
         b = len(lineEdit.text()) != 0 and lineEdit.text().isdigit()
         (self.pushButtonDec if private else self.pushButtonEnc).setEnabled(b)
         if b:
             lineEdit.setStyleSheet("background-color: rgb(255, 255, 255);")
         else:
             lineEdit.setStyleSheet("background-color: rgb(255, 70, 70);")   


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = App() 
    app.exec_()
        