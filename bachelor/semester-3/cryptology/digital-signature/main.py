# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 15:47:37 2020

@author: Krcma
"""

import sys, random, math, os, time, hashlib
from os import path
from zipfile import ZipFile
from textwrap import wrap
from PyQt5 import QtWidgets, uic
from PyQt5.QtWidgets import QFileDialog

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
        data = ''.join((9 - len(w)) * '0' + w for w in list(bin(c)[2:] for c in data))
        # long bit string split on block (72b or less per each) and then each block encrypt independently
        return list(pow(int(block, 2), key[1], key[0]) for block in wrap(data, 72))
    else:
        # each block decrypt independently (number = block)
        data = list(bin(pow(int(num), key[1], key[0]))[2:] for num in data.split(" "))
        # from all block create long bit string, if length of block is lower than 9 then add some 0 bits
        out = list(int(char, 2) for block in data for char in wrap((9 - len(block) % 9) * '0' + block, 9))
        for b in out:
            if b not in range(0, 256):
                return None
        return bytearray(out)

# GUI
class App(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(App, self).__init__()
        uic.loadUi('mainwindow.ui', self)
        self.show()
        
        self.pushButtonSave.clicked.connect(self.saveFile)
        self.pushButtonOpen.clicked.connect(lambda: self.openFile(QFileDialog.getOpenFileName(self, 'Open file', ''), True))
        self.pushButtonGenK.clicked.connect(self.generateKey)
        self.pushButtonExportK.clicked.connect(self.exportKey)      
        self.pushButtonImportK.clicked.connect(self.importKey)
        
        self.pushButtonCreateS.clicked.connect(self.createSignature)
        self.pushButtonCheckS.clicked.connect(self.checkDocument)
         
        self.reset()
        
    def saveFile(self):
        file = QFileDialog.getSaveFileName(self, 'Save file', '', "ZIP file (*.zip)")
                
        if self.fileSign == None:
            return
        if not path.exists(self.fileSign[0]):
            return
        
        zipF = ZipFile(file[0], 'w')
        zipF.write(self.labelPath.text(), path.basename(self.labelPath.text()))
        zipF.write(self.fileSign[0], path.basename(self.fileSign[0]))
        zipF.close()
        
    def openFile(self, file, reset): 
        if file[0] == '':
            return
        if reset:
            self.reset()
        
        file_path = path.basename(file[0]).split(".")
        self.labelName.setText(file_path[0])
        self.labelPath.setText(file[0])
        if len(file_path) > 1:
            self.labelType.setText(file_path[1])    
        size, i = os.path.getsize(file[0]), 0
        while size > 1000: 
            size /= 1000
            i += 1
        self.labelSize.setText(str(size) + " " + " KMGT"[i] + "B")
        self.labelLastModified.setText(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(os.path.getmtime(file[0]))))     
        self.openedFile = open(file[0], "rb")  
        self.checkKey(True)
    
    def generateKey(self):
        keys = generateKeys()
        self.PrvN = keys[1][0]
        self.PrvD = keys[1][1]
        self.PubN = keys[0][0]
        self.PubE = keys[0][1] 
        self.checkKey(True)
        self.checkKey(False)
        
    def exportKey(self):
        file = QFileDialog.getSaveFileName(self, 'Export key', '')
        
        f = open(file[0] + ".pub", "w")
        f.write("RSA " + str(self.PubN) + " " + str(self.PubE))
        f.close()
        
        f = open(file[0] + ".priv", "w")
        f.write("RSA " + str(self.PrvN) + " " + str(self.PrvD))
        f.close()
    
    def importKey(self):
        file = QFileDialog.getOpenFileName(self, 'Import key', '', "Public key (*.pub);;Private key (*.priv)")  
        if file[0] == '': return
        f = open(file[0], "r")
        s = f.read().split(" ")
        if file[0].endswith(".pub"):
            self.PubN = int(s[1])
            self.PubE = int(s[2])
            self.checkKey(False)
        else:
            self.PrvN = int(s[1])
            self.PrvD = int(s[2])
            self.checkKey(True)
        f.close()
    
    def createSignature(self):
        if self.openedFile == None:
            return
        
        h = hashlib.sha512(self.openedFile.read()).digest()       
        sign = ' '.join(str(num) for num in RSA(h, [self.PrvN, self.PrvD], True))   
        self.labelES.setText("Created ✓")
        self.labelES.setStyleSheet("color: rgb(50, 205, 50);")
        
        self.fileSign = QFileDialog.getSaveFileName(self, 'Save signature', '', "Signature (*.sign)")
        f = open(self.fileSign[0], "w")
        f.write("RSA_SHA512 " + sign)
        f.close()
        
        self.pushButtonSave.setEnabled(True)
    
    def checkDocument(self):
        file = QFileDialog.getOpenFileName(self, 'Open zip file with signature', '', "ZIP file (*.zip)")
        if file[0] == '': return
        dirName = path.basename(file[0]).split(".")[0]
        with ZipFile(file[0], 'r') as zip_ref:
            zip_files = zip_ref.namelist()
            zip_ref.extractall(dirName)
        
        if zip_files[0].split(".")[-1] == 'sign':
            zip_files[0], zip_files[1] = zip_files[1], zip_files[0]
        
        self.openFile([dirName + "/" + zip_files[0], ''], False)    
        h = hashlib.sha512(self.openedFile.read()).digest() 
        
        signature = open(dirName + "/" + zip_files[1], "r")
        sign = RSA(signature.read()[11:], [self.PubN, self.PubE], False)
     
        self.labelCheck.setText(
            "Everything is OK, with the document was not manipulated during sending ✓" 
            if h == sign else 
            "Problem, the file could be modified ✗"
            )
        self.labelCheck.setStyleSheet("color: rgb(50, 205, 50);" if h == sign else "color: rgb(205, 50, 50);")
    
    def checkKey(self, private):
         b = (self.PrvN != 0 or self.PrvD != 0) and private or (self.PubN != 0 or self.PubE != 0) and not private
         
         (self.pushButtonCreateS if private else self.pushButtonCheckS).setEnabled(
             b and not(private and self.openedFile == None))
         
         self.pushButtonExportK.setEnabled(b)
         
         (self.labelPrivKey if private else self.labelPubKey).setText(
             "Loaded ✓" if b else "Not exist ✗")
         
         (self.labelPrivKey if private else self.labelPubKey).setStyleSheet(
             "color: rgb(50, 205, 50);" if b else "color: rgb(205, 50, 50);")
        
    def reset(self):
        self.openedFile = None
        self.PubN = 0
        self.PubE = 0
        self.PrvN = 0
        self.PrvD = 0
        self.pushButtonSave.setEnabled(False)
        self.checkKey(True)
        self.checkKey(False)
        self.labelES.setText("Not exist✗")
        self.labelES.setStyleSheet("color: rgb(205, 50, 50);")
        
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = App() 
    app.exec_()