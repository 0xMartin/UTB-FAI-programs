# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 13:29:24 2020

@author: Krcma
"""
import sys, random, string, numpy, math, unicodedata
from PyQt5 import QtWidgets, uic, QtCore, QtGui
    
rep = [
       ["1", "XONEX"],
       ["2", "XTWOX"],
       ["3", "XTHREEX"],
       ["4", "XFOURX"],
       ["5", "XFIVEX"],
       ["6", "XSIXX"],
       ["7", "XSEVENX"],
       ["8", "XEIGHTX"],
       ["9", "XNINEX"],
       ["0", "XZEROX"]
]


"""
    podle velikosti navrati prislusnou znakovou sadu
    size - 5->ADFGX, jina hodnota->ADFGVX
"""
def charSET(size):
    return ['A', 'D', 'F', 'G', 'X'] if size == 5 else ['A', 'D', 'F', 'G', 'V', 'X']


"""
     filtrace vstupnich dat
     alphabetOnly - True(jen znaky A-Z)
"""
def inputFilter(txt, alphabetOnly):  
    txt = str(unicodedata.normalize('NFKD', txt).encode('ASCII', 'ignore'))[2:-1]
    out = ""
    if alphabetOnly:
        for c in txt.upper():
            if c in string.ascii_uppercase:
                out += c 
    else:
        for c in txt.upper():
            if c in string.ascii_uppercase or c in string.digits or c == ' ':
                out += c                  
    return out


"""
    nahodne vygeneruje tabluku
    b - True(ADFGX) / False(ADFGVX)
    unusedChar - nepouzivany znak
"""
def generateRandomMatrix(b, unusedChar):
    size = 5 if b else 6
    
    chars = list(string.ascii_uppercase)
    if not b:
        chars += list(string.digits)
    if b:
        chars.remove(unusedChar)
    random.shuffle(chars)
    
    matrix = numpy.full((size, size), '')
    for c, x in zip(chars, numpy.nditer(matrix, op_flags=['readwrite'])):
        x[...] = c
    
    return matrix.tolist()


"""
    Transpozice v tabulce s klicem
    (sifrovani / desifrovani)
    txt - vstupni text
    key - klic
    enc - True (sifrovani), False (Desifrovani)
"""
def transposition(txt, key, enc): 
    if len(key) == 0:
        return ""  
    if not enc:
        arr = txt.split()
        r = int(len(txt.replace(" ", ""))/len(arr))
        txt = ''.join(map(lambda c: c + (r + 1 - len(c)) * " ", arr))
        
    # vypocet indexu sloupcu po klicove transformaci matice
    sortedKey = sorted(key)
    defaultKey = list(key)
    if not enc:
        sortedKey, defaultKey = defaultKey, sortedKey
    colI = []
    for c in defaultKey:
        colI.append(sortedKey.index(c))
        sortedKey[colI[-1]] = chr(1)
        
    # vlozi znaky textu do matice
    cols = len(key)
    rows = math.ceil(len(txt)/cols)
    matrix = numpy.full((cols, rows), '')  
    if enc:
        # sifrovani
        # naplneni matice textem
        for i, c in enumerate(txt):
            matrix[colI[i % cols]][int(i / cols)] = c 
    else:
        # desifrovani
        # naplneni matice textem
        for i, c in enumerate(txt):  
            matrix[colI[int(i / rows)]][i % rows] = c
                      
    # znaky z transformovane matice vlozi do retezce (znaky zapise po sloupcich)
    if not enc:
        matrix = matrix.transpose()      
                
    return (' ' if enc else '').join(map(lambda col: ''.join(col).replace(' ', ''), matrix))


"""
      zasifruje vstupni text
      txt - text
      matrix - sifrovaci matice
      key - transformacni klic
"""
def encrypt(txt, matrix, key):
    if len(txt) == 0 or len(txt) * 2 < len(key):
        return ""
    out = ""  
    size = len(matrix)
    cS = charSET(size)
    for c in txt:
        for i in range(0, size):
            if c in matrix[i]:
                out += cS[matrix[i].index(c)] + cS[i]
                break
                
    return transposition(out, key, True)


"""
      desifruje vstupni text
      txt - text
      matrix - sifrovaci matice
      key - transformacni klic
"""
def decrypt(txt, matrix, key):
    if len(txt) == 0 or len(txt) < len(key):
        return ""
    
    # kontrola znaku
    for c in txt:
        if c not in charSET(len(matrix)) and c != ' ':
            return ""
    
    txt = transposition(txt, key, False)
        
    out = "" 
    row = chr(0)
    cS = charSET(len(matrix))
    for c in txt:
        if row != chr(0):
            out += matrix[cS.index(c)][cS.index(row)]
            row = chr(0)
            continue
        row = c
    return out


# GUI aplikace
class App(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(App, self).__init__()
        uic.loadUi('mainwindow.ui', self)
        self.show()

        self.rbADFGVX = self.findChild(QtWidgets.QRadioButton, 'rbADFGVX') 
        self.rbADFGX = self.findChild(QtWidgets.QRadioButton, 'rbADFGX')
        self.rbADFGX.toggled.connect(lambda: self.changeMode(self.rbADFGX.isChecked()))
        self.group = QtWidgets.QButtonGroup()
        self.group.addButton(self.rbADFGVX)
        self.group.addButton(self.rbADFGX)
        
        self.tableWidget = self.findChild(QtWidgets.QTableWidget, 'tableWidget')
        self.tableWidget.setUpdatesEnabled(True)
        self.tableWidget.itemChanged.connect(self.checkMatrix)
        self.lvRemainingChars = self.findChild (QtWidgets.QListView, 'lvRemainingChars')
        self.modelRemainingChars = QtGui.QStandardItemModel()
        self.lvRemainingChars.setModel(self.modelRemainingChars)
        
        self.pushButtonEnc = self.findChild (QtWidgets.QPushButton, 'pushButtonEnc')
        self.pushButtonEnc.clicked.connect(self.encrypt)
        self.pushButtonDec = self.findChild (QtWidgets.QPushButton, 'pushButtonDec')
        self.pushButtonDec.clicked.connect(self.decrypt)
        self.pushButtonRnd = self.findChild (QtWidgets.QPushButton, 'pushButtonRnd')
        self.pushButtonRnd.clicked.connect(lambda: self.generateMatrix(self.rbADFGX.isChecked()))
        
        self.textEditIn = self.findChild(QtWidgets.QTextEdit, 'textEditIn')
        self.textEditIn.textChanged.connect(self.checkKey)
        self.textEditOut = self.findChild(QtWidgets.QTextEdit, 'textEditOut')
        self.textEditKey = self.findChild(QtWidgets.QTextEdit, 'textEditKey')
        self.textEditKey.textChanged.connect(self.checkKey)
        
        self.comboBoxLan = self.findChild(QtWidgets.QComboBox, 'comboBoxLan')
        self.comboBoxLan.currentIndexChanged.connect(lambda: self.changeLang(self.comboBoxLan.currentText()));
         
        self.changeMode(self.rbADFGX.isChecked())
        self.changeLang("CZ") 
        self.checkKey()

        
    def encrypt(self):
        txt = inputFilter(self.textEditIn.toPlainText(), False)
        key = inputFilter(self.textEditKey.toPlainText(), True)
        
        txt = txt.replace(" ", "XMEZERAX")
        # nahrazovani cislic (jen pro ADFGX) + nahrazeni nepozivaneho znaku
        if self.rbADFGX.isChecked():
            for r in rep:
                txt = txt.replace(r[0], r[1])   
            txt = txt.replace(self.UNUSED_CHAR, self.REPLACE_CHAR)   
        
        # sifrovani
        txt = encrypt(txt, self.matrix, key)
        
        self.textEditOut.setText(txt)
    
    
    def decrypt(self):
        txt = inputFilter(self.textEditIn.toPlainText(), False)
        key = inputFilter(self.textEditKey.toPlainText(), True)
    
        # desifrovani
        txt = decrypt(txt, self.matrix, key)
        
        txt = txt.replace("XMEZERAX", " ")
        # nahrazovani cislic (jen pro ADFGX)
        if self.rbADFGX.isChecked():
            for r in rep:
                txt = txt.replace(r[1], r[0])
        
        self.textEditOut.setText(txt)
      
    
    def changeLang(self, lang):
        # zmena znaku dle zvoleneho jazyka
        if lang == "CZ":
            self.UNUSED_CHAR = 'W'
            self.REPLACE_CHAR = 'V'
        else:
            self.UNUSED_CHAR = 'J'
            self.REPLACE_CHAR = 'I'  
        self.checkMatrix()

    def changeMode(self, ADFGX):
        self.generateMatrix(ADFGX)  
        # zmana pristupnosti ke komponentum
        self.comboBoxLan.setEnabled(ADFGX)


    def generateMatrix(self, ADFGX):
        # zmena tabulky
        self.matrix = generateRandomMatrix(ADFGX, self.UNUSED_CHAR if ADFGX else '#')
        # matici zapise do tabulky
        self.tableWidget.setColumnCount(len(self.matrix))
        self.tableWidget.setRowCount(len(self.matrix))
        self.tableWidget.clear()
        for i, row in enumerate(self.matrix):
            for j, cell in enumerate(row):
                self.tableWidget.setItem(j, i, QtWidgets.QTableWidgetItem(cell))
        self.tableWidget.setHorizontalHeaderLabels(charSET(len(self.matrix)))
        self.tableWidget.setVerticalHeaderLabels(charSET(len(self.matrix))) 
        self.tableWidget.show()
      
     
    # zobrazeni zbyvajicich znaku
    def remainingChars(self):
        # vsechny znaky nachazejici se v matici
        allChars = ''.join(map(lambda c: ''.join(c), self.matrix))
        if self.rbADFGX.isChecked():
            allChars += self.UNUSED_CHAR
        # vsechny znaky mozne pro dany mod sifrovani/desifrovani (ADFGX nebo ADFGVX)
        rC = list(string.ascii_uppercase if self.rbADFGX.isChecked() else string.ascii_uppercase + string.digits)
        # vypise jen ty ktere jeste nejsou zapsane v matici
        for c in allChars:
            if c in rC:
                rC.remove(c)
        self.modelRemainingChars.clear()
        for c in rC:
            self.modelRemainingChars.appendRow(QtGui.QStandardItem(c))
        return len(rC) == 0
    
    
    # kontrola sifrovaci matice
    def checkMatrix(self):
        for r in range(self.tableWidget.rowCount()):
            for c in range(self.tableWidget.columnCount()):
                item = self.tableWidget.item(r, c)
                if item != None:
                    if not self.isMatrixCellValid(item):         
                        item.setBackground(QtGui.QColor(220, 100, 100))   
                    else:
                        item.setBackground(QtGui.QColor(255, 255, 255))  
                
        # povoleni/zakazani sifrovani/desifrovani
        valid = self.remainingChars()
        self.pushButtonEnc.setEnabled(valid)
        self.pushButtonDec.setEnabled(valid)
    
    
    # kontrola platnosti hodnoty v bunce
    def isMatrixCellValid(self, item):
        if item != None:
            valid = True
            # omezeni pro hodnoty v bunkach matice
            if len(item.text()) != 0:
                item.setText(item.text().upper()[0])
                c = item.text()[0]
                if self.rbADFGX.isChecked():
                    if c not in string.ascii_uppercase:
                        valid = False  
                else:
                    if c not in string.ascii_uppercase and c not in string.digits:
                        valid = False  
                
                m = self.tableWidget.findItems(item.text(), QtCore.Qt.MatchExactly)
                valid &= len(m) == 1
            else:
                valid = False     
            # zmena provedena v tabulce bude provedena i v matici
            self.matrix[item.column()][item.row()] = item.text()
            return valid
        return False
    
    
    # kontrola klice
    def checkKey(self):
        txt = inputFilter(self.textEditIn.toPlainText(), False)
        key = inputFilter(self.textEditKey.toPlainText(), True)
        self.pushButtonEnc.setEnabled(len(txt) * 2 >= len(key) and len(key) != 0 and len(txt) != 0)
        self.pushButtonDec.setEnabled(len(txt) >= len(key) and len(key) != 0 and len(txt) != 0)



if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = App() 
    app.exec_()
    