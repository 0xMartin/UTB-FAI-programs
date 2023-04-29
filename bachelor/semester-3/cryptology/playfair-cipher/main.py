# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 22:42:39 2020

@author: Martin
"""

import sys
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
    Rep(" ", "XEORQY"),     Rep("0", "XTBOZY"),     Rep("1", "XDOAQY"),
    Rep("2", "XFUAHY"),     Rep("3", "XDKGEY"),     Rep("4", "XUGCTY"),
    Rep("5", "XKSLDY"),     Rep("6", "XFEZBY"),     Rep("7", "XCICEY"),
    Rep("8", "XAPOHY"),     Rep("9", "XPMITY")
}

"""
     filtrace vstupnich dat
     alphabetOnly - True(jen znaky A-Z)
"""
def inputFilter(txt, alphabetOnly):  
    txt = str(unicodedata.normalize('NFKD', txt).encode('ASCII', 'ignore'))[2:-1]
    out = ""
    if alphabetOnly:
        for c in txt.upper():
            if c >= 'A' and c <= 'Z':
                out += c 
    else:
        for c in txt.upper():
            if (c >= 'A' and c <= 'Z') or (c >= '0' and c <= '9') or c == ' ':
                out += c                  
    return out

"""
    oddelovac digramu
    txt - znaky digramu
    c1 - prvni nahrazujici znak
    c2 - druhy nahrazujici znak
"""
def digramSplitter(txt, c1, c2):
    # v textu eliminuje ty digramy ktere maji dva stejne znaky
    out = ""
    i = 0
    while i < len(txt):
        if i + 1 >= len(txt):
            out += txt[i]
            break;
        elif txt[i] == txt[i + 1]:
            out += txt[i] + (c1 if txt[i] != c1 else c2) 
            i += 1
        else:
            out += txt[i : i + 2]
            i += 2        
            
    # doplneni na sudy pocet znaku
    if len(out) % 2 != 0:
        out += c1 if out[-1] != c1 else c2
    
    return out

"""
    sestaveni tabluky
    key - klic
"""
def buildTable(key, unused, replacer):
    # sestaveni sifrovaci tabulky
    table = []
    used = []
    # vlozeni znaku klice do tabulky
    for c in key.upper().replace(' ', ''):
        if c not in used:
            if c == unused:
                c = replacer
            used.append(c)
            table.append(c)
    # vlozeni zbyvajicich znaku abecedy
    for c in range(65, 91):
        if chr(c) not in used and chr(c) != unused:
            table.append(chr(c))
            if len(table) == 25:
                break
    return table


"""
    pozice znaku v tabulce
    table - tabulka 5x5
    char - hledany znak
"""
def locationOf(table, char):
    index = 0
    for el in table:
        if el == char:
             return [index % 5, int(index / 5)]   
        index += 1        
    return [-1, -1]

"""
    sifra playfair (sifrovani / desifrovani)
    txt - vstupni text
    key - klic sifry
    m - mode: True[sifrovani], False[desifrovani]
"""
def playfair(txt, table, m): 
    if len(txt) == 0:
        return ""
    # sifrovani/desifrovani            
    out = ""
    for pair in wrap(txt, 2):
        if len(pair) != 2:
            break
        if pair[0] == pair[1]:
            continue
        # nalezeni pozice pro oba znaky digramu
        p1 = locationOf(table, pair[0])
        p2 = locationOf(table, pair[1])  
        # offset pro radek a sloupec 
        # 0 a v pripade ze znaky lezi ve stejnem sloupci nebo radku pak je 1/-1
        # m = True -> 1 / m = False -> 0
        inCol = int(p1[0] == p2[0]) * (1 if m else -1)
        inRow = int(p1[1] == p2[1]) * (1 if m else -1)
        # pro oba znaky digramu najde jejich prislusne zasifrovane znaky
        c1 = table[(p2[0] + inRow) % 5 + (p1[1] + inCol) % 5 * 5]
        c2 = table[(p1[0] + inRow) % 5 + (p2[1] + inCol) % 5 * 5]
        out += (c2 + c1) if inRow else (c1 + c2)   
    return out


# GUI aplikace
class App(QtWidgets.QMainWindow):
    
    def __init__(self):
        super(App, self).__init__()
        uic.loadUi('mainwindow.ui', self)
        self.show()
        self.pushButtonEnc = self.findChild(QtWidgets.QPushButton, 'pushButtonEnc')
        self.pushButtonEnc.clicked.connect(self.encrypt)
        self.pushButtonDec = self.findChild(QtWidgets.QPushButton, 'pushButtonDec')
        self.pushButtonDec.clicked.connect(self.decrypt)
        self.textEditIn = self.findChild(QtWidgets.QTextEdit, 'textEditIn')
        self.textEditOut = self.findChild(QtWidgets.QTextEdit, 'textEditOut')
        self.lineEditKey = self.findChild(QtWidgets.QLineEdit, 'lineEditKey')
        self.lineEditKey.textChanged.connect(self.builTable)
        self.labelTable = self.findChild(QtWidgets.QLabel, 'labelTable')
        self.labelReplace = self.findChild(QtWidgets.QLabel, 'labelReplace')
        self.comboBoxLan = self.findChild(QtWidgets.QComboBox, 'comboBoxLan')
        self.comboBoxLan.currentTextChanged.connect(self.lanChanged)
        # nastaveni default jazyku
        self.lanChanged("CZ")

    # zmena jazyka
    def lanChanged(self, value):
        if value == 'CZ':
            # CZ jazyk: W nahrazeno za V
            self.UNUSED_CHAR = 'W'
            self.REPLACE_CHAR = 'V'
            # odelujici znaky
            self.C1 = 'X'
            self.C2 = 'Z'
        else:
            # CZ jazyk: J nahrazeno za I
            self.UNUSED_CHAR = 'J'
            self.REPLACE_CHAR = 'I'
            # odelujici znaky
            self.C1 = 'Z'
            self.C2 = 'Q'
        self.labelReplace.setText(self.UNUSED_CHAR + " = " + self.REPLACE_CHAR)
        # znovu sestavy sifrovaci tabulku
        self.builTable()

    def builTable(self):
        key = inputFilter(self.lineEditKey.text(), True)
        # sestavy playfair tabulku
        self.table = buildTable(key, self.UNUSED_CHAR, self.REPLACE_CHAR)
        # zobrazi tabulku
        st = ""
        for i in range(0, 5):
            st += '\t\t\t'.join(self.table[0  + i * 5 : 5  + i * 5]) + '\n'
        self.labelTable.setText(st)

    def encrypt(self):   
        # nacte otevreny text a klic
        op = inputFilter(self.textEditIn.toPlainText(), False)
        op = op.replace(self.UNUSED_CHAR, self.REPLACE_CHAR)           
        
        # nahrazeni mezer a cislic
        for r in rep:
            op = op.replace(r.A, r.B)
                
        op = digramSplitter(op, self.C1, self.C2)
                    
        # zasifrovani textu            
        et = playfair(op, self.table, True)
        self.textEditOut.setText(' '.join(wrap(et, 5)))

    def decrypt(self):    
        # nacte sifrovaneho textu a klice
        et = inputFilter(self.textEditIn.toPlainText(), True)
        et = et.replace(self.UNUSED_CHAR, self.REPLACE_CHAR).replace(" ", "")
                
        # desifrovani
        op = playfair(et, self.table, False)
        
        # prislusne retezce nahradi za mezeru nebo cislo
        for r in rep:
            op = op.replace(r.B, r.A)
        
        self.textEditOut.setText(op)

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    app.exec_()  
