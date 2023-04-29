from SimpleApp import *
from Lsystem import *

MainViewID = 1


class MainView(View):
    def __init__(self):
        super().__init__("", MainViewID)

    @overrides(View)
    def createEvt(self):
        al = AbsoluteLayout(self)

        # LEFT SIDE
        # iterations label
        self.label_iter = Label(self, None, "Iteration: 2")
        al.addElement(self.label_iter, ['2%', '5%'])
        # combo box typy
        self.combobox = ComboBox(self, None, ["Tree 3D", "Tree", "Koch's Snowflake", "Square Sierpinski"])
        self.combobox.setValueChangeEvt(self.typeChanged)
        al.addElement(self.combobox, ['18%', '4%', '24%', '5%'])
        # axiom TextInput
        label = Label(self, None, "Axiom:", False, True)
        al.addElement(label, ['2%', '13.5%'])
        self.textinput = TextInput(self, None, "M")
        self.textinput.setTextChangedEvt(self.axiomChanged)
        al.addElement(self.textinput, ['12%', '11%', '30%', '5%'])
        # animation control
        self.btn_last = Button(self, None, "<")
        self.btn_last.setClickEvt(self.animationControl)
        al.addElement(self.btn_last, ['2%', '18%', '19%', '5%'])
        self.btn_next = Button(self, None, ">")
        self.btn_next.setClickEvt(self.animationControl)
        al.addElement(self.btn_next, ['22%', '18%', '20%', '5%'])
        # table
        self.word_list = []
        self.tData = {
            "header": ["Iteration", "Word"],
            "body": self.word_list
        }
        self.table = Table(self, None, self.tData)
        al.addElement(self.table, ['2%', '25%', '40%', '70%'])

        # RIGHT SIDE
        # canvas for l-system
        self.canvas = Canvas(self, None)
        self.canvas.enableMouseControl()
        al.addElement(self.canvas, ['43%', '10%', '55%', '85%'])
        # scale
        self.slider = Slider(self, None, 1.0, 0.01, 2.0)
        self.slider.setLabelFormat("Scale: @")
        self.slider.setOnValueChange(self.scale)
        al.addElement(self.slider, ['43%', '5%', '45%', '15'])

        # L system module
        self.lsystem = LSystem(self.canvas, self.word_list)
        self.combobox.setSelectedItem("Tree 3D")
        # pocet iteraci
        self.lsystem.setIterations(4)
        # do text inputu vlozi axiom
        self.textinput.setText(self.lsystem.getConfig().getAxiom())

        # vsechny elementy prida do toho view
        self.addGUIElements([
            self.canvas, self.table, self.btn_last, self.btn_next,
            self.label_iter, self.slider, self.textinput, label,
            self.combobox
        ])

    @overrides(View)
    def closeEvt(self):
        pass

    @overrides(View)
    def openEvt(self):
        self.refresh()

    @overrides(View)
    def hideEvt(self):
        pass

    @overrides(View)
    def reloadStyleEvt(self):
        pass

    def animationControl(self, btn):
        # ovladani animace (+ / - itarace)
        if btn == self.btn_last:
            # last
            self.lsystem.setIterations(self.lsystem.getIterations() - 1)
        else:
            # next
            self.lsystem.setIterations(self.lsystem.getIterations() + 1)
        # obnoveni
        self.refresh()

    def scale(self, scale):
        # zmena meritka vykreslovani
        self.lsystem.setScale(scale)
        self.lsystem.process()

    def axiomChanged(self, axiom):
        # zmena axiomu
        self.lsystem.setAxiom(axiom)
        self.refresh()

    def typeChanged(self, type):
        # zmena typu L-Systemu
        if type == 'Tree':
            if not isinstance(self.lsystem.getConfig(), Tree):
                self.lsystem.setConfig(Tree())
        if type == 'Tree 3D':
            if not isinstance(self.lsystem.getConfig(), Tree3D):
                self.lsystem.setConfig(Tree3D())
        elif type == 'Koch\'s Snowflake':
            if not isinstance(self.lsystem.getConfig(), KochsSnowflake):
                self.lsystem.setConfig(KochsSnowflake())
        elif type == 'Square Sierpinski':
            if not isinstance(self.lsystem.getConfig(), SquareSierpinski):
                self.lsystem.setConfig(SquareSierpinski())
        self.textinput.setText(self.lsystem.getConfig().getAxiom())
        self.refresh()

    def refresh(self):
        # obnoveni L-SYSTEMU
        self.lsystem.process()
        # obnoveni dat v tabulce
        self.table.refreshTable()
        # zmena labelu (iterace)
        self.label_iter.setText(
            "Iteration: " + str(self.lsystem.getIterations()))


def main():
    main_view = MainView()
    app = Application([main_view], 40, 1, True)
    app.init(1100, 700, "L-SYSTEMS", "")
    app.run(main_view)


if __name__ == "__main__":
    main()
