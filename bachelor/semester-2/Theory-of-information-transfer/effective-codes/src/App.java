import javax.swing.*;
import javax.swing.tree.TreeNode;
import java.awt.event.KeyAdapter;
import java.awt.event.KeyEvent;

public class App extends JFrame {

    private JTable table1;
    private JTable table2;
    private JPanel rootPanel;
    private JLabel jLabelAvgLength;
    private JTextArea textAreaOutput;
    private JTextArea textAreaInput;
    private JLabel jLabelEffectivity;
    private JPanel jPanelTreeView;
    private JComboBox comboBox1;

    //algoritmus pro navrh efektivniho kodu
    private EffectiveCodeAlghoritm effCodeAlg = new ShannonFanoAlgorithm();

    public App(String title, int width, int height) {
        super.setTitle(title);
        super.setSize(width, height);
        super.setLocationRelativeTo(null);
        super.setDefaultCloseOperation(EXIT_ON_CLOSE);
        super.add(rootPanel);
        this.comboBox1.addActionListener(actionEvent -> {
            //vybere algoritmus pro navrh efektivniho kodu
            switch(this.comboBox1.getSelectedIndex()){
                case 0:
                    this.effCodeAlg = new ShannonFanoAlgorithm();
                    break;
                case 1:
                    this.effCodeAlg = new HuffmanAlgorithm();
                    break;
                default:
            }

            //zakodovani vstupniho textu
            encodeInputText();
        });
    }

    private void createUIComponents() {
        this.table1 = new CustomTable(new String[]{"Znak","Slova"}, 0);
        this.table2 = new CustomTable(new String[]{"Znak", "Pravděpodobnost"}, 0);
    }

    public void init() {
        this.textAreaInput.addKeyListener(new KeyAdapter() {
            @Override
            public void keyReleased(KeyEvent e) {
                encodeInputText();
            }
        });
    }

    private void encodeInputText() {
        //vytvori kod pro vstupni text
        boolean success = effCodeAlg.createCode(textAreaInput.getText());

        if(success) {
            //vymaze vsechny vystupni data
            ((CustomTable)table1).removeAllRows();
            ((CustomTable)table2).removeAllRows();
            textAreaOutput.setText("");
            jLabelAvgLength.setText("");

            //vysledny kod
            CodeWord[] code = effCodeAlg.getCode();

            if(code == null){
                return;
            }

            for (CodeWord n : code) {
                //tabulka znaku a k nim prirazenych kodovych slov
                ((CustomTable) table1).addRow(new String[]{n.character + "", n.code});
                //tabulka s cetnosti vyskytu jednotlivych znaku
                ((CustomTable) table2).addRow(new String[]{n.character + "", String.format("%.3f", n.probability) + ""});
            }

            //zakoduje vstupni text dle kodovych slov
            textAreaOutput.setText(effCodeAlg.getCodedText());

            //pocet bitu
            int count = 0;
            for(int i = 0; i < textAreaOutput.getText().length(); i++){
                char c = textAreaOutput.getText().charAt(i);
                count += c == '1' || c == '0' ? 1 : 0;
            }
            textAreaOutput.setToolTipText(count+" bitu");

            //vypocita a vypise prumernou delku kodoveho slova
            jLabelAvgLength.setText(String.format("%.3f",effCodeAlg.getAvgCodeLength()) +" bit ");
            //vypocita a vypise efektivitu kodu
            jLabelEffectivity.setText(String.format("%.3f",effCodeAlg.getEffectivity()*100)+" % ");
        }
    }

    public static void main(String[] args) {
        //set windows look and feel
        try {
            UIManager.setLookAndFeel("com.sun.java.swing.plaf.windows.WindowsLookAndFeel");
        } catch (ClassNotFoundException | InstantiationException
                | IllegalAccessException | UnsupportedLookAndFeelException e) {
            e.printStackTrace();
        }
        //run application
        App app = new App("Efektivní kódy", 700, 550);
        app.init();
        app.setVisible(true);
    }

}
