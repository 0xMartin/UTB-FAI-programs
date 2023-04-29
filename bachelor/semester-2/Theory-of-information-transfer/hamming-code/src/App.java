import javax.swing.*;
import javax.swing.event.ListSelectionEvent;
import javax.swing.event.ListSelectionListener;
import javax.swing.text.*;
import java.awt.*;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.awt.event.MouseAdapter;
import java.awt.event.MouseEvent;
import java.util.ArrayList;

public class App extends JFrame {

    private JPanel panel;
    private JTable tableCharacters;
    private JComboBox comboBoxMatrix;
    private JPanel panelGeneratingMatrix;
    private JButton buttonSecureWord;
    private JLabel labelSelectedWord;
    private JLabel labelSecuredWord2;
    private JLabel labelSecuredWord;
    private JPanel panelControlMatrix;
    private JComboBox comboBoxError;
    private JButton buttonGenerateError;
    private JButton fixError;
    private JPanel panelSyndromMatrix;
    private JLabel labelErrorWord;
    private JLabel labelFixedWord;
    private JLabel labelOutputChar;



    //generujici matice
    private Matrix generatingMatrix;

    //controlni matice
    private Matrix controlMatrix;

    //syndrom
    private Matrix syndromMatrix;

    //zabezpecene kodove slovo
    private Matrix securedWord;

    //slovo s umele vytvorenou chybou
    private Matrix errorWord;


    public App(int width, int height) {
        super.setTitle("Hamminguv kod");
        super.setSize(width, height);
        super.getContentPane().add(panel);
        super.setLocationRelativeTo(null);
        super.setDefaultCloseOperation(EXIT_ON_CLOSE);
    }

    private void createUIComponents() {

        //component pro zobrazeni a editaci generujici matice
        this.panelGeneratingMatrix = new MatrixEditor();
        ((MatrixEditor) this.panelGeneratingMatrix).setAllowedNumbers(new char[]{'0', '1'});
        ((MatrixEditor) this.panelGeneratingMatrix).setNumberConstraint(0, 1);
        ((MatrixEditor) this.panelGeneratingMatrix).setValueChangedListener((evt) -> {

            //v zalozce zabezpeceni nastavy barvu vysledne hodnoty na LIGHT_GRAY
            labelSecuredWord2.setForeground(Color.LIGHT_GRAY);

            //v zalozce detekce a oprava chyb nastavy barvu vyslednych hodnot na LIGHT_GRAY
            this.labelOutputChar.setForeground(Color.LIGHT_GRAY);
            this.labelFixedWord.setForeground(Color.LIGHT_GRAY);
            this.panelSyndromMatrix.setForeground(Color.LIGHT_GRAY);
        });


        //component pro zobrazeni a editaci kontrolni matice
        this.panelControlMatrix = new MatrixEditor();
        ((MatrixEditor) this.panelControlMatrix).setAllowedNumbers(new char[]{'0', '1'});
        ((MatrixEditor) this.panelControlMatrix).setNumberConstraint(0, 1);

        //component pro zobrazeni a matice syndromu
        this.panelSyndromMatrix = new MatrixEditor();

        //tabulka znaku
        this.tableCharacters = new CustomTable(new String[]{"Znak", "Kodové slovo"}, 0) {
            public boolean editCellAt(int row, int column, java.util.EventObject e) {
                return false;
            }
        };
        ListSelectionModel selectionModel = this.tableCharacters.getSelectionModel();
        selectionModel.addListSelectionListener(new ListSelectionListener() {
            @Override
            public void valueChanged(ListSelectionEvent evt) {
                //vybere kodove slovo z tabulko pro kodovani
                labelSelectedWord.setText(
                        tableCharacters.getValueAt(tableCharacters.getSelectedRow(), 1).toString()
                );
                labelSecuredWord2.setForeground(Color.LIGHT_GRAY);
            }
        });
    }

    public void init() {
        //vyber generujici matice
        setGeneratingMatrix(0);

        //inicializace kontrolni matice
        this.controlMatrix = new Matrix(new int[]{
                0, 0, 0, 1, 1, 1, 1,
                0, 1, 1, 0, 0, 1, 1,
                1, 0, 1, 0, 1, 0, 1
        }, 3, 7);
        this.controlMatrix.BINARY_MODE = true;
        ((MatrixEditor) this.panelControlMatrix).setMatrix(this.controlMatrix);

        //inicializace hodnot v tabulce se znaky
        for (int i = 0; i < 16; ++i) {
            String bin = Integer.toBinaryString(i);
            while (bin.length() < 4) {
                bin = '0' + bin;
            }
            ((CustomTable) this.tableCharacters).addRow(
                    new Object[]{(char) ('A' + i) + "", bin});
        }

        //inicializace listeneru
        initListeners();
    }

    public void run() {
        super.setVisible(true);
    }

    public static void main(String[] agrs) {
        //set windows look and feel
        try {
            UIManager.setLookAndFeel("com.sun.java.swing.plaf.windows.WindowsLookAndFeel");
        } catch (ClassNotFoundException | InstantiationException
                | IllegalAccessException | UnsupportedLookAndFeelException e) {
            e.printStackTrace();
        }
        //run application
        App app = new App(600, 500);
        app.init();
        app.run();
    }

    /**
     * Nahrati html text s barevne oznacenymi znaky
     * @param indexes Index znaku, ktera meji byt barevne oznaceny
     * @param text Vstupni text
     * @param color Barva oznaceni
     * @return HTML text
     */
    private String highlightedCharacters(ArrayList<Integer> indexes, String text, String color) {
        StringBuilder builder = new StringBuilder();
        builder.append("<html>");

        for (int i = 0; i < text.length(); ++i) {
            boolean isIn = false;
            for (Integer index : indexes) {
                if ((index == i)) {
                    isIn = true;
                    break;
                }
            }
            if (isIn) {
                //pokud se index znaku nachazi v listu indexu pak tento znak obarvy
                builder.append("<font color=\"" + color + "\">" + text.charAt(i) + "</font>");
            } else {
                //noramlni barva
                builder.append(text.charAt(i));
            }
        }

        builder.append("</html>");
        return builder.toString();
    }

    /**
     * Nastaveni generujici matice
     * @param id
     */
    private void setGeneratingMatrix(int id) {
        switch (id) {
            case 0:
                this.generatingMatrix = new Matrix(new int[]{
                        1, 0, 0, 0, 0, 1, 1,
                        0, 1, 0, 0, 1, 0, 1,
                        0, 0, 1, 0, 1, 1, 0,
                        0, 0, 0, 1, 1, 1, 1
                }, 4, 7);
                break;
            case 1:
                this.generatingMatrix = new Matrix(new int[]{
                        1, 1, 1, 0, 0, 0, 0,
                        1, 0, 0, 1, 1, 0, 0,
                        0, 1, 0, 1, 0, 1, 0,
                        1, 1, 0, 1, 0, 0, 1
                }, 4, 7);
                break;
            case 2:
                this.generatingMatrix = new Matrix(new int[]{
                        1, 1, 0, 1, 0, 0, 1,
                        0, 1, 0, 1, 0, 1, 0,
                        1, 1, 1, 0, 0, 0, 0,
                        1, 0, 0, 1, 1, 0, 0
                }, 4, 7);
                break;
            default:
                this.generatingMatrix = new Matrix(4, 7);
                break;
        }

        //nastavy binarni mod
        this.generatingMatrix.BINARY_MODE = true;

        //nastavi tuto matici pro zobraze v matrix editoru
        ((MatrixEditor) this.panelGeneratingMatrix).setMatrix(this.generatingMatrix);
    }

    private void initListeners() {
        //zmena generujici matice
        comboBoxMatrix.addActionListener((evt) -> {
            setGeneratingMatrix(comboBoxMatrix.getSelectedIndex());
        });

        //zabezpeceni kodoveho slova
        buttonSecureWord.addActionListener((evt) -> {
            //prevede retezec na pole cisel
            int[] nums = new int[4];
            for (int i = 0; i < 4; ++i) {
                nums[i] = labelSelectedWord.getText().charAt(i) - '0';
            }

            //vytvori vektor s temito hodnotami
            Matrix word = new Matrix(nums, 1, 4);
            word.BINARY_MODE = true;

            //generujici matici vynasobi vektorem kodoveho slova -> zabezpecene kodove slovo
            securedWord = word.multiply(generatingMatrix);

            //na vystup zapise vysledne zabezpecene kodove slovo
            try {
                this.labelSecuredWord2.setText(this.securedWord.valuesToString());
                this.labelSecuredWord2.setForeground(Color.BLACK);

                this.labelSecuredWord.setText(this.labelSecuredWord2.getText());
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this,
                        "Chyba při nasobení matic",
                        "Error",
                        JOptionPane.ERROR_MESSAGE);
            }
        });

        //vytvoreni chyb v kodovem slovu
        buttonGenerateError.addActionListener((evt) -> {
            if (securedWord != null) {
                //pocet chyb
                int numbeOfErrors = Integer.parseInt(comboBoxError.getSelectedItem().toString());

                //nahodne vygeneruje indexy, na kterych ma vzniknou chyba
                ArrayList<Integer> errIndexs = new ArrayList<>();
                int count = 0;
                while (count < numbeOfErrors) {
                    int index = (int) (Math.random() * securedWord.getValues().length);
                    if (errIndexs.stream().allMatch((el) -> (el != index))) {
                        errIndexs.add(index);
                        ++count;
                    }
                }

                //na pozici kazdeho nahodne vygenerovaneho indexu zmeni hodnotu bitu na opacnou
                errorWord = securedWord.cloneMatrix();
                errIndexs.stream().forEach((index) -> {
                    errorWord.getValues()[index] = errorWord.getValues()[index] == 1 ? 0 : 1;
                });

                //kodove slovo s chybou zapise na vystup
                labelErrorWord.setText(
                        highlightedCharacters(errIndexs, errorWord.valuesToString(), "red")
                );

                //zmana barev
                this.labelOutputChar.setForeground(Color.LIGHT_GRAY);
                this.labelFixedWord.setForeground(Color.LIGHT_GRAY);
                this.panelSyndromMatrix.setForeground(Color.LIGHT_GRAY);
            }
        });

        //opraveni chyb
        fixError.addActionListener((evt) -> {
            //vypocima matici syndromu
            syndromMatrix = controlMatrix.multiply(errorWord.transpose());

            try {
                //zobrazi matici syndromu
                ((MatrixEditor) panelSyndromMatrix).setMatrix(syndromMatrix);

                //vytvori novy vektor, ve kterem opravy chybny bit
                Matrix fixedWord = this.errorWord.cloneMatrix();
                //najde index bitu, ktery ma opravyt (pozice sloupcoveho vektoru "syndromu" v kontrolni matici)
                int index = this.controlMatrix.colVectorPosition(this.syndromMatrix.getValues());
                fixedWord.setValue(fixedWord.getValue(0, index) == 1 ? 0 : 1, 0, index);

                //opravene kodove slovo zapise na vystup
                //zelene oznaci znak, ktery byl opraven
                ArrayList<Integer> indexes = new ArrayList<>();
                indexes.add(index);
                this.labelFixedWord.setText(
                        highlightedCharacters(indexes, fixedWord.valuesToString(), "green")
                );

                //dekoduje buvodne zakodovany znak
                int tableIndex = 0;
                for (int i = 0; i < 4; i++) {
                    //najde pozici, na ktere se nachazi bit puvodniho kodoveho slova a pak ho vynasoby jeho vahou
                    //tuto hodnotu pricte k indexu a nasledne tento postup opakuje jeste 3x
                    tableIndex += (int) Math.pow(2, 3 - i) * fixedWord.getValue(0,
                            this.generatingMatrix.colVectorPosition(new int[]{
                                    i == 0 ? 1 : 0,
                                    i == 1 ? 1 : 0,
                                    i == 2 ? 1 : 0,
                                    i == 3 ? 1 : 0
                            }));
                }
                this.labelOutputChar.setText(this.tableCharacters.getValueAt(tableIndex, 0).toString());

                //zmeni barvy
                this.labelOutputChar.setForeground(Color.BLACK);
                this.labelFixedWord.setForeground(Color.BLACK);
                this.panelSyndromMatrix.setForeground(Color.BLACK);
            } catch (Exception ex) {
                JOptionPane.showMessageDialog(this,
                        "Chyba při nasobení matic",
                        "Error",
                        JOptionPane.ERROR_MESSAGE);
            }
        });
    }

}
