import java.awt.*;

public class Matrix {

    /**
     * Pocet radku a sloupcu matice
     */
    private int rows, cols;

    /**
     * Hodnoty matice
     */
    private int[] values;

    /**
     * Binarni mod matice -> (2 * 3 = 6 -> 0)
     */
    public boolean BINARY_MODE = false;

    /**
     * Vytvoreni matice: Matrix( {1, 2, 3, 4, 5, 6}, 2, 3)
     * M = [1   2   3]
     *     [4   5   6]
     * @param values Hodnoty matice
     * @param rows Pocet randku matice
     * @param cols Pocet sloupcu matice
     */
    public Matrix(int[] values, int rows, int cols) {
        if(rows * cols != values.length){
            this.values = new int[rows * cols];
            if (this.values.length >= 0) {
                System.arraycopy(values, 0, this.values, 0, this.values.length);
            }
        }else {
            this.values = values;
        }
        this.rows = rows;
        this.cols = cols;
    }

    /**
     * Vytvoreni matice
     * @param rows Pocet randku matice
     * @param cols Pocet sloupcu matice
     */
    public Matrix(int rows, int cols) {
        this.values = new int[rows * cols];
        this.rows = rows;
        this.cols = cols;
    }

    /**
     * Navrati hodnoty matice
     * @return int[]
     */
    public int[] getValues() {
        return this.values;
    }

    /**
     * Zmena hodnot matice
     * @param vals Nove hodnoty
     */
    public void setValues(int[] vals) {
        this.values = vals;
    }

    /**
     * Navrati hodnotu na souradnicich
     * @param row Radek
     * @param col Sloupec
     * @return
     */
    public int getValue(int row, int col){
        if(row >= 0 && row < this.rows && col >= 0 && col < this.cols){
            return this.values[col + row * this.cols];
        }
        return 0;
    }

    /**
     * Zapise hodnotu na souradnicich
     * @param value Hodnota
     * @param row Radek
     * @param col Sloupec
     * @return
     */
    public void setValue(int value, int row, int col){
        if(this.BINARY_MODE){
            if(value != 0 && value != 1) {
                return;
            }
        }
        if(row >= 0 && row < this.rows && col >= 0 && col < this.cols){
            this.values[col + row * this.cols] = value;
        }
    }

    /**
     * Navrati pocet radku
     * @return int
     */
    public final int getRows() {
        return this.rows;
    }

    /**
     * Navrati pocet sloupcu
     * @return int
     */
    public final int getCols() {
        return this.cols;
    }

    /**
     * Nastaveni poctu radku matice
     * @return int
     */
    public void setRows(int val) {
        this.rows = val;
    }

    /**
     * Nastaveni poctu sloupcu matice
     * @return int
     */
    public void setCols(int val) {
        this.cols = val;
    }


    /**
     * Vynasobi tuto matici matici m
     * @param m Matice
     * @return Vysledni matice
     */
    public Matrix multiply(Matrix m) {
        if(m == null){
            return null;
        }

        if(this.cols != m.rows){
            return null;
        }

        //nova matice
        Matrix ret = new Matrix(this.rows, m.cols);
        //pokud obe predchozi matice jsou v bin modu pak bude i tato
        ret.BINARY_MODE = this.BINARY_MODE && m.BINARY_MODE;

        for(int row = 0; row < ret.rows; ++row){
            for(int col = 0; col < ret.cols; ++col){

                //vypoce vysledne hodnoty pro prvek matice [row, col]
                int value = 0;
                for(int i = 0; i < this.cols; ++i){
                    value += this.getValue(row,  i) * m.getValue(i, col);
                }

                //vlozeni vylsednou hodnotu do matice
                //koduje u obou matic binarni mode pak do matice vlozi je prvni bit vylsedneho cisla
                if(this.BINARY_MODE && m.BINARY_MODE) {
                    ret.setValue(value & 0x1, row, col);
                } else {
                    ret.setValue(value, row, col);
                }
            }
        }

        return ret;
    }

    /**
     * Transponuje toto matici a vysledek bude ulozen do nove matice
     * @return Transponovana matice
     */
    public Matrix transpose() {
        Matrix ret = new Matrix(this.cols, this.rows);
        ret.BINARY_MODE = this.BINARY_MODE;

        for(int row  = 0; row < this.rows; ++row) {
            for(int col  = 0; col < this.cols; ++col) {
                ret.setValue(this.getValue(row, col), col, row);
            }
        }

        return ret;
    }

    /**
     * Vrati pozici [col index] sloupcoveho vektoru v matice
     * @param vector Vektor int[], jeho delka musi byt stejna jako pocet radku matice
     * @return col index
     */
    public int colVectorPosition(int[] vector) {
        if(vector.length == this.rows) {
            L1:
            for (int col = 0; col < this.cols; ++col) {

                for (int row = 0; row < this.rows; ++row) {
                    if (this.getValue(row, col) != vector[row]) {
                        continue L1;
                    }
                }

                return col;
            }
        }
        return -1;
    }

    /**
     * Naklonuje toto matici
     * @return Matice
     */
    public Matrix cloneMatrix() {
        int[] newVals = new int[this.values.length];
        System.arraycopy(this.values, 0, newVals, 0, newVals.length);
        Matrix ret = new Matrix(newVals, this.rows, this.cols);
        ret.BINARY_MODE = this.BINARY_MODE;
        return ret;
    }

    /**
     * Prevede matici do string formatu
     * @return String
     */
    public String valuesToString() {
        StringBuilder ret = new StringBuilder();

        for(int row = 0; row < this.rows; ++row) {
            for(int col = 0; col < this.cols; ++col) {
                ret.append(this.values[col + row * this.cols]);
            }
            if(row + 1 < this.rows) {
                ret.append("\n");
            }
        }

        return ret.toString();
    }

}
