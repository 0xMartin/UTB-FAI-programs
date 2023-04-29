import javax.swing.*;
import java.awt.*;
import java.awt.event.*;
import java.util.Arrays;

public class MatrixEditor extends JPanel implements MouseListener, KeyListener {

    //matice, ktera se bude zobrazovat
    private Matrix matrix;

    //povolene cislice, ktere mohou byt vkladane do matice
    private char[] allowedNums;

    //omezeni min a max cisla
    private int min = Integer.MIN_VALUE, max = Integer.MAX_VALUE;

    //pozice oznaceneho prvku matice (col, row)
    private final Point selected = new Point(-1, -1);

    //listener, ktery zavola event pri zmene hodnot matice
    private ActionListener valueChangedListener;

    //menu s operacemi pro editaci matice
    private final JPopupMenu menu = new JPopupMenu();

    public MatrixEditor() {
        super.addMouseListener(this);
        super.addKeyListener(this);

        //init menu
        JMenuItem item;

        //trasponovat
        item = new JMenuItem("Transponovat");
        item.addActionListener((evt)->{
            Matrix m = this.matrix.transpose();
            this.matrix.setCols(m.getCols());
            this.matrix.setRows(m.getRows());
            System.arraycopy(m.getValues(), 0, this.matrix.getValues(), 0, this.matrix.getValues().length);
            super.repaint();

            //zavola event
            if(this.valueChangedListener != null) {
                this.valueChangedListener.actionPerformed(new ActionEvent(this, 0, ""));
            }
        });
        this.menu.add(item);

        //nulovat
        item = new JMenuItem("Nulovat");
        item.addActionListener((evt)->{
            Arrays.fill(this.matrix.getValues(), 0);
            super.repaint();

            //zavola event
            if(this.valueChangedListener != null) {
                this.valueChangedListener.actionPerformed(new ActionEvent(this, 0, ""));
            }
        });
        this.menu.add(item);

        //zmenit velikost
        item = new JMenuItem("Změnit velikost");
        item.addActionListener((evt)->{
            JTextField rows = new JTextField();
            JTextField cols = new JTextField();
            Object[] message = {
                    "Počet řádků:", rows,
                    "Počet sloupců:", cols
            };

            int option = JOptionPane.showConfirmDialog(null, message, "Změna velikosti matice", JOptionPane.OK_CANCEL_OPTION);
            if (option == JOptionPane.OK_OPTION) {
                try {
                    int iRow = Integer.parseInt(rows.getText());
                    int iCol = Integer.parseInt(cols.getText());

                    if (iRow > 0 && iCol > 0) {
                        this.matrix.setValues(new int[iRow * iCol]);
                        this.matrix.setRows(iRow);
                        this.matrix.setCols(iCol);
                        super.repaint();

                        //zavola event
                        if (this.valueChangedListener != null) {
                            this.valueChangedListener.actionPerformed(new ActionEvent(this, 0, ""));
                        }
                    }
                }catch (Exception ex) {}
            }
        });
        this.menu.add(item);
    }

    public void setMatrix(Matrix m){
        //nastaveni reference na matici
        this.matrix = m;

        //resetovani oznaceni
        this.selected.x = -1;

        //prekreslit matici
        super.repaint();
    }

    /**
     * Nastaveni povelenych cislic
     */
    public void setAllowedNumbers(char[] numbers) {
        this.allowedNums = numbers;
    }

    public void setNumberConstraint(int min, int max) {
        this.min = min;
        this.max = max;
    }

    public void setValueChangedListener(ActionListener listener) {
        this.valueChangedListener = listener;
    }

    //padding
    private int VPadding, HPadding;

    //physics size of matrix
    private Dimension size;

    private void computePaddingAndSize(Graphics2D g2) {
        //vypocita vertikalni mezery
        this.VPadding = 0;
        for (int value : this.matrix.getValues()) {
            this.VPadding = Math.max(this.VPadding, g2.getFontMetrics().stringWidth("" + value));
        }
        this.VPadding += 6;

        //vypocita horizontalni mezery
        this.HPadding = g2.getFontMetrics().getHeight();

        //vypocita fyzickou velikost matice
        this.size = new Dimension(
                this.VPadding * this.matrix.getCols() + 16,
                this.HPadding * this.matrix.getRows()
        );

        //nastavi velikost zobrazovaciho panelu
        super.setPreferredSize(new Dimension(this.size.width + 100, this.size.height + 100));
        super.revalidate();
    }

    @Override
    protected void paintComponent(Graphics g){
        super.paintComponent(g);

        if(this.matrix == null){
            return;
        }

        if(this.matrix.getValues() == null){
            return;
        }

        if(this.matrix.getRows() < 1 || this.matrix.getCols() < 1){
            return;
        }

        Graphics2D g2 = (Graphics2D) g;

        //vypocita xoffset a yoffset pro vycenterovani matice
        //vypocita jeji fyzickou velikost
        //zmeni velikost zobrazovaciho panelu
        computePaddingAndSize(g2);

        g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
        g2.setRenderingHint(RenderingHints.KEY_RENDERING, RenderingHints.VALUE_RENDER_QUALITY);
        g2.setRenderingHint(RenderingHints.KEY_STROKE_CONTROL, RenderingHints.VALUE_STROKE_PURE);

        //offset (center)
        int xOffset = (this.getWidth() - this.size.width) / 2 + VPadding,
                yOffset = (this.getHeight() - this.size.height) / 2;

        //vykresli cisla matice
        for(int row = 0; row < this.matrix.getRows(); ++row){
            for(int col = 0; col < this.matrix.getCols(); ++col){

                if(row == this.selected.y && col == this.selected.x) {
                    //oznaceni prvku
                    g2.setColor(Color.BLUE);
                    g2.fillRect(
                            xOffset + col * this.VPadding - 2,
                            yOffset + (int)((row - 0.25f) * this.HPadding) + 4,
                            this.VPadding,
                            this.HPadding
                    );
                    g2.setColor(Color.WHITE);
                } else {
                    //normalni zobrazeni
                    g2.setColor(this.getForeground());
                }

                g2.drawString(
                        this.matrix.getValue(row, col) +"",
                        xOffset + col * this.VPadding,
                        yOffset + (int)((row + 0.75f) * this.HPadding)
                );

            }
        }

        //zavorky matice
        g2.setStroke(new BasicStroke(2));
        g2.setColor(this.getForeground());

        //leva
        g2.drawLine(
                xOffset,
                yOffset,
                xOffset - 8,
                yOffset
        );
        g2.drawLine(
                xOffset,
                yOffset + this.HPadding * this.matrix.getRows(),
                xOffset - 8,
                yOffset + this.HPadding * this.matrix.getRows()
        );
        g2.drawLine(
                xOffset - 8,
                yOffset,
                xOffset - 8,
                yOffset + this.HPadding * this.matrix.getRows()
        );

        //prava
        xOffset += (int)((this.matrix.getCols() - 0.5f) * this.VPadding);

        g2.drawLine(
                xOffset,
                yOffset,
                xOffset + 8,
                yOffset
        );
        g2.drawLine(
                xOffset,
                yOffset + this.HPadding * this.matrix.getRows(),
                xOffset + 8,
                yOffset + this.HPadding * this.matrix.getRows()
        );
        g2.drawLine(
                xOffset + 8,
                yOffset,
                xOffset + 8,
                yOffset + this.HPadding * this.matrix.getRows()
        );

    }

    public void mouseClicked(MouseEvent evt){

    }

    public void mousePressed(MouseEvent evt) {
        if(this.matrix == null || !super.isEnabled()) {
            return;
        }

        this.requestFocus();

        //resetovani oznaceni
        this.selected.x = -1;

        //offset (center)
        int xOffset = (this.getWidth() - this.size.width) / 2 + this.VPadding,
                yOffset = (this.getHeight() - this.size.height) / 2;

        int xVal, yVal;
        for(int row = 0; row < this.matrix.getRows(); ++row){
            for(int col = 0; col < this.matrix.getCols(); ++col){

                //fyzicka pozice hodnoty
                xVal = xOffset + col * this.VPadding;
                yVal = yOffset + row * this.HPadding;

                //pokud se kurzor nachazi v oblasti hodnoty matice
                if(evt.getX() >= xVal && evt.getX() <= xVal + this.VPadding &&
                        evt.getY() >= yVal && evt.getY() <= yVal + this.HPadding){
                    this.selected.x = col;
                    this.selected.y = row;
                    break;
                }
            }
        }

        super.repaint();
    }

    public void mouseReleased(MouseEvent evt) {
        //zobrazeni menu
        if(evt.getButton() == 3 && this.matrix != null && super.isEnabled()) {
            this.menu.show(this, evt.getX(), evt.getY());
        }
    }

    public void mouseEntered(MouseEvent evt) {

    }

    public void mouseExited(MouseEvent evt) {

    }

    public void keyTyped(KeyEvent evt) {

    }

    public void keyPressed(KeyEvent evt) {
        if(this.selected.x < 0 && this.selected.y < 0 || !super.isEnabled()){
            return;
        }

        //pokud existuje omazeni cislic pak pokud nenajde cislici v listu nedojde ke zmene hodnoty prvku matice
        if(this.allowedNums != null && evt.getKeyCode() != KeyEvent.VK_BACK_SPACE) {
            boolean isIn = false;
            for(char c : this.allowedNums){
                if(c == evt.getKeyChar()) {
                    isIn = true;
                    break;
                }
            }
            if(!isIn) {
                return;
            }
        }

        //aktualni hodnota oznaceneho prvku
        int currentVal = this.matrix.getValue(this.selected.y, this.selected.x);

        if(evt.getKeyCode() == KeyEvent.VK_BACK_SPACE){
            //odebere cislici nejnizsiho radu
            currentVal = (currentVal - currentVal % 10) / 10;
        }else if(Character.isDigit(evt.getKeyChar())){
            //prida cislici
            currentVal = currentVal * 10 + (evt.getKeyChar() - '0');
        }else if(evt.getKeyChar() == '-'){
            //opacna hodnota
            currentVal *= -1;
        }

        //omezeni velikosti hodnoty
        currentVal = Math.min(currentVal, this.max);
        currentVal = Math.max(currentVal, this.min);

        //pokud se nova hodnota lisi od minule pak zavola event
        if(this.matrix.getValue(this.selected.y, this.selected.x) != currentVal
                && this.valueChangedListener != null) {
            this.valueChangedListener.actionPerformed(new ActionEvent(this, 0, ""));
        }

        //nastaveni hodnoty
        this.matrix.setValue(currentVal, this.selected.y, this.selected.x);

        super.repaint();
    }

    public void keyReleased(KeyEvent evt) {

    }

}
