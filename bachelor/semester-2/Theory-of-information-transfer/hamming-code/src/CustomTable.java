import javax.swing.*;
import javax.swing.table.DefaultTableCellRenderer;
import javax.swing.table.DefaultTableModel;
import java.awt.*;

public class CustomTable extends JTable {

    private final DefaultTableModel model;

    public CustomTable(String[] header, int rowCount) {
        //create table model
        this.model = new DefaultTableModel(header, rowCount);
        super.setModel(this.model);
        //set cell renderer
        super.setDefaultRenderer(Object.class, new CellRenderer(JLabel.CENTER));
        //table style
        super.setDragEnabled(false);
        super.setFont(new Font("Arial", Font.PLAIN, 14));
        super.setRowHeight(28);
        super.setGridColor(Color.LIGHT_GRAY);
        //header style
        super.tableHeader.setReorderingAllowed(false);
        super.tableHeader.setBorder(null);
        super.tableHeader.setDefaultRenderer(new HeaderRenderer(JLabel.CENTER));
    }

    @Override
    protected void paintComponent(Graphics g) {
        Graphics2D g2 =(Graphics2D) g;
        g2.setRenderingHint(
                RenderingHints.KEY_ANTIALIASING,
                RenderingHints.VALUE_ANTIALIAS_ON
        );
        super.paintComponent(g2);
    }

    public void addRow(Object[] rowData) {
        this.model.addRow(rowData);
    }

    public void removeAllRows(){
        //remove all row from table data vector and repaint table
        this.model.getDataVector().removeAllElements();
        super.revalidate();
        super.repaint();
    }

    private static class CellRenderer extends DefaultTableCellRenderer {

        public CellRenderer(int alignment) {
            super.setHorizontalAlignment(alignment);
        }

        @Override
        public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected,
                boolean hasFocus, int row, int col) {
            return super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, col);
        }
    }

    private static class HeaderRenderer extends DefaultTableCellRenderer {

        public HeaderRenderer(int alignment){
            super.setHorizontalAlignment(alignment);
        }

        @Override
        public Component getTableCellRendererComponent(JTable table, Object value, boolean isSelected,
                boolean hasFocus, int row, int col) {
            JLabel l = (JLabel)super.getTableCellRendererComponent(table, value, isSelected, hasFocus, row, col);
            l.setFont(l.getFont().deriveFont(Font.BOLD));
            l.setBackground(Color.DARK_GRAY);
            l.setForeground(Color.WHITE);
            return l;
        }
    }

}
