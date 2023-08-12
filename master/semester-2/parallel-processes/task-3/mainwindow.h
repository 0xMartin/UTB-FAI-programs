#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

#include "worker.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void on_actionStart_all_triggered();

    void on_actionStop_all_triggered();

    void on_pushButton_start_1_clicked();

    void on_pushButton_stop_1_clicked();

    void on_pushButton_end_1_clicked();

    void on_pushButton_start_2_clicked();

    void on_pushButton_stop_2_clicked();

    void on_pushButton_end_2_clicked();

    void on_actionEnd_all_triggered();

public slots:
    void worker1UpdateSlot(float time, float progress);
    void worker2UpdateSlot(float time, float progress);

    void worker1EstimatedTimeSlot(float time);
    void worker2EstimatedTimeSlot(float time);

    void worker1WorkFinishedSlot(void * result);
    void worker2WorkFinishedSlot(void * result);

private:
    Ui::MainWindow *ui;

    Worker * w1;
    Worker * w2;
};
#endif // MAINWINDOW_H
