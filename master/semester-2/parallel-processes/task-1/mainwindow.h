#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

#include "bgtask.h"

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

    // controll
    void start();
    void stop();
    void reset();

private slots:
    void on_pushButtonStart_clicked();

    void on_pushButtonStop_clicked();

    void on_pushButtonReset_clicked();

    void on_actionStart_triggered();

    void on_actionStop_triggered();

    void on_actionReset_triggered();

public slots:
    // progress
    void progress(float progress, long double pi_value);
    void finished(long double pi_value);

private:
    Ui::MainWindow *ui;

    BgTask task;
};
#endif // MAINWINDOW_H
