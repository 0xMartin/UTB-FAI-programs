#include "mainwindow.h"
#include "./ui_mainwindow.h"

#include <QMessageBox>
#include "factorial.h"
#include "sieveoferatosthenes.h"


MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    this->w1 = new Factorial(0);
    connect(this->w1, &Factorial::estimatedTime, this, &MainWindow::worker1EstimatedTimeSlot);
    connect(this->w1, &Factorial::update, this, &MainWindow::worker1UpdateSlot);
    connect(this->w1, &Factorial::workFinished, this, &MainWindow::worker1WorkFinishedSlot);

    this->w2 = new SieveofEratosthenes(2);
    connect(this->w2, &SieveofEratosthenes::estimatedTime, this, &MainWindow::worker2EstimatedTimeSlot);
    connect(this->w2, &SieveofEratosthenes::update, this, &MainWindow::worker2UpdateSlot);
    connect(this->w2, &SieveofEratosthenes::workFinished, this, &MainWindow::worker2WorkFinishedSlot);
}


MainWindow::~MainWindow()
{
    delete ui;
    if(this->w1) delete this->w1;
    if(this->w2) delete this->w2;
}


void MainWindow::on_actionStart_all_triggered()
{
    on_pushButton_start_1_clicked();
    on_pushButton_start_2_clicked();
}


void MainWindow::on_actionStop_all_triggered()
{
    on_pushButton_stop_1_clicked();
    on_pushButton_stop_2_clicked();
}


void MainWindow::on_pushButton_start_1_clicked()
{
    QString str = this->ui->lineEdit_in_1->text();
    std::string str_std = str.toStdString();
    for(char c : str_std) {
        if((c < '0' || c > '9') && c != '-') {
            QMessageBox::warning(nullptr, "Factorial", "Invalid number 'n' for factorial! Number must consists only from digits or symbol '-'");
            return;
        }
    }
    ((Factorial*)this->w1)->setN(str_std);

    this->ui->textEdit_res_1->setText("");
    this->ui->progressBar->setValue(0);
    this->ui->pushButton_start_1->setEnabled(false);
    this->ui->pushButton_stop_1->setEnabled(true);
    this->ui->pushButton_end_1->setEnabled(true);
    this->ui->lineEdit_in_1->setEnabled(false);

    this->w1->startWorker();
}


void MainWindow::on_pushButton_stop_1_clicked()
{
    this->w1->stopWorker();
    this->ui->pushButton_start_1->setEnabled(true);
    this->ui->pushButton_stop_1->setEnabled(false);
    this->ui->pushButton_end_1->setEnabled(true);
}


void MainWindow::on_pushButton_end_1_clicked()
{
    this->w1->endWorker();
    this->ui->progressBar->setValue(0);
    this->ui->pushButton_start_1->setEnabled(true);
    this->ui->pushButton_stop_1->setEnabled(false);
    this->ui->pushButton_end_1->setEnabled(false);
    this->ui->lineEdit_in_1->setEnabled(true);
}


void MainWindow::on_pushButton_start_2_clicked()
{
    QString str = this->ui->lineEdit_in_2->text();
    bool ok = true;
    long long n = str.toLongLong(&ok);
    if(!ok) {
        QMessageBox::warning(nullptr, "Factorial", "Invalid number 'n' for factorial! Number must consists only from digits or symbol '-'");
        return;
    }
    ((SieveofEratosthenes*)this->w2)->setN(n);

    this->ui->textEdit_res_2->setText("");
    this->ui->progressBar_2->setValue(0);
    this->ui->pushButton_start_2->setEnabled(false);
    this->ui->pushButton_stop_2->setEnabled(true);
    this->ui->pushButton_end_2->setEnabled(true);
    this->ui->lineEdit_in_2->setEnabled(false);

    this->w2->startWorker();
}


void MainWindow::on_pushButton_stop_2_clicked()
{
    this->w2->stopWorker();
    this->ui->pushButton_start_2->setEnabled(true);
    this->ui->pushButton_stop_2->setEnabled(false);
    this->ui->pushButton_end_2->setEnabled(true);
}


void MainWindow::on_pushButton_end_2_clicked()
{
    this->w2->endWorker();
    this->ui->progressBar_2->setValue(0);
    this->ui->pushButton_start_2->setEnabled(true);
    this->ui->pushButton_stop_2->setEnabled(false);
    this->ui->pushButton_end_2->setEnabled(false);
    this->ui->lineEdit_in_2->setEnabled(true);
}

void MainWindow::worker1UpdateSlot(float time, float progress)
{
    if(!this->w1->paused()) {
        this->ui->lineEdit_t_1->setText(QString::number((int)time) + " s");
        this->ui->progressBar->setValue(progress);
    }
}

void MainWindow::worker2UpdateSlot(float time, float progress)
{
    if(!this->w2->paused()) {
        this->ui->lineEdit_t_2->setText(QString::number((int)time) + " s");
        this->ui->progressBar_2->setValue(progress);
    }
}

void MainWindow::worker1EstimatedTimeSlot(float time)
{
    if(!this->w1->paused()) {
        this->ui->lineEdit_et_1->setText(QString::number((int)time) + " s");
    }
}

void MainWindow::worker2EstimatedTimeSlot(float time)
{
    if(!this->w2->paused()) {
        this->ui->lineEdit_et_2->setText(QString::number((int)time) + " s");
    }
}

void MainWindow::worker1WorkFinishedSlot(void * result)
{
    FactorialType fres = (FactorialType) result;
    this->ui->textEdit_res_1->setText(fres->toString());
    this->ui->progressBar->setValue(100);
    this->ui->pushButton_start_1->setEnabled(true);
    this->ui->pushButton_stop_1->setEnabled(false);
    this->ui->pushButton_end_1->setEnabled(false);
    this->ui->lineEdit_in_1->setEnabled(true);
}

void MainWindow::worker2WorkFinishedSlot(void * result)
{
    SieveofEratosthenesType sres = (SieveofEratosthenesType)result;

    this->ui->textEdit_res_2->clear();
    FOR_EACH_PRIME(sres)
    this->ui->textEdit_res_2->append(QString::number(FOR_EACH_PRIME_NUMBER));
    END_FOR_EACH_PRIME
    this->ui->progressBar_2->setValue(100);
    this->ui->pushButton_start_2->setEnabled(true);
    this->ui->pushButton_stop_2->setEnabled(false);
    this->ui->pushButton_end_2->setEnabled(false);
    this->ui->lineEdit_in_2->setEnabled(true);
}

void MainWindow::on_actionEnd_all_triggered()
{
    on_pushButton_end_1_clicked();
    on_pushButton_end_2_clicked();
}




