#include "mainwindow.h"
#include "./ui_mainwindow.h"

#include <QMessageBox>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
    , task(50000000)
{
    ui->setupUi(this);

    connect(&this->task, SIGNAL(progress(float, long double)), this, SLOT(progress(float, long double)));
    connect(&this->task, SIGNAL(finished(long double)), this, SLOT(finished(long double)));
}

MainWindow::~MainWindow()
{
    delete ui;
}


void MainWindow::on_pushButtonStart_clicked()
{
    this->start();
}


void MainWindow::on_pushButtonStop_clicked()
{
    this->stop();
}


void MainWindow::on_pushButtonReset_clicked()
{
    this->reset();
}


void MainWindow::on_actionStart_triggered()
{
    this->start();
}


void MainWindow::on_actionStop_triggered()
{
    this->stop();
}


void MainWindow::on_actionReset_triggered()
{
    this->reset();
}

void MainWindow::start()
{
    this->ui->pushButtonStart->setEnabled(false);
    this->ui->pushButtonStop->setEnabled(true);
    this->ui->pushButtonReset->setEnabled(false);
    this->task.start();
}

void MainWindow::stop()
{
    this->ui->pushButtonStart->setEnabled(true);
    this->ui->pushButtonStop->setEnabled(false);
    this->ui->pushButtonReset->setEnabled(true);
    this->task.stop();
}

void MainWindow::reset()
{
    this->task.reset();
    this->ui->lineEdit->setText(tr("Vypocet nebyl zahajen!"));
    this->ui->progressBar->setValue(0);
}

void MainWindow::progress(float progress, long double pi_value)
{
    this->ui->lineEdit->setText(QString::number(pi_value, 'f', 15));
    this->ui->progressBar->setValue(progress);
}

void MainWindow::finished(long double pi_value)
{
    this->ui->lineEdit->setText(QString::number(pi_value, 'f', 15));
    this->stop();
    QMessageBox::information(this, tr("Vypocet cisla PI"), tr("Vypocet byl dokoncen!"));
}

