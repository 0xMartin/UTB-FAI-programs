#include "mainwindow.h"
#include "./ui_mainwindow.h"

#include <QPaintEvent>
#include <QPainter>
#include <QTime>

MainWindow::MainWindow(QWidget *parent)
    : QMainWindow(parent)
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);

    this->state = None;

    // prednacteni zdroju
    this->meteorImage = QImage(":/img/img/meteor.png");
    this->playerImage = QImage(":/img/img/ufo.png");

    // vytvori hrace
    this->player = new Player(this->playerImage, this);
    connect(this, SIGNAL(drawGame(QPainter&)), this->player, SLOT(paintPlayer(QPainter&)));

    // timer pro game update
    this->gameTimer = new QTimer(this);
    connect(this->gameTimer, SIGNAL(timeout()), this, SLOT(updateGame()));
    this->gameTimer->setInterval(20);
    this->gameTimer->setSingleShot(false);
    this->gameTimer->start();

    // spawn timer
    this->spawnTimer.setSingleShot(true);
    connect(&this->spawnTimer, SIGNAL(timeout()), this, SLOT(addSprite()));

    // mereni casu
    this->timer.restart();
    this->timer.start();
}

MainWindow::~MainWindow()
{
    for(Meteor *m : this->meteores) {
        if(m) delete m;
    }
    if(this->gameTimer) delete this->gameTimer;
    delete ui;
}

void MainWindow::addSprite()
{
    Meteor *m = new Meteor(this->meteorImage, this);
    connect(this, SIGNAL(drawGame(QPainter&)), m, SLOT(paintMeteor(QPainter&)));
    connect(m, SIGNAL(death(Meteor*)), this, SLOT(killMeteor(Meteor*)));
    this->meteores.append(m);
    qDebug() << m << " added!!";

    if(this->state == InGame) {
        srand (time(NULL));
        this->spawnTimer.setInterval(rand() % 2000 + 500);
        this->spawnTimer.start();
    }
}

void MainWindow::updateGame()
{
    if(this->state != InGame) {
        return;
    }

    // repaint
    this->repaint();

    // detekce kolize
    if(this->player != NULL) {
        if(this->player->collisionDetection(this->meteores)) {
            qDebug() << "Player hit!!";
            this->state = GameOver;
            this->repaint();
        }
    }
}

void MainWindow::paintEvent(QPaintEvent *evt)
{
    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing);

    // pozadi
    painter.fillRect(QRect(0, 0, this->width(), this->height()), QBrush(Qt::black));

    // vykresli vsechny objekty hry
    emit this->drawGame(painter);

    // game over text
    if(this->state == GameOver) {
        QFont font("Arial", 44);
        painter.setFont(font);
        QString text = "GAME OVER";
        QRect textRect = painter.boundingRect(rect(), Qt::AlignCenter, text);
        painter.drawText(textRect, Qt::AlignCenter, text);
    }

    // cas
    QFont font("Arial", 18);
    painter.setFont(font);
    QTime time(0, 0);
    time = time.addMSecs(timer.elapsed());
    QString text = time.toString("mm:ss");
    QRect textRect = painter.boundingRect(QRect(0, 0, this->width(), 40), Qt::AlignCenter, text);
    painter.drawText(textRect, Qt::AlignCenter, text);
}

void MainWindow::keyPressEvent(QKeyEvent *event)
{
    if(this->state != InGame) {
        return;
    }

    // ovladani hrace
    if(this->player != NULL) {
        this->player->keyEvent(event, true);
    }
}

void MainWindow::keyReleaseEvent(QKeyEvent *event)
{
    if(this->state != InGame) {
        return;
    }

    // ovladani hrace
    if(this->player != NULL) {
        this->player->keyEvent(event, false);
    }
}


void MainWindow::killMeteor(Meteor *m)
{
    qsizetype index = this->meteores.indexOf(m);
    if(index >= 0) {
        this->meteores.removeAt(index);
        qDebug() << m << " killed!!";
    }
}


void MainWindow::on_actionStart_triggered()
{
    // spusti hru
    if(this->state == InGame) {
        return;
    }

    //pocitani casu
    this->timer.restart();
    this->timer.start();

    // stav hry na "InGame"
    this->state = InGame;

    // spusti spawnovani
    this->spawnTimer.setInterval(500);
    this->spawnTimer.start();

    // clear list
    for(Meteor *m : this->meteores) {
        if(m != NULL) delete m;
    }
    this->meteores.clear();

    // repaint
    this->repaint();
}


void MainWindow::on_actionEnd_triggered()
{
    // ukonci hru
    if(this->state != InGame) {
        return;
    }
    this->state = GameOver;
    this->repaint();
}

