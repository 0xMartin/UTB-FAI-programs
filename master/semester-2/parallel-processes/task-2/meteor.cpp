#include "meteor.h"

#include <QBrush>

Meteor::Meteor(QImage &image, QWidget * parrent)
{
    this->parrent = parrent;

    // nahodne vygeneruje pohybove vlastnosti
    srand (time(NULL));
    this->size = rand() % 60 + 40;
    this->velocityX = -(rand() % 30 + 9) / 10.0;
    this->velocityY = (rand() % 80 - 40) / 10.0;
    this->velocityA = (rand() % 6 - 3);

    this->X = parrent->width() + size + (rand() % 120);
    this->Y = rand() % parrent->height();

    // nastavi timer pro vypocet fyziky tohoto objektu
    this->setInterval(15);
    this->setSingleShot(false);
    this->start();

    // nacte obazek reference
    this->image = image;
}

int Meteor::getX() const
{
    return X;
}

void Meteor::setX(int newX)
{
    X = newX;
}

int Meteor::getY() const
{
    return Y;
}

void Meteor::setY(int newY)
{
    Y = newY;
}

float Meteor::getSize() const
{
    return size;
}

void Meteor::timerEvent(QTimerEvent *evt)
{
    // vypocet fyziky
    QSize size = this->parrent->size();

    // pohyb
    this->X += this->velocityX;
    this->Y += this->velocityY;
    if(this->Y < 0 || this->Y > size.height()) {
        this->velocityY *= -1; // dodraz jen v ose Y
    }

    // rotace
    this->radius += this->velocityA;
    while(radius < 0) radius += 360.0;

    // pokud je za levym okraje (az ve chvili kdy neni videt) vysle signal pro odstraneni
    if(this->X + this->size / 2 < 0) {
        emit death(this);
    }
}

void Meteor::paintMeteor(QPainter &painter)
{
    // vykresleni objektu
    painter.save();
    painter.translate(QPoint(this->X, this->Y));
    painter.rotate(this->radius);
    painter.drawImage(
                QRectF(- this->size/2, - this->size/2, this->size, this->size),
                this->image,
                QRectF(0, 0, this->image.width(), this->image.height()));
    painter.restore();
}


