#include "player.h"

#include <QPainter>
#include <QKeyEvent>

Player::Player(QImage &image, QWidget *parent)
{
    this->parent = parent;
    this->X = 60;
    this->Y = this->parent->height() / 2;
    this->action_up = false;
    this->action_down = false;

    // nacte obazek objektu s reference
    this->image = image;

    // nastavi timer pro vypocet fyziky tohoto objektu
    this->setInterval(20);
    this->setSingleShot(false);
    this->start();
}

float Player::getY() const
{
    return Y;
}

void Player::setY(float newY)
{
    Y = newY;
}

float Player::getX() const
{
    return X;
}

void Player::setX(float newX)
{
    X = newX;
}

void Player::paintPlayer(QPainter &painter)
{
    painter.drawImage(
                QRectF(this->X - PLAYER_SIZE/2, this->Y - PLAYER_SIZE/2, PLAYER_SIZE, PLAYER_SIZE),
                this->image,
                QRectF(0, 0, this->image.width(), this->image.height()));
}

void Player::timerEvent(QTimerEvent *evt)
{
    if(this->action_up && !this->action_down) {
        this->Y -= PLAYER_SPEED;
    }
    if(!this->action_up && this->action_down) {
        this->Y += PLAYER_SPEED;
    }

    this->Y = qMax((float)PLAYER_SIZE/2, qMin((float)this->Y, (float)this->parent->height() - PLAYER_SIZE/2));
}

void Player::keyEvent(QKeyEvent *event, bool pressed)
{
    int key = event->key();
    if (key == Qt::Key_W || key == Qt::Key_Up) {
        this->action_up = pressed;
    } else if (key == Qt::Key_S || key == Qt::Key_Down) {
        this->action_down = pressed;
    }
}

bool Player::collisionDetection(QList<Meteor*> &meteors)
{
    for(Meteor *m : meteors) {
        if(m == NULL) continue;
        float dist2 = qPow(this->X - m->getX(), 2) + qPow(this->Y - m->getY(), 2);
        if(dist2 < qPow(m->getSize()/2, 2) + qPow(PLAYER_SIZE/2, 2) - 9 * 9) {
            return true;
        }
    }
    return false;
}

