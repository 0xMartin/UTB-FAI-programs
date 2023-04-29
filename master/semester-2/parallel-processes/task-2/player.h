#ifndef PLAYER_H
#define PLAYER_H

#include <QWidget>
#include <QTimer>
#include <QImage>

#include "meteor.h"

#define PLAYER_SPEED 4.0
#define PLAYER_SIZE 100

class Player : public QTimer
{
    Q_OBJECT
public:
    explicit Player(QImage &image, QWidget *parent = nullptr);

    float getX() const;
    void setX(float newX);

    float getY() const;
    void setY(float newY);

    void keyEvent(QKeyEvent *event, bool pressed);

    bool collisionDetection(QList<Meteor*> &meteors);

public slots:
    void paintPlayer(QPainter &painter);

protected:
    void timerEvent(QTimerEvent * evt) override;

private:
    QWidget *parent;

    float X, Y;

    bool action_up, action_down;

    QImage image;
};

#endif // PLAYER_H
