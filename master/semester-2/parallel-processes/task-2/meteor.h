#ifndef SPRITE_H
#define SPRITE_H

#include <QWidget>
#include <QTimer>
#include <QPainter>

class Meteor : public QTimer
{ 
    Q_OBJECT
public:
    /**
     * @brief Meteor
     * @param parrent - Rodicovsky objekt, herni 2D scena
     */
    explicit Meteor(QImage &image, QWidget * parrent);

    int getX() const;
    void setX(int newX);

    int getY() const;
    void setY(int newY);

    float getSize() const;

protected:
    void timerEvent(QTimerEvent * evt) override;

public slots:
    void paintMeteor(QPainter &painter);

signals:
    void death(Meteor *m);

private:
    QWidget * parrent;  /** Rodicovsky objekt, herni 2D scena */

    int X, Y, radius; /** Pozice x, y, rotace */
    float size; /** Velikost */
    float velocityX, velocityY, velocityA; /** Rychlost x, y, rotace */

    QImage image; /** Obrazek objektu */
};

#endif // SPRITE_H
