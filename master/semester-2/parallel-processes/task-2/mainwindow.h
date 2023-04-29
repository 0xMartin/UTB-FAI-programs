#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QElapsedTimer>

#include "player.h"
#include "meteor.h"

QT_BEGIN_NAMESPACE
namespace Ui { class MainWindow; }
QT_END_NAMESPACE

enum GameState {
    None,
    InGame,
    GameOver
};

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

protected:
    void paintEvent(QPaintEvent *evt) override;

    void keyPressEvent(QKeyEvent *event) override;

    void keyReleaseEvent(QKeyEvent *event) override;

private slots:
    /**
     * @brief Update hry
     */
    void updateGame();

    /**
     * @brief Zabiti spritu
     * @param s - Sprite
     */
    void killMeteor(Meteor *m);

    /**
     * @brief Prida novy meteor do hry
     */
    void addSprite();

    void on_actionStart_triggered();

    void on_actionEnd_triggered();

signals:
    /**
     * @brief Vykresleni vsech hernich objektu
     * @param painter - QPainter
     */
    void drawGame(QPainter &painter);

private:
    Ui::MainWindow *ui;

    QTimer *gameTimer;
    QTimer spawnTimer;
    QElapsedTimer timer;

    GameState state; /** Stav hry*/
    Player *player; /** Hrac */
    QList<Meteor*> meteores; /** Seznam vsech meteoru */

    QImage meteorImage;
    QImage playerImage;
};
#endif // MAINWINDOW_H
