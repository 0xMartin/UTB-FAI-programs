#ifndef BGTASK_H
#define BGTASK_H

#include <QTimer>

class BgTask: public QTimer
{
    Q_OBJECT
public:
    explicit BgTask(qlonglong iterations, QObject *parent = nullptr);

    long double getPIValue() const;

    void reset();

protected:
    void timerEvent(QTimerEvent * evt) override;

signals:
    void progress(float progress, long double pi_value);
    void finished(long double pi_value);

private:
    qlonglong pts_cnt_circle;
    qlonglong current_it;
    qlonglong iterations;

    double x = 0.0;
    double y = 0.0;
    double dist = 0.0;
};

#endif // BGTASK_H
