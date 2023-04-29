#include "bgtask.h"
#include <QDebug>

BgTask::BgTask(qlonglong iterations, QObject *parent)
{
    this->pts_cnt_circle = 0;
    this->current_it = 0;
    this->iterations = iterations;

    this->setSingleShot(false);
    this->setInterval(0);
    srand(time(NULL));
}

long double BgTask::getPIValue() const
{
    return 4.0 * (long double) this->pts_cnt_circle / (long double) this->current_it;
}

void BgTask::reset()
{
    this->pts_cnt_circle = 0;
    this->current_it = 0;
}

void BgTask::timerEvent(QTimerEvent * evt)
{
    if(this->current_it >= this->iterations) {
        this->stop();
        emit this->finished(this->getPIValue());
        return;
    }

    // vypocet cisla PI
    this->x = (double)rand() / RAND_MAX;
    this->y = (double)rand() / RAND_MAX;
    dist = x * x + y * y;
    if (dist < 1) {
        this->pts_cnt_circle++;
    }

    ++this->current_it;

    // progress bar (po 1000 iteracich)
    if(this->current_it % 1000 == 0) {
        emit progress(
                    (float) this->current_it / (float) this->iterations * 100.0,
                    4.0 * (long double) this->pts_cnt_circle / (long double) this->current_it
                    );
    }
}
