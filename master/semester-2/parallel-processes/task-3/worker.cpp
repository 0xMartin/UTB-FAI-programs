#include "worker.h"

#include <QDebug>

Worker::Worker()
{
    this->result = NULL;
    this->current_time = 0.0;
    this->estimated_time = 0.0;
    this->started = false;
    this->m_paused = false;
    this->m_stopRequested = false;
    this->var_reset_requested = false;
    this->timeStoped = false;
    this->start_time = std::chrono::high_resolution_clock::now();
}

Worker::~Worker()
{
    this->m_stopRequested = true;
    this->m_paused = false;
    this->m_waitCondition.wakeAll();
}

void Worker::startWorker()
{
    qDebug() << "Thread " << QThread::currentThreadId() << ": Worker start";

    if(!this->started) {
        this->result = NULL;
        this->current_time = 0.0;
        this->started = true;
        this->start();
    }

    if(this->timeStoped) {
        this->timeStoped = false;
        this->start_time += std::chrono::high_resolution_clock::now() - this->stop_time;
    } else {
        this->start_time = std::chrono::high_resolution_clock::now();
    }

    this->m_paused = false;
    this->m_waitCondition.wakeAll();
}

void Worker::stopWorker()
{
    qDebug() << "Thread " << QThread::currentThreadId() << ": Worker stop";
    this->m_paused = true;
    this->timeStoped = true;
    this->stop_time = std::chrono::high_resolution_clock::now();
}

void Worker::endWorker()
{
    qDebug() << "Thread " << QThread::currentThreadId() << ": Worker end";
    this->m_paused = true;
    this->timeStoped = false;
    this->var_reset_requested = true;
    this->current_time = 0.0;
}

void Worker::run()
{
    qDebug() << "Thread " << QThread::currentThreadId() << " running now!!!";

    while (!this->m_stopRequested) {
        // stop mechanismus
        m_mutex.lock();
        while (m_paused) {
            m_waitCondition.wait(&m_mutex);
        }
        m_mutex.unlock();

        // process() -> provede jednu iteraci vypoctu
        if(this->process()) {
            // vypocet dokoncen -> zastavi worker + vysle signal
            emit this->workFinished(this->result);
            this->endWorker();
        } else {
            // vysle signaly -> odhadovany cas + aktualni stav
            emit this->estimatedTime(this->getEstimatedTime());
            auto now = std::chrono::high_resolution_clock::now();
            auto diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - this->start_time).count();
            emit this->update(diff_ms/1000.0, this->getProgress());
        }

        // reset internich promennych pokud je to vyzadano
        if(this->var_reset_requested) {
            this->var_reset_requested = false;
            this->varReset();
        }
    }

    qDebug() << "Thread " << QThread::currentThreadId() << " stopped!!!";
}

bool Worker::paused() const
{
    return m_paused;
}
