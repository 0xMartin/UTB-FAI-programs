#ifndef WORKER_H
#define WORKER_H

#include <QObject>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QtMath>

#define STORE_RESULT(type, value) \
    if(this->result) delete (type*)this->result; \
    this->result = (void *) new type(value)

/**
 * @brief Abstraktni worker trida vykonavajici urcity problem
 */
class Worker : public QThread
{
    Q_OBJECT
public:
    Worker();

    virtual ~Worker();

    /**
     * @brief Spusti worker
     */
    virtual void startWorker();

    /**
     * @brief Zastavi worker
     */
    virtual void stopWorker();

    /**
     * @brief Ukonci worker
     */
    virtual void endWorker();

    /**
     * @brief Aktualni stav o tom zda je worker pozastaven nebo bezi
     * @return Aktualni stav
     */
    bool paused() const;

private:
    /**
     * @brief Reset internich promennych
     */
    virtual void varReset() = 0;

signals:
    /**
     * @brief Signal vyslan ve chvili dokonceni vypoctu
     * @param result - Vysledne cislo
     */
    void workFinished(void * result);

    /**
     * @brief Signal je vysilan v pravidelnych intervalech. Navraci vypocteny odhadovany cas vypoctu
     * @param time - Odhadovany cas
     */
    void estimatedTime(float time);

    /**
     * @brief Update signal je vysilan pravidelne a navraci aktualni cas a aktualni stav vypoctu
     * @param time - Aktulani cas
     * @param progress - Aktualni stav vypoctu v %
     */
    void update(float time, float progress);

protected:
    float current_time; /** Aktulani cas */
    float estimated_time; /** Odhadovany cas vypoctu */

    void * result; /** vysledek vypoctu */

    /**
     * @brief Metoda vlakna - run
     */
    virtual void run() override;

    /**
     * @brief Virtualni metoda obsahujici vypocetni algoritmus
     * @return True -> vypocet dokoncen
     */
    virtual bool process() = 0;

    /**
     * @brief Navrati cas do zbyvajici do konce
     * @return Cas v sekundach
     */
    virtual float getEstimatedTime() = 0;

    /**
     * @brief Navrati aktualni progress
     * @return Progress
     */
    virtual float getProgress() const = 0;

private:
    bool started;

    bool var_reset_requested;

    bool m_stopRequested;
    bool m_paused;

    QMutex m_mutex;
    QWaitCondition m_waitCondition;

protected:
    std::chrono::high_resolution_clock::time_point start_time;
    std::chrono::high_resolution_clock::time_point stop_time;

    bool timeStoped;
};

#endif // WORKER_H
