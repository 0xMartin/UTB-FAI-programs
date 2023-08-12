#include "sieveoferatosthenes.h"

#include <QtMath>

// O(N*LOG(LOG(N)))
#define COMPLEXITY(n) (n * qLn(qLn(n)))

SieveofEratosthenes::SieveofEratosthenes(long long n)
{
    this->current_i = 2;
    this->setN(n);
}

long long SieveofEratosthenes::getN() const
{
    return this->n;
}

void SieveofEratosthenes::setN(const long long newN)
{
    this->n = newN;
    if(this->n < 2) {
        this->n = 2;
    }

    this->is_prime = QVector<bool>(this->n+1, true);
    this->is_prime[0] = false;
    this->is_prime[1] = false;

    // vypocet celkove prace
    this->total_compute_complexity = 0;
    for (long long i = 2; i * i <= this->n; ++i) {
        this->total_compute_complexity += COMPLEXITY(i);
    }
}

void SieveofEratosthenes::startWorker()
{
    Worker::startWorker();
}

void SieveofEratosthenes::stopWorker()
{
    Worker::stopWorker();
}

void SieveofEratosthenes::endWorker()
{
    Worker::endWorker();
}

bool SieveofEratosthenes::process()
{
    if(this->is_prime[this->current_i]) {
        for (long long j = this->current_i * this->current_i;
            j <= n;
            j += this->current_i) {
            this->is_prime[j] = false;
        }
    }
    this->current_i++;

    // casove omezeni
    this->msleep(100);

    if(this->current_i * this->current_i <= this->n) {
        return false;
    } else {
        STORE_RESULT(QVector<bool>, this->is_prime);
        return true;
    }
}

float SieveofEratosthenes::getEstimatedTime()
{
    // vypocet celkove vykonane prace
    this->done_compute_complexity = 0;
    for (long long i = 2; i <= this->current_i; ++i) {
        this->done_compute_complexity += COMPLEXITY(i);
    }

    // prevod vypocetni slozitosti na cas
    auto now = std::chrono::high_resolution_clock::now();
    auto diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - this->start_time).count();
    float remaining_time = (float) diff_ms * (total_compute_complexity - done_compute_complexity) / (float) done_compute_complexity;

    return remaining_time / 1000.0;
}

float SieveofEratosthenes::getProgress() const
{
    return (float) this->done_compute_complexity / this->total_compute_complexity * 100.0;
}

void SieveofEratosthenes::varReset()
{
    this->current_i = 2;
    this->is_prime = QVector<bool>(this->n+1, true);
    this->is_prime[0] = false;
    this->is_prime[1] = false;
}
