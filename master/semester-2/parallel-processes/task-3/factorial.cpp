#include "factorial.h"
#include "bignumber.h"

// O(N*N)
#define COMPLEXITY(n) (n * n)

Factorial::Factorial(BigNumber n)
{
    this->setN(n);
    this->current_n = 1;
    this->fatorial = 1;
    this->done_compute_complexity = 0;
    this->total_compute_complexity = 0;
}

BigNumber Factorial::getN() const
{
    return this->n;
}

void Factorial::setN(const BigNumber &newN)
{
    this->n = newN;
    if(this->n < 0) {
        this->n = 0;
    }

    // vypocet celkove prace
    this->total_compute_complexity = 0;
    long long end = n.toLongLong();
    long digit_count = 1;
    for (long long i = 1; i <= end; ++i) {
         // pocet cislic v aktualnim i
        if(i % 10 == 0) {
            ++digit_count;
        }
        this->total_compute_complexity += COMPLEXITY(digit_count);
    }
}

void Factorial::startWorker()
{
    Worker::startWorker();
}

void Factorial::stopWorker()
{
    Worker::stopWorker();
}

void Factorial::endWorker()
{
    Worker::endWorker();
}

bool Factorial::process()
{
    if(this->current_n <= this->n) {
        this->fatorial = this->fatorial * this->current_n;
        this->current_n = this->current_n + 1;
        return false;
    } else {
        STORE_RESULT(BigNumber, this->fatorial);
        return true;
    }
}

float Factorial::getEstimatedTime()
{
    // vypocet celkove vykonane prace
    this->done_compute_complexity = 0;
    long long end = current_n.toLongLong();
    long digit_count = 1;
    for (long long i = 1; i <= end; ++i) {
        // pocet cislic v aktualnim i
        if(i % 10 == 0) {
            ++digit_count;
        }
        this->done_compute_complexity += COMPLEXITY(digit_count);
    }

    // prevod vypocetni slozitosti na cas
    auto now = std::chrono::high_resolution_clock::now();
    auto diff_ms = std::chrono::duration_cast<std::chrono::milliseconds>(now - this->start_time).count();
    float remaining_time = (float) diff_ms * (total_compute_complexity - done_compute_complexity) / (float) done_compute_complexity;

    return remaining_time / 1000.0;
}

float Factorial::getProgress() const
{
    return (float) this->done_compute_complexity / this->total_compute_complexity * 100.0;
}

void Factorial::varReset()
{
    this->current_n = 1;
    this->fatorial = 1;
}
