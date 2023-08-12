#ifndef SIEVEOFERATOSTHENES_H
#define SIEVEOFERATOSTHENES_H

#include "worker.h"
#include <vector>

#define FOR_EACH_PRIME(sres) \
        long long end = sres->size() - 1;\
        for (int i = 2; i <= end; i++) { \
        if ((*sres)[i]) {

#define END_FOR_EACH_PRIME }}

#define FOR_EACH_PRIME_NUMBER i

typedef QVector<bool> * SieveofEratosthenesType;

class SieveofEratosthenes : public Worker
{
public:
    SieveofEratosthenes(long long n);

    long long getN() const;
    void setN(const long long newN);

protected:
    long long n; /** pocet cislic prvocisla */

    virtual void startWorker() override;

    virtual void stopWorker() override;

    virtual void endWorker() override;

    virtual bool process() override;

    virtual float getEstimatedTime() override;

    virtual float getProgress() const override;

private:
    QVector<bool> is_prime;
    long long current_i;

    double done_compute_complexity; /** celkova vykonana vypocetni slozitost */
    double total_compute_complexity; /** celkove vypocetni slozitost */

    virtual void varReset() override;
};

#endif // SIEVEOFERATOSTHENES_H
