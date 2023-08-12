#ifndef FACTORIAL_H
#define FACTORIAL_H

#include "worker.h"

#include "bignumber.h"

typedef BigNumber* FactorialType;

class Factorial : public Worker
{
public:
    Factorial(BigNumber n);

    BigNumber getN() const;
    void setN(const BigNumber &newN);

protected:
    BigNumber fatorial; /** Vysledny faktorial */
    BigNumber n; /** N pro ktere pocitame faktorial */
    BigNumber current_n; /** Aktualni N */

    virtual void startWorker() override;

    virtual void stopWorker() override;

    virtual void endWorker() override;

    virtual bool process() override;

    virtual float getEstimatedTime() override;

    virtual float getProgress() const override;

private:
    long long done_compute_complexity; /** celkova vykonana vypocetni slozitost */
    long long total_compute_complexity; /** celkove vypocetni slozitost */

    virtual void varReset() override;
};

#endif // FACTORIAL_H
