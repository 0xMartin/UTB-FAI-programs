#ifndef BIGNUMBER_H
#define BIGNUMBER_H

#include <QVector>

/**
 * @brief Trida umoznujici praci z big numbers
 */
class BigNumber
{
public:
    /**
     * @brief Vytvori big number. Jeho hodnota bude 0
     */
    BigNumber();

    /**
     * @brief Vytvori big number. Jeho hodnota bude odpovidat hodnote zapsane ve stringu
     * @param numberString - String definujici hodnotu cisla
     */
    BigNumber(const std::string& numberString);

    /**
     * @brief Vytvori big number. Jeho hodnota bude odpovidat hodnote ulozene ve vektoru
     * @param digits - Vektor obsahujici cislice. Na prvni pozici cislo s nejnizsi vahou
     */
    BigNumber(QVector<char>& digits);

    /**
     * @brief Vytvori big number. Jeho hodnota bude odpovidat hodnote cisla long long
     * @param num - Cislo long long
     */
    BigNumber(long long num);

    /**
     * @brief Prevede big number na string
     * @return Cislo ve string formatu
     */
    QString toString() const;

    /**
     * @brief Prevede big number na long long
     * @return long long
     */
    long long toLongLong() const;

    /**
     * @brief Navrati pocet cislic
     * @return Pocet cislic
     */
    int numDigits() const;

    /********************************************************************/
    // aritmeticke operace
    BigNumber operator+(const BigNumber& other) const;
    BigNumber operator-(const BigNumber& other) const;
    BigNumber operator*(const BigNumber& other) const;
    BigNumber operator/(const BigNumber& other) const;
    BigNumber operator%(const BigNumber& other) const;

    /********************************************************************/
    // logicke operace
    bool operator ==(const BigNumber &other) const;
    bool operator !=(const BigNumber &other) const;
    bool operator <(const BigNumber &other) const;
    bool operator >(const BigNumber &other) const;
    bool operator <=(const BigNumber &other) const;
    bool operator >=(const BigNumber &other) const;

protected:
    QVector<char> digits; /** Cislice velkeho cisla */
    bool negative; /** True -> cislo je negativni */
};

#endif // BIGNUMBER_H
