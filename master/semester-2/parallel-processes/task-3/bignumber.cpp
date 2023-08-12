#include "bignumber.h"

#define BASE 10

#include <QtMath>

BigNumber::BigNumber()
{
    this->negative = false;
    this->digits.push_back(0);
}

BigNumber::BigNumber(const std::string &numberString)
{
    this->negative = false;
    for(int i = numberString.size() - 1; i >= 0; i--) {
        if(numberString[i] == '-') {
            this->negative = true;
            break;
        }
        this->digits.push_back(numberString[i] - '0');
    }
    if(digits.empty()) {
        this->digits.push_back(0);
    }
}

BigNumber::BigNumber(QVector<char> &digits)
{
    this->negative = false;
    for(int digit : digits) {
        this->digits.push_back(digit);
    }
}

BigNumber::BigNumber(long long num) {
    this->negative = num < 0;
    num = num < 0 ? -num : num;
    while (num > 0) {
        digits.push_back(num % 10);
        num /= 10;
    }
    if (digits.empty()) {
        digits.push_back(0);
    }
}

QString BigNumber::toString() const
{
    QString res = "";

    if(this->negative) {
        res += "-";
    }

    for(int i = digits.size() - 1; i >= 0; i--) {
        res += QChar(digits[i] + '0');
    }

    return res;
}

long long BigNumber::toLongLong() const
{
    long long res = 0;

    for(int i = 0; i < digits.size(); i++) {
        res += (int)(digits[i]) * qPow(10, i);
    }

    if(this->negative) {
        res *= -1;
    }

    return res;
}

int BigNumber::numDigits() const
{
    return this->digits.count();
}

BigNumber BigNumber::operator+(const BigNumber &other) const
{
    const QVector<char>& a = this->digits.size() > other.digits.size() ? this->digits : other.digits;
    const QVector<char>& b = this->digits.size() > other.digits.size() ? other.digits : this->digits;

    BigNumber result;
    result.digits.resize(a.size());

    int carry = 0;
    for(int i = 0; i < a.size(); i++) {
        int sum = a[i] + (i < b.size() ? b[i] : 0) + carry;
        result.digits[i] = sum % 10;
        carry = sum / 10;
    }
    if (carry > 0) {
        result.digits.push_back(carry);
    }

    return result;
}

BigNumber BigNumber::operator-(const BigNumber &other) const
{
    if(*this < other) {
        BigNumber b = other - *this;
        b.negative = true;
        return b;
    }

    QVector<char> resultDigits;
    int borrow = 0;

    int numDigits = digits.size();
    int otherNumDigits = other.digits.size();

    int i;
    for(i = 0; i < numDigits && i < otherNumDigits; ++i) {
        int diff = digits[i] - borrow - other.digits[i];
        if(diff < 0) {
            diff += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        resultDigits.append(diff);
    }

    while(i < numDigits) {
        int diff = digits[i] - borrow;
        if(diff < 0) {
            diff += 10;
            borrow = 1;
        } else {
            borrow = 0;
        }
        resultDigits.append(diff);
        ++i;
    }

    while(resultDigits.size() > 1 && resultDigits.back() == 0) {
        resultDigits.pop_back();
    }

    return BigNumber(resultDigits);
}

BigNumber BigNumber::operator*(const BigNumber &other) const
{
    QVector<char> resultDigits(digits.size() + other.digits.size(), 0);

    for (int i = 0; i < digits.size(); ++i) {
        int carry = 0;
        for (int j = 0; j < other.digits.size(); ++j) {
            int product = digits[i] * other.digits[j] + carry + resultDigits[i+j];
            carry = product / 10;
            resultDigits[i+j] = product % 10;
        }
        if (carry > 0) {
            resultDigits[i + other.digits.size()] += carry;
        }
    }

    while (resultDigits.size() > 1 && resultDigits.back() == 0) {
        resultDigits.pop_back();
    }

    BigNumber res = BigNumber(resultDigits);
    res.negative = (this->negative && !other.negative) || (!this->negative && other.negative);
    return res;
}

BigNumber BigNumber::operator/(const BigNumber &other) const
{
    if (other == 0) {
        qFatal("Divide by zero exception.");
    }
    if (*this < other) {
        return BigNumber(0);
    }
    if (other == 1) {
        return *this;
    }

    QVector<char> result(digits.size());
    BigNumber remainder(0);
    for (int i = digits.size() - 1; i >= 0; i--) {
        remainder = remainder * BASE + digits[i];
        int j = 0;
        int k = BASE - 1;
        while (j < k) {
            int mid = (j + k + 1) / 2;
            if (other * mid > remainder) {
                k = mid - 1;
            } else {
                j = mid;
            }
        }
        result[i] = j;
        remainder = remainder - other * j;
    }

    while (result.size() > 1 && result.back() == 0) {
        result.pop_back();
    }

    BigNumber res = BigNumber(result);
    res.negative = (this->negative && !other.negative) || (!this->negative && other.negative);
    return res;
}

BigNumber BigNumber::operator%(const BigNumber &other) const
{
    if (other == 0) {
        throw std::invalid_argument("Division by zero");
    }
    if (*this < other) {
        return *this;
    }

    BigNumber quotient;
    BigNumber remainder = *this;

    while (remainder >= other) {
        int numZeros = remainder.numDigits() - other.numDigits();
        BigNumber temp = other * pow(10, numZeros);
        while (temp > remainder) {
            temp = temp / 10;
            numZeros--;
        }
        quotient = quotient + pow(10, numZeros);
        remainder = remainder - temp;
    }
    return remainder;
}

bool BigNumber::operator==(const BigNumber& other) const {
    return digits == other.digits;
}

bool BigNumber::operator!=(const BigNumber& other) const {
    return digits != other.digits;
}

bool BigNumber::operator<(const BigNumber& other) const {
    if (digits.size() != other.digits.size()) {
        return digits.size() < other.digits.size();
    }
    for (int i = digits.size() - 1; i >= 0; i--) {
        if (digits[i] != other.digits[i]) {
            return digits[i] < other.digits[i];
        }
    }
    return false;
}

bool BigNumber::operator>(const BigNumber& other) const {
    if (digits.size() != other.digits.size()) {
        return digits.size() > other.digits.size();
    }
    for (int i = digits.size() - 1; i >= 0; i--) {
        if (digits[i] != other.digits[i]) {
            return digits[i] > other.digits[i];
        }
    }
    return false;
}

bool BigNumber::operator<=(const BigNumber& other) const {
    return (*this < other) || (*this == other);
}

bool BigNumber::operator>=(const BigNumber& other) const {
    return (*this > other) || (*this == other);
}
