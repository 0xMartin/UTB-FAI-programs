#include <iostream>
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <omp.h>

using namespace std;
namespace mp = boost::multiprecision;

int main()
{
    unsigned long long N;
    int num_threads;
    mp::cpp_dec_float_100 pi = 0.0;

    // nastaveni N (pocet iteraci vypoctu cisla pi)
    cout << "Zadejte hodnotu N: ";
    cin >> N;

    // zadani poctu procesu kterych bude pouzito pri vypoctu (omezeno na moznost nastaveni od 1 po pocet vlaken procesu)
    int max_threads = omp_get_max_threads();
    do {
        cout << "Zadejte počet použitých procesorů (1 - " << max_threads << "): ";
        cin >> num_threads;
    } while (num_threads < 1 || num_threads > max_threads);

    // nastaveni poctu vlaken
    omp_set_num_threads(num_threads);


    // pocatecni cas pro mereni delky vypoctu jednotlivych vlaken
    double start_time = omp_get_wtime();

    // vlastni redukce pro prommenou pi (boost/multiprecision/cpp_dec_float.hpp)
    mp::cpp_dec_float_100 global_sum = 0.0;
    #pragma omp declare reduction(custom_reduction : mp::cpp_dec_float_100 : omp_out += omp_in) initializer(omp_priv = 0.0)

    // paralelni vypocet pi
    #pragma omp parallel for reduction(custom_reduction : global_sum)
    for (int i = 0; i < N; i++) {
        mp::cpp_dec_float_100 x = (i + 0.5) / N;
        global_sum += 4.0 / (1.0 + x * x);
    }

    pi = global_sum / N;


    double end_time = omp_get_wtime();

    // Výpis výsledku
    cout << "Cas vypoctu: " << (end_time - start_time) << " sec" << endl;
    cout << setprecision(numeric_limits<mp::cpp_dec_float_100>::digits10 + 1);
    cout << "PI = " << pi << endl;

    return 0;
}
