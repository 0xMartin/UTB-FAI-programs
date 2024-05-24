#ifndef __ARGUMENT_PARSER_H__
#define __ARGUMENT_PARSER_H__

#include <stdlib.h>
#include <vector>

#define ARGUMENT_START_SYMBOL '-'

namespace ArgumentParser
{

    /// @brief Argument predany programu. Sklada se vzdy z nazvu a hodnoty. Napriklad: -i input_file.png
    struct Argument
    {
        char name[16];
        char value[256];
    };

    /// @brief Zpracuje vstupni argumenty programu
    /// @param argc Pocet argumentu
    /// @param args Argumenty predane pri spousteni programu
    /// @return True v pripade uspesneho zpracovani
    extern bool parseArguments(std::vector<Argument> &arguments, int argc, char *args[]);

    /// @brief Najde argument v poli argumentu podle jeho jmena
    /// @param name Cele jmeno argumentu (Do jmena je zahrnut i symbol "-")
    /// @param output Vystupni argument
    /// @return True pokud je argument v poli nalezen
    extern bool findByName(std::vector<Argument> &arguments, ArgumentParser::Argument & output, const char * name);

}

#endif