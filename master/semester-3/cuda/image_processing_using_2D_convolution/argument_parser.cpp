#include "argument_parser.h"

#include <string.h>
#include <iostream>

namespace ArgumentParser
{

    bool parseArguments(std::vector<Argument> &arguments, int argc, char *args[])
    {
        if (argc == 0 || args == nullptr)
        {
            return false;
        }

        arguments.clear();

        bool nameIsEmpty = true;
        ArgumentParser::Argument buffer;

        for (int index = 1; index < argc; ++index)
        {

            // hodnota argumentu s pojemenovanim
            if (!nameIsEmpty)
            {
                nameIsEmpty = true;
                strncpy(buffer.value, args[index], 256);
                arguments.push_back(buffer);
                continue;
            }

            // nazev, vzdy zacina symbolem "-" a zaroven nejde o hodnotu argumentu
            if (nameIsEmpty && args[index][0] == ARGUMENT_START_SYMBOL)
            {
                strncpy(buffer.name, args[index], 16);
                nameIsEmpty = false;
            }
            else
            {
                // argument bez pojmenovani
                buffer.name[0] = 0x0;
                strncpy(buffer.value, args[index], 64);
                arguments.push_back(buffer);
            }
        }

        // posledni agrument bez hodnoty
        if (!nameIsEmpty)
        {
            buffer.value[0] = 0x0;
            arguments.push_back(buffer);
        }

        return true;
    }

    bool findByName(std::vector<Argument> &arguments, ArgumentParser::Argument &output, const char *name)
    {
        for (ArgumentParser::Argument &arg : arguments)
        {
            if (strcmp(arg.name, name) == 0)
            {
                output = arg;
                return true;
            }
        }

        return false;
    }

}
