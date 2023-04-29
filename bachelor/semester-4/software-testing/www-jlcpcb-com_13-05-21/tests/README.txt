Automatizovane testy pro webovou stranku jlcpcb.com
========================================================================================================================
*Testy byly provedeny na systemu Kali Linux, kernel release: 5.10.0-kali7-amd64
*Pouzity prohlizec: Google Chrome 90.0.4430.93
*Python 3.8.3
========================================================================================================================
*Spousteni testu:
    [Linux]
        1. spusteni jedne zvolene testovaci sady:
            ./run.sh <nazev testovaci sady>
            *priklad -> ./run.sh TS_03
            *nebo i bez vstupniho argumentu (nazev bude nacten z konzole): ./run.sh
        2. spusteni vsech testovacich sad:
            ./run_all.sh
    [Windows]
        1. spusteni vsech testovacich sad:
            run_all_windows.bat
========================================================================================================================
*Popis:
    Kazda testovaci sada ma svuj vlastni adresar (TS_03, TS_04, ...). V kazdem z techto adresaru je robot file pro
    tuto testovaci sadu, output, report a logy. Screenshoty pro vsechny TS jsou ve slozce "screenshots". Ve slozce
    "resources" jsou soubory potrebne pro testy. Slozka library obsahuje dodatecne knihovny.

    Dalsi Soubory:
        page_*          ->      potrebne promenne pro danou stranku
        keywords.robot  ->      keywordy pouzivane ve vsech TS
        settings.robot  ->      globalni nastaveni
========================================================================================================================
Testovaci sady:
    TS_03   ->  Tato testovaci sada overi spravnou funkcnost navigace na webove strance jlcpcb.com
    TS_04   ->  Testovaci sada pro overeni spravneho zobrazovani UI domovske stranky
    TS_05   ->  Tato sada overi zakladni funkcnost systemu pro vytvareni objednavky noveho PCB
    TS_06   ->  Tato sada overi zakladni funkcnost systemu pro vytvareni objednavky SMT-Stencil
    TS_07   ->  Tato testovaci sada overi, zakladni funkcnost knihovny soucastek (filtrovani, vyhledavani,
                zobrazovani kategorii)
========================================================================================================================