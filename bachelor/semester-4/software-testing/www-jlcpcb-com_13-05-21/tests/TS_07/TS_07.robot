*** Settings ***
Library  SeleniumLibrary  run_on_failure=Nothing
Documentation  Tato testovaci sada overi, zakladni funkcnost knihovny soucastek (filtrovani, vyhledavani, zobrazovani kategorii)
Metadata  Author  Martin Krcma

Resource  ../settings.robot
Resource  ../keywords.robot
Resource  ../page_smt_parts_lib.robot

Suite Setup  TS_07 Precondition
Suite Teardown  TS_07 Postcondition


*** Variables ***
${TC_02_CATEGORY}  Resistors

${TC_03_SUB_CATEGORY}  Switching Diode

${TC_04_MANUFACTURER}  TEXAS INSTRUMENTS

${TC_05_FILTER_NOT_ALLOWED}  not-allowed

${TC_06_SEARCH_WORD}  resistor
${TC_06_SEARCH_RESULT}  RESISTOR

${TC_08_SEARCH_WORD}  PANASONIC

${TC_09_CATEGORY}  SMD Crystal Resonators


*** Test Cases ***
TS_07-01 – Zobrazeni kategorii soucastek
    [Teardown]  Capture Page Screenshot  TS_07-01${SCREENSHOT_NAME_POSTFIX}
    Scroll And Check Visibility  ${PAGE_STM_PARTS_LIB_AMPLIFIERS_XPATH}
    Scroll And Check Visibility  ${PAGE_STM_PARTS_LIB_DIODER_XPATH}
    Scroll And Check Visibility  ${PAGE_STM_PARTS_LIB_FILTERS_XPATH}
    Scroll And Check Visibility  ${PAGE_STM_PARTS_LIB_CONNECTORS_XPATH}
    Scroll And Check Visibility  ${PAGE_STM_PARTS_LIB_CAPACITORS_XPATH}

TS_07-02 – Vstoupeni do kategorie „Resistors“
    [Teardown]  Run Keywords
    ...  Log Location
    ...  AND  Go To And Wait  ${PAGE_STM_PARTS_LIB_URL}  ${PAGE_STM_PARTS_LIB_WAIT_FOR_ELEMENT_XPATH}
    Scroll And Check Visibility  ${PAGE_STM_PARTS_LIB_RESISTORS_XPATH}
    Click On Element And Wait  ${PAGE_STM_PARTS_LIB_RESISTORS_XPATH}  ${PAGE_STM_PARTS_LIB_ACTIVE_CATEGORY_XPATH}
    Element Should Contain  ${PAGE_STM_PARTS_LIB_ACTIVE_CATEGORY_XPATH}  ${TC_02_CATEGORY}

TS_07-03 – Vstoupeni do podkategorie „Switching Diode“
    [Teardown]  Run Keywords
    ...  Log Location
    ...  AND  Go To And Wait  ${PAGE_STM_PARTS_LIB_URL}  ${PAGE_STM_PARTS_LIB_WAIT_FOR_ELEMENT_XPATH}
    Scroll And Check Visibility  ${PAGE_STM_PARTS_LIB_RESISTORS_XPATH}
    Click On Element And Wait  ${PAGE_STM_PARTS_LIB_DIODER_XPATH}  ${PAGE_STM_PARTS_LIB_SW_DIODES_XPATH}
    Click On Element And Wait  ${PAGE_STM_PARTS_LIB_SW_DIODES_XPATH}  ${PAGE_STM_PARTS_LIB_ACTIVE_SUB_CATEGORY_XPATH}
    Element Should Contain  ${PAGE_STM_PARTS_LIB_ACTIVE_SUB_CATEGORY_XPATH}  ${TC_03_SUB_CATEGORY}

TS_07-04 – Overeni funkcnosti vyhledavaciho filtru
    [Teardown]  Run Keywords
    ...  Capture Page Screenshot  TS_07-04${SCREENSHOT_NAME_POSTFIX}
    ...  AND  Go To And Wait  ${PAGE_STM_PARTS_LIB_URL}  ${PAGE_STM_PARTS_LIB_WAIT_FOR_ELEMENT_XPATH}
    Scroll And Check Visibility  ${PAGE_STM_PARTS_LIB_AMPLIFIERS_XPATH}
    Click On Element And Wait  ${PAGE_STM_PARTS_LIB_AMPLIFIERS_XPATH}  ${PAGE_STM_PARTS_LIB_FILTER_TEXAS_INS_XPATH}
    Click On  ${PAGE_STM_PARTS_LIB_FILTER_TEXAS_INS_XPATH}
    Click On  ${PAGE_STM_PARTS_LIB_FILTER_APPLY_XPATH}
    Scroll Element Into View  ${PAGE_STM_PARTS_RESULT_3_XPATH}
    Element Should Contain  ${PAGE_STM_PARTS_RESULT_1_XPATH}  ${TC_04_MANUFACTURER}
    Element Should Contain  ${PAGE_STM_PARTS_RESULT_2_XPATH}  ${TC_04_MANUFACTURER}
    Element Should Contain  ${PAGE_STM_PARTS_RESULT_3_XPATH}  ${TC_04_MANUFACTURER}

TS_07-05 – Overeni resetovani filtru
    [Teardown]  Go To And Wait  ${PAGE_STM_PARTS_LIB_URL}  ${PAGE_STM_PARTS_LIB_WAIT_FOR_ELEMENT_XPATH}
    Scroll And Check Visibility  ${PAGE_STM_PARTS_LIB_AMPLIFIERS_XPATH}
    Click On Element And Wait  ${PAGE_STM_PARTS_LIB_AMPLIFIERS_XPATH}  ${PAGE_STM_PARTS_LIB_FILTER_TEXAS_INS_XPATH}
    Click On  ${PAGE_STM_PARTS_LIB_FILTER_TEXAS_INS_XPATH}
    Click On  ${PAGE_STM_PARTS_LIB_FILTER_RESET_XPATH}
    Element Should Not Have Class  ${PAGE_STM_PARTS_LIB_FILTER_TEXAS_INS_XPATH}  ${TC_05_FILTER_NOT_ALLOWED}

TS_07-06 – Overeni vyhledani soucastek s nazvem resistor
    [Teardown]  Run Keywords
    ...  Capture Page Screenshot  TS_07-06${SCREENSHOT_NAME_POSTFIX}
    ...  AND  Go To And Wait  ${PAGE_STM_PARTS_LIB_URL}  ${PAGE_STM_PARTS_LIB_WAIT_FOR_ELEMENT_XPATH}
    Input Text  ${PAGE_STM_PARTS_SEARCH_INPUT_XPATH}  ${TC_06_SEARCH_WORD}
    Click On Element And Wait  ${PAGE_STM_PARTS_SEARCH_BTN_XPATH}  ${PAGE_STM_PARTS_LIB_ACTIVE_CATEGORY_XPATH}
    Element Should Contain  ${PAGE_STM_PARTS_RESULT_1_XPATH}  ${TC_06_SEARCH_RESULT}
    Element Should Contain  ${PAGE_STM_PARTS_RESULT_2_XPATH}  ${TC_06_SEARCH_RESULT}
    Element Should Contain  ${PAGE_STM_PARTS_RESULT_3_XPATH}  ${TC_06_SEARCH_RESULT}

#neprojde (vystup stranky neodpovida ocekavanemu vysledku)
TS_07-07 - Vyhledavani s prazdnym polem
    [Setup]  Clear Element Text  ${PAGE_STM_PARTS_SEARCH_INPUT_XPATH}
    [Teardown]  Run Keywords
    ...  Capture Page Screenshot  TS_07-07${SCREENSHOT_NAME_POSTFIX}
    ...  AND  Go To And Wait  ${PAGE_STM_PARTS_LIB_URL}  ${PAGE_STM_PARTS_LIB_WAIT_FOR_ELEMENT_XPATH}
    Click On Element And Wait  ${PAGE_STM_PARTS_SEARCH_BTN_XPATH}  ${PAGE_STM_PARTS_LIB_ACTIVE_CATEGORY_XPATH}
    Page Should Not Contain Element  ${PAGE_STM_PARTS_RESULT_1_XPATH}

TS_07-08 – Overeni vyhledani soucastek s nazvem resistor
    [Teardown]  Run Keywords
    ...  Capture Page Screenshot  TS_07-08${SCREENSHOT_NAME_POSTFIX}
    ...  AND  Go To And Wait  ${PAGE_STM_PARTS_LIB_URL}  ${PAGE_STM_PARTS_LIB_WAIT_FOR_ELEMENT_XPATH}
    Input Text  ${PAGE_STM_PARTS_SEARCH_INPUT_XPATH}  ${TC_08_SEARCH_WORD}  True
    Click On Element And Wait  ${PAGE_STM_PARTS_SEARCH_BTN_XPATH}  ${PAGE_STM_PARTS_LIB_ACTIVE_CATEGORY_XPATH}
    Element Should Contain  ${PAGE_STM_PARTS_RESULT_1_XPATH}  ${TC_08_SEARCH_WORD}
    Element Should Contain  ${PAGE_STM_PARTS_RESULT_2_XPATH}  ${TC_08_SEARCH_WORD}
    Element Should Contain  ${PAGE_STM_PARTS_RESULT_3_XPATH}  ${TC_08_SEARCH_WORD}

TS_07-09 – Vstup do podkategorie z hlavni nabidky kategorii
    [Teardown]  Capture Page Screenshot  TS_07-09${SCREENSHOT_NAME_POSTFIX}
    Scroll Element Into View  ${PAGE_STM_PARTS_LIB_CRYSTALS_XPATH}
    Set Focus To Element  ${PAGE_STM_PARTS_LIB_CRYSTALS_XPATH}
    Mouse Over  ${PAGE_STM_PARTS_LIB_CRYSTALS_XPATH}
    Click On Element And Wait  ${PAGE_STM_PARTS_LIB_CRYSTAL_RESONATORS_XPATH}  ${PAGE_STM_PARTS_LIB_ACTIVE_CATEGORY_XPATH}
    Element Should Contain  ${PAGE_STM_PARTS_LIB_ACTIVE_SUB_CATEGORY_XPATH}  ${TC_09_CATEGORY}

*** Keywords ***
#TS precondition
TS_07 Precondition
    Open Browser With URL  ${BROWSER}  ${PAGE_STM_PARTS_LIB_URL}  ${PAGE_STM_PARTS_LIB_WAIT_FOR_ELEMENT_XPATH}
    Init Tests
    Click On  ${PAGE_ORDER_ACCEPT_COOKIES_XPATH}

#TS postcondition
TS_07 Postcondition
    Close Browser

Scroll And Check Visibility
    [Arguments]  ${element}
    Scroll Element Into View  ${element}
    Element Should Be Visible  ${element}
