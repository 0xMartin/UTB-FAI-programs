*** Settings ***
Library  SeleniumLibrary  run_on_failure=Nothing
Documentation  Tato sada overi zakladni funkcnost systemu pro vytvareni objednavky SMT-Stencil
Metadata  Author  Martin Krcma

Resource  ../settings.robot
Resource  ../keywords.robot
Resource  ../page_order.robot

Suite Setup  TS_06 Precondition
Suite Teardown  TS_06 Postcondition


*** Variables ***
${TC_03_DIMENSION}  460*460(Valid area 380mm*430mm, $20.00 ,1.95kg)
${TC_03_PRICE}  $20.00
${TC_03_WEIGHT}  1.95kg

${TC_04_QTY}  5
${TC_04_PRICE}  $35.23
${TC_04_WEIGHT}  4.50kg

${TC_05_QTY}  -2
${TC_05_QTY_EXPECTED}  2


*** Test Cases ***
TS_06-03 – Vyber rozmeru sablony
    [Setup]   Click On  ${PAGE_ORDER_SMT_STENCIL_TYPE_XPATH}
    [Teardown]  Reload Page And Wait  ${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}
    Select From List By Label  ${PAGE_ORDER_SMT_STENCIL_DIMENSION_XPATH}  ${TC_03_DIMENSION}
    Sleep  2
    Element Should Contain  ${PAGE_ORDER_PRICE_XPATH}  ${TC_03_PRICE}
    Element Should Contain  ${PAGE_ORDER_WEIGHT_XPATH}  ${TC_03_WEIGHT}

TS_06-04 – Zmena poctu kusu
    [Setup]   Click On  ${PAGE_ORDER_SMT_STENCIL_TYPE_XPATH}
    [Teardown]  Reload Page And Wait  ${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}
    Press Keys  ${PAGE_ORDER_SMT_STENCIL_QTY_XPATH}  \DELETE
    Input Text  ${PAGE_ORDER_SMT_STENCIL_QTY_XPATH}  ${TC_04_QTY}
    Sleep  2
    Click On  ${PAGE_ORDER_PRICE_XPATH}
    Element Should Contain  ${PAGE_ORDER_PRICE_XPATH}  ${TC_04_PRICE}
    Element Should Contain  ${PAGE_ORDER_WEIGHT_XPATH}  ${TC_04_WEIGHT}

TS_06-05 – Zadani zaporneho poctu kusu
    [Setup]   Click On  ${PAGE_ORDER_SMT_STENCIL_TYPE_XPATH}
    Press Keys  ${PAGE_ORDER_SMT_STENCIL_QTY_XPATH}  \DELETE
    Input Text  ${PAGE_ORDER_SMT_STENCIL_QTY_XPATH}  ${TC_05_QTY}
    Sleep  2
    Textfield Value Should Be  ${PAGE_ORDER_SMT_STENCIL_QTY_XPATH}  ${TC_05_QTY_EXPECTED}

*** Keywords ***
#TS precondition
TS_06 Precondition
    Open Browser With URL  ${BROWSER}  ${PAGE_ORDER_URL}  ${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}
    Init Tests
    Click On  ${PAGE_ORDER_ACCEPT_COOKIES_XPATH}

#TS postcondition
TS_06 Postcondition
    Close Browser
