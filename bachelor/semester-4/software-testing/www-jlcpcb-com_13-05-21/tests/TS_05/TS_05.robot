*** Settings ***
Library  SeleniumLibrary  run_on_failure=Nothing
Library  ../library/file.py
Documentation  Tato sada overi zakladni funkcnost systemu pro vytvareni objednavky noveho PCB
Metadata  Author  Martin Krcma

Resource  ../settings.robot
Resource  ../keywords.robot
Resource  ../page_order.robot

Suite Setup  TS_05 Precondition
Suite Teardown  TS_05 Postcondition


*** Variables ***
${VALID_PCB}  ${RESOURCE_DIR}/valid_pcb.zip
${INVALID_PCB}  ${RESOURCE_DIR}/invalid_format_pcb.data

${TC_01_HEIGHT}  72
${TC_01_WIDTH}  100

${TC_03_PRICE}  $8.00

${TC_04_HEIGHT}  50
${TC_04_WIDTH}  50

${TC_05_HEIGHT}  -50
${TC_05_WIDTH}  -50
${TC_05_HEIGHT_EXPECTED}  50
${TC_05_WIDTH_EXPECTED}  50

${TC_06_HEIGHT}  600
${TC_06_WIDTH}  50

${TC_07_CURRENT_COLOR_CLASS}  cur

${TC_08_PRICE}  $38.90

${TC_10_QTY}  100
${TC_10_PRICE}  $73.00


*** Test Cases ***
TS_05-01 – Upload souboru s navrhem PCB
    [Setup]  File Check  ${VALID_PCB}
    [Teardown]  Run Keywords
    ...  Capture Page Screenshot  TS_05-01${SCREENSHOT_NAME_POSTFIX}
    ...  AND  Reload Page And Wait  ${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}
    Upload file  ${VALID_PCB}
    Wait Until Element Is Visible  ${PAGE_ORDER_PCB_DESIGN_VIEW_XPATH}  100s
    Element Attribute Value Should Be  ${PAGE_ORDER_HEIGHT_XPATH}  value  ${TC_01_HEIGHT}
    Element Attribute Value Should Be  ${PAGE_ORDER_WIDTH_XPATH}  value  ${TC_01_WIDTH}

TS_05-02 – Upload souboru s neplatnym formatem
    [Setup]  File Check  ${INVALID_PCB}
    [Teardown]  Reload Page And Wait  ${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}
    Upload file  ${INVALID_PCB}
    Wait Until Element Is Visible  ${PAGE_ORDER_INCORRECT_FORMAT_DIALOG_XPATH}  20s

TS_05-03 – Zmena poctu vrstev PCB
    [Teardown]  Reload Page And Wait  ${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}
    Click On  ${PAGE_ORDER_4_LAYERS_XPATH}
    Sleep  2
    Element Should Contain  ${PAGE_ORDER_PRICE_XPATH}  ${TC_03_PRICE}

TS_05-04 – Zmena rozmeru PCB
    [Teardown]  Reload Page And Wait  ${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}
    Input Text  ${PAGE_ORDER_HEIGHT_XPATH}  ${TC_04_HEIGHT}  True
    Input Text  ${PAGE_ORDER_WIDTH_XPATH}  ${TC_04_WIDTH}  True
    Select From List By Label  ${PAGE_ORDER_UNITS_XPATH}  mm
    Element Should Not Be Visible  ${PAGE_ORDER_DIM_EXCEED_XPATH}

TS_05-05 – Vlozeni zapornych rozmeru PCB
    [Teardown]  Reload Page And Wait  ${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}
    Input Text  ${PAGE_ORDER_HEIGHT_XPATH}  ${TC_05_HEIGHT}  True
    Input Text  ${PAGE_ORDER_WIDTH_XPATH}  ${TC_05_WIDTH}  True
    Textfield Value Should Be  ${PAGE_ORDER_HEIGHT_XPATH}  ${TC_05_HEIGHT_EXPECTED}
    Textfield Value Should Be  ${PAGE_ORDER_WIDTH_XPATH}  ${TC_05_WIDTH_EXPECTED}

TS_05-06 – Vloženi velikych rozmeru PCB
    [Teardown]  Reload Page And Wait  ${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}
    Input Text  ${PAGE_ORDER_HEIGHT_XPATH}  ${TC_06_HEIGHT}  True
    Input Text  ${PAGE_ORDER_WIDTH_XPATH}  ${TC_06_WIDTH}  True
    Select From List By Label  ${PAGE_ORDER_UNITS_XPATH}  mm
    Element Should Be Visible  ${PAGE_ORDER_DIM_EXCEED_XPATH}

TS_05-07 – Zmena barvy PCB
    [Teardown]  Reload Page And Wait  ${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}
    Click On  ${PAGE_ORDER_COLOR_BLUE_XPATH}
    Sleep  1
    Element Should Have Class  ${PAGE_ORDER_COLOR_BLUE_XPATH}  ${TC_07_CURRENT_COLOR_CLASS}
    Element Should Not Have Class  ${PAGE_ORDER_COLOR_GREEN_XPATH}  ${TC_07_CURRENT_COLOR_CLASS}
    Element Should Not Have Class  ${PAGE_ORDER_COLOR_RED_XPATH}  ${TC_07_CURRENT_COLOR_CLASS}
    Element Should Not Have Class  ${PAGE_ORDER_COLOR_YELLOW_XPATH}  ${TC_07_CURRENT_COLOR_CLASS}
    Element Should Not Have Class  ${PAGE_ORDER_COLOR_WHITE_XPATH}  ${TC_07_CURRENT_COLOR_CLASS}
    Element Should Not Have Class  ${PAGE_ORDER_COLOR_BLACK_XPATH}  ${TC_07_CURRENT_COLOR_CLASS}

TS_05-08 – Zmena tloustky PCB
    [Teardown]  Reload Page And Wait  ${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}
    Click On  ${PAGE_ORDER_THICKNESS_2_XPATH}
    Sleep  2
    Element Should Contain  ${PAGE_ORDER_PRICE_XPATH}  ${TC_08_PRICE}

TS_05-09 – Zobrazeni pokrocilych moznosti
    [Teardown]  Reload Page And Wait  ${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}
    Click On  ${PAGE_ORDER_ACCEPT_COOKIES_XPATH}
    Scroll Element Into View  ${PAGE_ORDER_ADVANCED_XPATH}
    Click On  ${PAGE_ORDER_ADVANCED_XPATH}
    Scroll Element Into View   ${PAGE_ORDER_ADVANCED_OPT2_XPATH}
    Element Should Be Visible  ${PAGE_ORDER_ADVANCED_OPT2_XPATH}

TS_05-10 – Zmena poctu objednanych kusu
    [Teardown]  Reload Page And Wait  ${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}
    Click On  ${PAGE_ORDER_QTY_XPATH}
    Click On  ${PAGE_ORDER_QTY_100_XPATH}
    Sleep  1
    Element Should Contain  ${PAGE_ORDER_QTY_XPATH}  ${TC_10_QTY}
    Element Should Contain  ${PAGE_ORDER_PRICE_XPATH}  ${TC_10_PRICE}

TS_05-11 – Resetovani nahraneho navrhu PCB
    [Setup]  File Check  ${VALID_PCB}
    Upload file  ${VALID_PCB}
    Wait Until Element Is Visible  ${PAGE_ORDER_PCB_DESIGN_VIEW_XPATH}  100s
    Click On Element And Wait  ${PAGE_ORDER_RESET_FILE_XPATH}  ${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}
    Element Should Be Visible  ${PAGE_ORDER_UPLOAD_BTN_XPATH}

*** Keywords ***
#TS precondition
TS_05 Precondition
    Open Browser With URL  ${BROWSER}  ${PAGE_ORDER_URL}  ${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}
    Init Tests

#TS postcondition
TS_05 Postcondition
    Close Browser

#overi zda soubor existuje
File Check
    [Arguments]  ${file}
    ${fileExists}=  File Exists  ${file}
    Should Be True  ${fileExists}

#nahraje soubor s navrhem PCB
Upload file
    [Arguments]  ${file}
    Wait Until Page Contains Element  ${PAGE_ORDER_ADD_GERBER_FILE_XPATH}  30s
    Scroll Element Into View  ${PAGE_ORDER_ADD_GERBER_FILE_XPATH}
    ${fileAbsolute}=  File Absolute Path  ${file}
    Choose File  ${PAGE_ORDER_ADD_GERBER_FILE_XPATH}  ${fileAbsolute}