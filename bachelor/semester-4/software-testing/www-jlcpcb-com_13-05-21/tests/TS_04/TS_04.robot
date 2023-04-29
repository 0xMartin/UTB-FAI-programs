*** Settings ***
Library  SeleniumLibrary  run_on_failure=Nothing
Documentation  Testovaci sada pro overeni spravneho zobrazovani UI domovske stranky
Metadata  Author  Martin Krcma

Resource  ../settings.robot
Resource  ../keywords.robot
Resource  ../page_home.robot

Suite Setup  TS_01 Precondition
Suite Teardown  TS_01 Postcondition


*** Test Cases ***
TS_04-01 – Zobrazeni navigace
    [Teardown]  Capture Page Screenshot  TS_04-01${SCREENSHOT_NAME_POSTFIX}
    Scroll Element Into View  ${PAGE_HOME_NAVIGATION_XPATH}
    Element Should Be Visible  ${PAGE_HOME_NAVIGATION_XPATH}

TS_04-02 – Zobrazeni panelu pro vyber objednavky
    Scroll Element Into View  ${PAGE_HOME_INSTANT_QUOTE_XPATH}
    Element Should Be Visible  ${PAGE_HOME_INSTANT_QUOTE_XPATH}

TS_04-03 – Zobrazeni nabidky sluzeb
    [Teardown]  Capture Page Screenshot  TS_04-03${SCREENSHOT_NAME_POSTFIX}
    Scroll Element Into View  ${PAGE_HOME_OFFERED_SERVICES_XPATH}
    Element Should Be Visible  ${PAGE_HOME_OFFERED_SERVICES_XPATH}

TS_04-04 – Zobrazeni prezentacniho videa
    [Teardown]  Run Keywords
    ...  Capture Page Screenshot  TS_04-04${SCREENSHOT_NAME_POSTFIX}
    ...  AND  Click On  ${PAGE_HOME_VIDEO_CLOSE_BTN_XPATH}
    Scroll Element Into View  ${PAGE_HOME_VIDEO_XPATH}
    Click On Element And Wait  ${PAGE_HOME_VIDEO_XPATH}  ${PAGE_HOME_VIDEO_IFRAME_XPATH}
    Click On  ${PAGE_HOME_VIDEO_IFRAME_XPATH}

TS_04-05 – Zobrazeni videi vyrobniho procesu
    [Teardown]  Run Keywords
    ...  Capture Page Screenshot  TS_04-05${SCREENSHOT_NAME_POSTFIX}
    ...  AND  Click On  ${PAGE_HOME_MANUFACTURING_VIDEO_CLOSE_BTN_XPATH}
    Scroll Element Into View  ${PAGE_HOME_MANUFACTURING_CONATINER_XPATH}
    Click On Element And Wait  ${PAGE_HOME_MANUFACTURING_FIRST_VIDEO_XPATH}  ${PAGE_HOME_MANUFACTURING_VIDEO_IFRAME_XPATH}
    Element Should Be Visible  ${PAGE_HOME_MANUFACTURING_VIDEO_IFRAME_XPATH}

TS_04-07 – Zobrazeni footeru stranky
    [Teardown]  Capture Page Screenshot  TS_04-07${SCREENSHOT_NAME_POSTFIX}
    Scroll Element Into View  ${PAGE_HOME_FOOTER_XPATH}
    Element Should Be Visible  ${PAGE_HOME_FOOTER_XPATH}

TS_04-08 – Zobrazeni navodneho videa
    [Teardown]  Capture Page Screenshot  TS_04-09${SCREENSHOT_NAME_POSTFIX}
    Scroll Element Into View  ${PAGE_HOME_TUTORIAL_VIDEO_XPATH}
    Element Should Be Visible  ${PAGE_HOME_TUTORIAL_VIDEO_XPATH}


*** Keywords ***
#TS precondition
TS_01 Precondition
    Open Browser With URL  ${BROWSER}  ${PAGE_HOME_URL}  ${PAGE_HOME_WAIT_FOR_ELEMENT_XPATH}
    Init Tests

#TS postcondition
TS_01 Postcondition
    Close Browser