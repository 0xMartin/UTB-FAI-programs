*** Settings ***
Library  SeleniumLibrary  run_on_failure=Nothing
Documentation  Tato testovaci sada overi spravnou funkcnost navigace na webove strance jlcpcb.com
Metadata  Author  Martin Krcma

Resource  ../settings.robot
Resource  ../keywords.robot
Resource  ../page_home.robot
Resource  ../page_about.robot
Resource  ../page_capabilities.robot
Resource  ../page_support.robot
Resource  ../page_smt_assembly.robot
Resource  ../page_smt_parts_lib.robot
Resource  ../page_easyeda.robot
Resource  ../page_LCSC.robot
Resource  ../page_order.robot

Suite Setup  TS_03 Precondition
Suite Teardown  TS_03 Postcondition


*** Test Cases ***
TS_03-01 – Odkaz navigace „Why JLCPCB?“
    [Setup]  Element Should Be Visible  ${PAGE_HOME_NAVIGATION_XPATH}
    [Teardown]  Log Location And Back
    Click On Element And Wait  ${PAGE_HOME_NAVIGATION_WHY_XPATH}  ${PAGE_ABOUT_WAIT_FOR_ELEMENT_XPATH}
    Location Should Be  ${PAGE_ABOUT_URL}

TS_03-02 – Odkaz navigace „Capabilities“
    [Setup]  Element Should Be Visible  ${PAGE_HOME_NAVIGATION_XPATH}
    [Teardown]  Log Location And Back
    Click On Element And Wait  ${PAGE_HOME_NAVIGATION_CAPABILITIES_XPATH}  ${PAGE_CAPABILITIES_WAIT_FOR_ELEMENT_XPATH}
    Location Should Be  ${PAGE_CAPABILITIES_URL}

TS_03-03 – Odkaz navigace „Support“
    [Setup]  Element Should Be Visible  ${PAGE_HOME_NAVIGATION_XPATH}
    [Teardown]  Run Keywords
    ...  Capture Page Screenshot  TS_03-03${SCREENSHOT_NAME_POSTFIX}
    ...  AND  Close Window
    ...  AND  Select Last Browser Tab
    Click On Element And Wait  ${PAGE_HOME_NAVIGATION_SUPPORT_XPATH}  ${PAGE_SUPPORT_WAIT_FOR_ELEMENT_XPATH}
    Select Last Browser Tab
    Location Should Contain  ${PAGE_SUPPORT_URL}

TS_03-04 – Drop down menu „Resources“
    [Setup]  Element Should Be Visible  ${PAGE_HOME_NAVIGATION_XPATH}
    [Teardown]  Capture Page Screenshot  TS_03-04${SCREENSHOT_NAME_POSTFIX}
    Open Drop Down Menu Res

TS_03-05 – Drop down menu Resources – SMT assembly
    [Setup]  Element Should Be Visible  ${PAGE_HOME_NAVIGATION_XPATH}
    [Teardown]  Run Keywords
    ...  Capture Page Screenshot  TS_03-05${SCREENSHOT_NAME_POSTFIX}
    ...  AND  Close Window
    ...  AND  Select Last Browser Tab
    Open Drop Down Menu Res
    Click On Element And Wait  ${PAGE_HOME_NAVIGATION_RES_SMT_ASSEMBLY_XPATH}  ${PAGE_STM_ASSEMBLY_WAIT_FOR_ELEMENT_XPATH}
    Select Last Browser Tab
    Location Should Be  ${PAGE_STM_ASSEMBLY_URL}

TS_03-06 – Drop down menu Resources – SMT Parts Library
    [Setup]  Element Should Be Visible  ${PAGE_HOME_NAVIGATION_XPATH}
    [Teardown]  Run Keywords
    ...  Log Location
    ...  AND  Close Window
    ...  AND  Select Last Browser Tab
    Open Drop Down Menu Res
    Click On Element And Wait  ${PAGE_HOME_NAVIGATION_RES_SMT_PART_LIB_XPATH}  ${PAGE_STM_PARTS_LIB_WAIT_FOR_ELEMENT_XPATH}
    Select Last Browser Tab
    Location Should Be  ${PAGE_STM_PARTS_LIB_URL}

TS_03-07 – Drop down menu Resources – EasyEDA
    [Setup]  Element Should Be Visible  ${PAGE_HOME_NAVIGATION_XPATH}
    [Teardown]  Run Keywords
    ...  Capture Page Screenshot  TS_03-07${SCREENSHOT_NAME_POSTFIX}
    ...  AND  Close Window
    ...  AND  Select Last Browser Tab
    Open Drop Down Menu Res
    Click On Element And Wait  ${PAGE_HOME_NAVIGATION_RES_EEDA_XPATH}  ${PAGE_EASYEDA_WAIT_FOR_ELEMENT_XPATH}
    Select Last Browser Tab
    Location Should Contain  ${PAGE_EASYEDA_URL}

TS_03-08 – Drop down menu Resources – LCSC electronics
    [Setup]  Element Should Be Visible  ${PAGE_HOME_NAVIGATION_XPATH}
    [Teardown]  Run Keywords
    ...  Log Location
    ...  AND  Close Window
    ...  AND  Select Last Browser Tab
    Open Drop Down Menu Res
    Click On Element And Wait  ${PAGE_HOME_NAVIGATION_RES_LCSC_XPATH}  ${PAGE_LCSC_WAIT_FOR_ELEMENT_XPATH}
    Select Last Browser Tab
    Location Should Contain  ${PAGE_LCSC_URL}

TS_03-09 – Order now
    [Setup]  Element Should Be Visible  ${PAGE_HOME_NAVIGATION_XPATH}
    [Teardown]  Run Keywords
    ...  Log Location
    ...  AND  Go Back
    Click On Element And Wait  ${PAGE_HOME_NAVIGATION_ORDER_XPATH}  ${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}
    Location Should Contain  ${PAGE_ORDER_URL}

TS_03-13 – Navrat zpet na hlavni stranku
    [Setup]  Element Should Be Visible  ${PAGE_HOME_NAVIGATION_XPATH}
    [Teardown]  Run Keywords
    ...  Capture Page Screenshot  TS_03-13${SCREENSHOT_NAME_POSTFIX}
    ...  AND  Go Back
    Click On Element And Wait  ${PAGE_HOME_NAVIGATION_CAPABILITIES_XPATH}  ${PAGE_CAPABILITIES_WAIT_FOR_ELEMENT_XPATH}
    Click On Element And Wait  ${PAGE_CAPABILITIES_BACK_XPATH}  ${PAGE_HOME_WAIT_FOR_ELEMENT_XPATH}
    Location Should Be  ${PAGE_HOME_URL}


*** Keywords ***
#TS precondition
TS_03 Precondition
    Open Browser With URL  ${BROWSER}  ${PAGE_HOME_URL}  ${PAGE_HOME_WAIT_FOR_ELEMENT_XPATH}
    Init Tests

#TS postcondition
TS_03 Postcondition
    Close Browser

Log Location And Back
    Log Location
    Go Back

Open Drop Down Menu Res
    Mouse Over  ${PAGE_HOME_NAVIGATION_RES_XPATH}
    Element Should Be Visible  ${PAGE_HOME_NAVIGATION_RES_DROP_DOWN_XPATH}