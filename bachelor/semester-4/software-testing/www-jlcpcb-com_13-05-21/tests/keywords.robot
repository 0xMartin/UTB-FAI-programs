*** Keywords ***

Open Browser With URL
    [Arguments]  ${browser}  ${url}  ${element}
    Open Browser  ${url}  ${browser}
    Wait Until Element Is Visible  ${element}

Click On Element And Wait
    [Arguments]  ${click_el}  ${element}
    Element Should Be Visible  ${click_el}
    Set Focus To Element  ${click_el}
    Mouse Over  ${click_el}
    Click Element  ${click_el}
    Wait Until Element Is Visible  ${element}  20s

Click On
    [Arguments]  ${btn}
    Element Should Be Visible  ${btn}
    Set Focus To Element  ${btn}
    Mouse Over  ${btn}
    Click Element  ${btn}

Select Last Browser Tab
    @{windows} =  Get Window Handles
    ${count} =  Get Length  ${windows}
    ${latest_window} =  Evaluate  ${count}-1
    Switch Window  ${windows}[${latest_window}]

Reload Page And Wait
    [Arguments]  ${element}
    Reload Page
    Element Should Be Visible  ${element}

Element Should Have Class
    [Arguments]  ${element}  ${className}
    ${class}=  Get Element Attribute  ${element}  class
    Should Contain  ${class}  ${className}

Element Should Not Have Class
    [Arguments]  ${element}  ${className}
    ${class}=  Get Element Attribute  ${element}  class
    Should Not Contain  ${class}  ${className}

Go To And Wait
    [Arguments]  ${url}  ${element}
    Go To  ${url}
    Element Should Be Visible  ${element}