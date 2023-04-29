*** Variables ***

${BROWSER}  Chrome
${SELENIUM_SPEED}  0.2
${SCREENSHOT_DIR}  ../screenshots
${RESOURCE_DIR}  ../resources
${SCREENSHOT_NAME_POSTFIX}  -screenshot.png


*** Keywords ***
Init Tests
    Maximize Browser Window
    Set Selenium Speed  ${SELENIUM_SPEED}
    Set Screenshot Directory  ${SCREENSHOT_DIR}
