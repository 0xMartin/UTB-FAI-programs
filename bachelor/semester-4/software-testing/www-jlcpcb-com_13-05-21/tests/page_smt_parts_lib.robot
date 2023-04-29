*** Variables ***

#-<DEFAULT>-------------------------------------------------------------------------------------------------------------
${PAGE_STM_PARTS_LIB_URL}  https://jlcpcb.com/parts
${PAGE_STM_PARTS_LIB_WAIT_FOR_ELEMENT_XPATH}  //body

${PAGE_ORDER_ACCEPT_COOKIES_XPATH}  //button[@ng-click="accept()"]

#-<ALL CATEGORIES>------------------------------------------------------------------------------------------------------
#CATEGORIES
${PAGE_STM_PARTS_LIB_AMPLIFIERS_XPATH}  //span[contains(text(),'Amplifiers')]
${PAGE_STM_PARTS_LIB_DIODER_XPATH}  //span[contains(text(),'Diodes')]
${PAGE_STM_PARTS_LIB_FILTERS_XPATH}  //span[contains(text(),'Filters')]
${PAGE_STM_PARTS_LIB_CONNECTORS_XPATH}  //span[contains(text(),'Connectors')]
${PAGE_STM_PARTS_LIB_CAPACITORS_XPATH}  //span[contains(text(),'Capacitors')]
${PAGE_STM_PARTS_LIB_RESISTORS_XPATH}  //span[contains(text(),'Resistors')]
${PAGE_STM_PARTS_LIB_CRYSTALS_XPATH}  //span[contains(text(),'Crystals')]

#SUB CATEGORIES
${PAGE_STM_PARTS_LIB_SW_DIODES_XPATH}  //span[contains(text(),'Switching Diode')]
${PAGE_STM_PARTS_LIB_CRYSTAL_RESONATORS_XPATH}  //span[contains(text(),'SMD Crystal Resonators')]

#CATEGORY TITLE
${PAGE_STM_PARTS_LIB_ACTIVE_CATEGORY_XPATH}  //ol/li[contains(@class,'active')][1]
${PAGE_STM_PARTS_LIB_ACTIVE_SUB_CATEGORY_XPATH}  //ol/li[contains(@class,'active')][2]


#-<FILTER>------------------------------------------------------------------------------------------------------
#MANUFACTURER AMPLIFIERS
${PAGE_STM_PARTS_LIB_FILTER_TEXAS_INS_XPATH}  //span[contains(text(),'Texas Instruments')]
${PAGE_STM_PARTS_LIB_FILTER_APPLY_XPATH}  //span[contains(text(),'Apply')]
${PAGE_STM_PARTS_LIB_FILTER_RESET_XPATH}  //span[contains(text(),'Reset All')]

${PAGE_STM_PARTS_RESULT_1_XPATH}  //tbody/tr[1]
${PAGE_STM_PARTS_RESULT_2_XPATH}  //tbody/tr[2]
${PAGE_STM_PARTS_RESULT_3_XPATH}  //tbody/tr[3]


#-<SEARCH>------------------------------------------------------------------------------------------------------
${PAGE_STM_PARTS_SEARCH_INPUT_XPATH}  //body/div[@id='foreignTradeFrontDesk']/div[2]/div[1]/div[3]/input[1]
${PAGE_STM_PARTS_SEARCH_BTN_XPATH}  //body/div[@id='foreignTradeFrontDesk']/div[2]/div[1]/div[3]/img[1]
