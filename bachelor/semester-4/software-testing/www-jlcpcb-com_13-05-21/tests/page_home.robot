*** Variables ***

#-<DEFAULT>-------------------------------------------------------------------------------------------------------------
${PAGE_HOME_URL}  https://jlcpcb.com/
${PAGE_HOME_WAIT_FOR_ELEMENT_XPATH}  //body


#-<NAVIGATION>----------------------------------------------------------------------------------------------------------
#NAVIGATION CONTAIBER
${PAGE_HOME_NAVIGATION_XPATH}  //div[contains(@class, "home-header-cont")]
${PAGE_HOME_NAVIGATION_WHY_XPATH}  //span[contains(text(),'Why JLCPCB?')]
${PAGE_HOME_NAVIGATION_CAPABILITIES_XPATH}  //span[contains(text(),'Capabilities')]
${PAGE_HOME_NAVIGATION_SUPPORT_XPATH}  //span[contains(text(),'Support')]

${PAGE_HOME_NAVIGATION_RES_XPATH}  //span[contains(text(),'Resources')]
${PAGE_HOME_NAVIGATION_RES_DROP_DOWN_XPATH}  //body/div[2]/div[2]/div[2]/div[1]/div[3]/ul[1]
${PAGE_HOME_NAVIGATION_RES_SMT_ASSEMBLY_XPATH}  //a[contains(text(),'SMT Assembly')]
${PAGE_HOME_NAVIGATION_RES_SMT_PART_LIB_XPATH}  //a[contains(text(),'SMT Parts Library')]
${PAGE_HOME_NAVIGATION_RES_EEDA_XPATH}  //p[contains(@class,'mt10')]/a[contains(text(),'EasyEDA')]
${PAGE_HOME_NAVIGATION_RES_LCSC_XPATH}  //a[contains(text(),'LCSC Electronics')]

${PAGE_HOME_NAVIGATION_ORDER_XPATH}  //a[contains(text(),'Order now')]


#-<BODY ELEMENTS>----------------------------------------------------------------------------------------------------------
#GET INSTANT QUOTE ELEMENT
${PAGE_HOME_INSTANT_QUOTE_XPATH}  //body/div[@id='bannerbox']/div[3]/div[1]

#OFFERED SERVICES ELEMENT
${PAGE_HOME_OFFERED_SERVICES_XPATH}  //div[contains(@class, "homecont-01")]

#VIDEO
${PAGE_HOME_VIDEO_XPATH}  //div[contains(@class, "pcbMediaBg")]
${PAGE_HOME_VIDEO_IFRAME_XPATH}  //iframe[@id='video2']
${PAGE_HOME_VIDEO_CLOSE_BTN_XPATH}  //body/div[@id='videoModal1']/a[1]

#MANUFACTURING PROCESS
${PAGE_HOME_MANUFACTURING_CONATINER_XPATH}  //div[contains(@class, 'homecont-03')]
${PAGE_HOME_MANUFACTURING_FIRST_VIDEO_XPATH}  //body/div[4]/div[4]/div[2]/div[1]/div[2]/div[1]/ul[1]/li[1]
${PAGE_HOME_MANUFACTURING_VIDEO_IFRAME_XPATH}  //iframe[@id='video1']
${PAGE_HOME_MANUFACTURING_VIDEO_CLOSE_BTN_XPATH}  //body/div[@id='videoModal']/a[1]

#FOOTER
${PAGE_HOME_FOOTER_XPATH}  //div[contains(@class, 'newFooter')]

#GIF
${PAGE_HOME_TUTORIAL_VIDEO_XPATH}  //video[@id='video']

*** Keywords ***

