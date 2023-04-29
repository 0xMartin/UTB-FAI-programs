*** Variables ***

#-<DEFAULT>-------------------------------------------------------------------------------------------------------------
${PAGE_ORDER_URL}  https://cart.jlcpcb.com
${PAGE_ORDER_WAIT_FOR_ELEMENT_XPATH}  //body

${PAGE_ORDER_ACCEPT_COOKIES_XPATH}  //button[@ng-click="accept()"]

#-<Charge Details>------------------------------------------------------------------------------------------------------
${PAGE_ORDER_PRICE_XPATH}  //strong[@class="ver ng-binding"]
${PAGE_ORDER_WEIGHT_XPATH}  //div[@class="pt10"]/span[@class="pull-right ng-binding"]

#-<FORM PCB>----------------------------------------------------------------------------------------------------------------
#file input
${PAGE_ORDER_ADD_GERBER_FILE_XPATH}  //input[@id='flieUpload']
${PAGE_ORDER_PCB_DESIGN_VIEW_XPATH}  //body[1]/div[1]/div[2]/div[1]/div[1]/div[1]/div[2]/div[1]/div[1]/div[1]/div[1]/div[1]/img[1]
${PAGE_ORDER_RESET_FILE_XPATH}  //body/div[@id='myContainer']/div[2]/div[1]/div[1]/div[1]/div[2]/div[1]/div[1]/div[3]/div[1]/a[1]
${PAGE_ORDER_UPLOAD_BTN_XPATH}  //body/div[@id='myContainer']/div[2]/div[1]/div[1]/div[1]/div[2]/label[1]/a[1]

#incorrenct format dialog
${PAGE_ORDER_INCORRECT_FORMAT_DIALOG_XPATH}  //div[@id='msg'][contains(string(),'Incorrect format')]

#layers
${PAGE_ORDER_4_LAYERS_XPATH}  //div[contains(@divname, "stencil-layer-pcb-div") and @class="formgroup"]/button[3]

#dimension
${PAGE_ORDER_HEIGHT_XPATH}  //body/div[@id='myContainer']/div[2]/div[1]/div[1]/div[1]/div[3]/div[1]/div[3]/div[1]/input[1]
${PAGE_ORDER_WIDTH_XPATH}  //body/div[@id='myContainer']/div[2]/div[1]/div[1]/div[1]/div[3]/div[1]/div[3]/div[1]/input[2]
${PAGE_ORDER_UNITS_XPATH}  //body/div[@id='myContainer']/div[2]/div[1]/div[1]/div[1]/div[3]/div[1]/div[3]/div[1]/select[1]
${PAGE_ORDER_DIM_EXCEED_XPATH}  //span[contains(text(),"Pls don't exceed the max dimension of 400mm*500mm.")]

#Qty
${PAGE_ORDER_QTY_XPATH}  //body/div[@id='myContainer']/div[2]/div[1]/div[1]/div[1]/div[3]/div[1]/div[5]/div[1]/div[1]/a[1]/span[1]
${PAGE_ORDER_QTY_100_XPATH}  //body/div[@id='myContainer']/div[2]/div[1]/div[1]/div[1]/div[3]/div[1]/div[5]/div[1]/div[1]/ul[1]/li[9]/button[1]

#COLORS
${PAGE_ORDER_COLOR_GREEN_XPATH}  //button[contains(@ng-click, "setSelectedColor('Green')")]
${PAGE_ORDER_COLOR_RED_XPATH}  //button[contains(@ng-click, "setSelectedColor('Red')")]
${PAGE_ORDER_COLOR_YELLOW_XPATH}  //button[contains(@ng-click, "setSelectedColor('Yellow')")]
${PAGE_ORDER_COLOR_BLUE_XPATH}  //button[contains(@ng-click, "setSelectedColor('Blue')")]
${PAGE_ORDER_COLOR_WHITE_XPATH}  //button[contains(@ng-click, "setSelectedColor('White')")]
${PAGE_ORDER_COLOR_BLACK_XPATH}  //button[contains(@ng-click, "setSelectedColor('Black')")]

#THICKNESS
${PAGE_ORDER_THICKNESS_2_XPATH}  //button[contains(text(),'2.0')]

#ADVANCED OPTS
${PAGE_ORDER_ADVANCED_XPATH}  //div[@ng-click="form.isShowAdvancedOptions=!form.isShowAdvancedOptions;"]
${PAGE_ORDER_ADVANCED_OPT2_XPATH}  //body/div[@id='myContainer']/div[2]/div[1]/div[1]/div[1]/div[3]/div[1]/div[36]/div[5]


#-<FORM SMT STENCIL>----------------------------------------------------------------------------------------------------
${PAGE_ORDER_SMT_STENCIL_TYPE_XPATH}  //a[contains(text(),'SMT-Stencil')]

#DIMENSION
${PAGE_ORDER_SMT_STENCIL_DIMENSION_XPATH}  //select[@id='steelmeshSellingPriceRecordNum']

#GTY
${PAGE_ORDER_SMT_STENCIL_QTY_XPATH}  //body/div[@id='myContainer']/div[2]/div[1]/div[1]/div[2]/div[1]/div[5]/div[1]/input[1]