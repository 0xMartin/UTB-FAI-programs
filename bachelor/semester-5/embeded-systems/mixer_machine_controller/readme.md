# Coffee Mixing Unit Controller

[GO BACK](https://github.com/0xMartin/UTB-FAI-programs)

## Introduction:
We have selected the Coffee Mixing Unit model as our project and aimed to create a program that simulates a coffee vending machine and utilizes all the inputs and outputs of the mixing unit.

## Enumerators:

SV_X: SV1 = 0, SV2 = 1, SV3 = 2, SV4 = 3, SV5 = 4
    
* Used for defining individual valves and as a parameter in the valve open and close function

Tank1_Levels: H1_Level = 0, H2_Level = 1, H3_Level = 2

* Values corresponding to individual levels in Tank1 and as a parameter in the function for filling and emptying Tank1

Tank2_Levels: H4_Level = 3, H5_Level = 4

* Values corresponding to individual levels in Tank2 and as a parameter in the function for filling and emptying Tank2

Tank3_Levels: H6_Level = 5, H7_Level = 6, H8_Level = 7

* Values corresponding to individual levels in Tank3 and as a parameter in the function for filling and emptying Tank3

Tank: Tank1, Tank2, Tank3

* Values corresponding to the coffee vending machine tanks

## Functions:
All functions in the mixer module can operate in synchronous or asynchronous mode. For asynchronous execution of functions, it is necessary to define the ASYNCHRONOUS directive in mixer.h.

* __MIXER_init()__: Initialization of pins

* __MIXER_fillTank1(Tank1_Levels stopLevel)__: Function for filling Tank1; the level to which the tank should be filled is passed as a parameter
    
* __MIXER_fillTank2(Tank2_Levels stopLevel)__: Function for filling Tank2; the level to which the tank should be filled is passed as a parameter

* __MIXER_fillTank3(Tank3_Levels stopLevel)__: Function for filling Tank3; the level to which the tank should be filled is passed as a parameter

* __MIXER_tankIsEmpty(Tank t)__: Function that returns true if the tank passed as a parameter is empty; otherwise, it returns false

* __MIXER_valveControl(SV_X valve, bool open)__: Function for opening and closing valves; the first parameter specifies the valve we want to manipulate, and the second parameter specifies whether we want to open the valve (true) or close it (false)

* __MIXER_runMixer(uint32_t time)__: Function for learning the mixing time; the time in milliseconds is passed as a parameter

* __MIXER_reset()__: Function for resetting the coffee vending machine

After implementing all the driver functions, we created an infinite loop in the main function, which is controlled by the "state" variable. In the switch case, we defined the entire coffee production process using the mixer driver functions, which is realized as a finite state machine. The user is informed about the coffee vending machine's status on the LCD display. The production process can be canceled at any time by pressing the SW2 button.