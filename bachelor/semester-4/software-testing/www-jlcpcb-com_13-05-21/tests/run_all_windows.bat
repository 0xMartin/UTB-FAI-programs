@echo off

set TEST_SUITS=TS_03 TS_04 TS_05 TS_06 TS_07

(for %%i in (%TEST_SUITS%) do (
   if exist %%i/%%i.robot (
    cd %%i
    robot "%%i.robot"
    cd ..
   )
))

