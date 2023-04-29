#!/usr/bin/env bash

TEST_SUITS=("TS_03" "TS_04" "TS_05" "TS_06" "TS_07")

for ts_file_name in ${TEST_SUITS[@]}
do
   if [ -f "$ts_file_name/$ts_file_name.robot" ]; then
      cd $ts_file_name
      python3 -m robot $ts_file_name".robot"
      cd ..
   fi
done