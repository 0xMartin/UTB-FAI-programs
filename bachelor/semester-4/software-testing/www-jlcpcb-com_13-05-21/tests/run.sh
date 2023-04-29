#!/usr/bin/env bash

if [ $# == 1 ]; then
  if [ -f "$1/$1.robot" ]; then
    cd $1
    python3 -m robot $1".robot"
    exit 0
  fi
fi

echo "Type name of Test Suit (example: TS_03):"
read name

if [ -f "$name/$name.robot" ]; then
  cd $name
  python3 -m robot $name".robot"
else
   echo "Not exists"
fi