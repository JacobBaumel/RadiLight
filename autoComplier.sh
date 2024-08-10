#! /bin/bash
cd ~/bru
rm -rf ~/bru/build/*
cd ~/bru/build
cmake ..
make
cd ~/bru/build
./opencvTest
