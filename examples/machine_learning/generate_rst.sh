#!/bin/bash

jupyter nbconvert --to rst --output-dir=../../doc/source/examples/machine_learning/ *.ipynb
