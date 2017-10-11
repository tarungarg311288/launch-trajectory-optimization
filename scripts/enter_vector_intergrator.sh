#!/bin/sh

filename="../workspace/$1/val.txt"
echo $filename
opFilename="../workspace/$1/input.cst"

strAdd="	! state [km,km,km,km/s,km/s,km/s] OR [km,-,deg,deg,deg,deg] "


read -r line < $filename
sed -i '$d' $opFilename
echo $line $strAdd >> $opFilename 
