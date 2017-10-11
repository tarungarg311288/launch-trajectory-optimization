#!/bin/bash

FILE="../workspace/$1/output/sv_planetIAU.out"
NEWFILE="../workspace/$1/output/new.txt"
NEWFILE1="../workspace/$1/output/FinalVector.txt"

if [ -f $NEWFILE ] ; then
    rm $NEWFILE  
fi
touch $NEWFILE

echo "$(tac $FILE | grep -m 1 '.')" >> $NEWFILE
cut -c 77- $NEWFILE > $NEWFILE1
rm $NEWFILE
