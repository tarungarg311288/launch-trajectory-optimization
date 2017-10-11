#!/bin/sh

Filename='../Orbit_Sim/input.cst'
opFilename='../Orbit_Sim/inp_tmp.txt'

touch $opFilename

strAddStart='! beginning of time interval, [utc]'
strAddEnd='! end of time interval, [utc]'

newStart="$1		$strAddStart"
newEnd="$2		$strAddEnd"

sed "/beginning of time interval/c\
$1		$strAddStart" < $Filename > $opFilename
mv $opFilename $Filename

touch $opFilename

sed "/end of time interval/c\
$2		$strAddEnd" < $Filename > $opFilename
mv $opFilename $Filename

