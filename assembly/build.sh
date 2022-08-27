#!/bin/bash
BASENAME=$(basename $1 .asm)
as -o $BASENAME.o $BASENAME.asm && \
ld -macosx_version_min 12.0.0 -o $BASENAME $BASENAME.o -lSystem -syslibroot `xcrun -sdk macosx --show-sdk-path` -e _start -arch arm64 && \
rm $BASENAME.o
