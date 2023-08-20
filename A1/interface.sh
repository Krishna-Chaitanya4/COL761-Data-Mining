#!/bin/bash
ACTION="$1"
INPUT="$2"
OUTPUT="$3"
if [ "$ACTION" == "C" ]; then
	./a1 "$INPUT" "$OUTPUT"
elif [ "$ACTION" == "D" ]; then
	./a2  "$INPUT" "$OUTPUT" 
fi
