#!/bin/bash

MAX_ITE=10

i=1
while [ $i -le $MAX_ITE ]
do
	srun $1
	i=$((i+1))
done