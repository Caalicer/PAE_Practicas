#!/bin/bash

#SBATCH -N 1
#SBATCH --mem 8G
#SBATCH -t 06:00:00

for i in {0..9}; do

	'./'$1 $2 $3 $4 $5

done