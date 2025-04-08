#!/bin/bash

#SBATCH -N 1
#SBATCH -c 1
#SBATCH --mem 128G
#SBATCH -t 00:10:00
#SBATCH -J P4_1
#SBATCH -o ./outs/P4_1.o
#SBATCH -e ./outs/P4_1.e

module load gcc
gcc -O2 -o main $1

./main

rm main
