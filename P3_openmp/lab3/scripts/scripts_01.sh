#!/bin/bash

#SBATCH -N 1
#SBATCH -c 64
#SBATCH --mem 1G
#SBATCH -t 00:02:00
#SBATCH -J P3_1
#SBATCH -o ./outs/P3_1.o
#SBATCH -e ./outs/P3_1.e

# Valid for exercises [01, 02, 03]

module load gcc
gcc -O2 -fopenmp -o main $1

echo "GCC data"
./main

rm main

module load intel
icc -O2 -qopenmp -o main $1

echo "ICC data"
./main

rm main