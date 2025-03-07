#!/bin/bash

#SBATCH -N 1
#SBATCH -c 32
#SBATCH --mem 64G
#SBATCH --gres gpu:a100:1
#SBATCH -t 00:02:00
#SBATCH -J P2_1
#SBATCH -o ./outs/P2_1.o
#SBATCH -e ./outs/P2_1.e

# Valid for exercises [0, 6]
nvcc -O2 -o main $1

./main

rm main