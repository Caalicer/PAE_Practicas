/**
 * @brief Template for Labs
 *
 * PAE [G4012452] Labs
 * Last update: 03/02/2025
 * Issue date:  30/01/2022
 *
 * Student name: Jorge Lojo Abal y Pablo Liste Cancela
 *
 */

// General utilities
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>

// The file loadPGM.h will be used for defining load and export functions
#include "pgmio.h"

#define GRAY_LEVELS 256

/**
 * @brief Calculate the histogram of an image
 *
 * @param xu8 Image data
 * @param w Image width
 * @param h Image height
 * @param hist Histogram array
 */
void calculateHistogram(unsigned char* xu8, size_t w, size_t h, int* hist);

/**
 *
 * @brief Print the histogram of an image in CSV format
 * @param hist Histogram array
 * @return void
 *
 */
void printHistogram(int* hist);

/**
 * 
 * @brief Main function
 * @param argc Number of arguments
 * @param argv Arguments
 * @return int Exit status
 *
 */
int main(int argc, char *argv[]) {

    if (argc < 2) {
        printf("Use %s file.pgm\n", argv[0]);
        return EXIT_FAILURE;
    }

    struct timeval start;
    struct timeval start2;
    struct timeval end;
    struct timeval memoryStart;
    struct timeval memoryEnd;

    int w;
    int h;

    unsigned char* xu8 = loadPGMu8(argv[1], &w, &h);

    if (xu8 == NULL) {
        printf("Error: Image could not be loaded\n");
        return EXIT_FAILURE;
    }

    int hist[GRAY_LEVELS];

    gettimeofday(&memoryStart, NULL);

    memset(hist, 0, sizeof(int) * GRAY_LEVELS);

    gettimeofday(&memoryEnd, NULL);

    gettimeofday(&start, NULL);
    gettimeofday(&start2, NULL);

    calculateHistogram(xu8, w, h, hist);

    gettimeofday(&end, NULL);

    #ifdef DEBUG
        printHistogram(hist);
    #endif

    double overhead    = (start2.tv_sec - start.tv_sec) +                      \
                         (start2.tv_usec - start.tv_usec) / 1e6;

    double time        = (end.tv_sec - start2.tv_sec) +                        \
                         (end.tv_usec - start2.tv_usec) / 1e6 - overhead;

    double memory_time = (memoryEnd.tv_sec - memoryStart.tv_sec) +             \
                         (memoryEnd.tv_usec - memoryStart.tv_usec) /           \
                         1e6 - overhead;

    printf("PAE,%f,%f,%d\n", time, memory_time, hist[0]);

    free(xu8);

    return EXIT_SUCCESS;
    
}

void calculateHistogram(unsigned char* xu8, size_t w, size_t h, int* hist) {

    for (size_t i = 0; i < w * h; i++) {

        hist[xu8[i]]++;

    }
}

void printHistogram(int* hist) {

    for (int i = 0; i < GRAY_LEVELS; i++) {

        printf("%d, %d\n", i, hist[i]);

    }
}