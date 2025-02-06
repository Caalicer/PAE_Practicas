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
void calculateHistogram(unsigned char* xu8, int w, int h, int* hist) {

    // Calculate histogram
    for (int i = 0; i < w * h; i++) {

        hist[xu8[i]]++;

    }

}

/**
 * @brief Print the histogram of an image
 *
 * @param hist Histogram array
 */
void printHistogram(int* hist) {

    // Print histogram
    for (int i = 0; i < GRAY_LEVELS; i++) {

        // Display the number of pixels with intensity i in the image using CSV format
        printf("%d, %d\n", i, hist[i]);

    }

}

/**
 * @brief Main function
 *
 * @param argc Number of arguments
 * @param argv Arguments
 * @return int Exit status
 */
int main(int argc, char *argv[]) {

    // Check for arguments
    if (argc < 2) {
        printf("Use %s file.pgm\n", argv[0]);
        exit(-1);
    }

    struct timeval start;
    struct timeval start2;
    struct timeval end;

    // Image size variables (width and height)
    int w;
    int h;

    // Load .pgm image
    unsigned char* xu8 = loadPGMu8(argv[1], &w, &h);

    // Check if the image has been loaded correctly
    if (xu8 == NULL) {

        printf("Error: Image could not be loaded\n");
        return EXIT_FAILURE;

    }

    // Histogram array
    int hist[GRAY_LEVELS] = {0};

    // Start timer
    gettimeofday(&start, NULL);
    gettimeofday(&start2, NULL);

    // Compute histogram of the image
    calculateHistogram(xu8, w, h, hist);

    // Stop timer
    gettimeofday(&end, NULL);

    // Print histogram of the image
    #ifdef DEBUG
        printHistogram(hist);
    #endif

    // Free memory
    free(xu8);

    // Calculate the elapsed time
    double overhead = (start2.tv_sec - start.tv_sec) + (start2.tv_usec - start.tv_usec) / 1e6;
    double time = (end.tv_sec - start2.tv_sec) + (end.tv_usec - start2.tv_usec) / 1e6 - overhead;

    // Print the results
    printf("PAE | Time: %f\n", time);

    return EXIT_SUCCESS;

}
