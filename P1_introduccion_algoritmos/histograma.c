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

// The file loadPGM.h will be used for defining load and export functions
#include "pgmio.h"

#define GRAY_LEVELS 256

void calculateHistogram(unsigned char* xu8, int w, int h, int* hist) {

    // Calculate histogram
    for (int i = 0; i < w * h; i++) {

        hist[xu8[i]]++;

    }

}

void printHistogram(int* hist) {

    // Print histogram
    for (int i = 0; i < GRAY_LEVELS; i++) {

        printf("%d: %d\n", i, hist[i]);

    }

}

int main(int argc, char *argv[]) {

    // Check for arguments
    if (argc < 2) {
        printf("Use %s file.pgm\n", argv[0]);
        exit(-1);
    }

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

    // Compute histogram of the image
    calculateHistogram(xu8, w, h, hist);

    // Print histogram of the image
    #ifdef DEBUG
        printHistogram(hist);
    #endif

    free(xu8);

    return EXIT_SUCCESS;

}
