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

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <string.h>
#include <math.h>

#include "pgmio.h"

// General utilities

// Custon utilities (in case of need)

// Implement the exercise in a function here

/**
 * @brief 2d convolution
 *
 * @param image
 * @param image_widht
 * @param img_h
 * @param conv
 * @return int Exit status
 */
int conv2d (float *image, int img_w, int img_h, double* conv);

// Main program
int main(int argc, char *argv[]) {

    // Check for arguments
    if (argc < 2) {
        printf("Use %s image.pgm\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    struct timeval start;
    struct timeval start2;
    struct timeval end;

    // Image size variables (width and height)
    int img_w;
    int img_h;

    // Load .pgm image
    float* image = loadPGM32(argv[1], &img_w, &img_h);

    // Check if the image has been loaded correctly
    if (image == NULL) {
        printf("Error: Image could not be loaded\n");
        return EXIT_FAILURE;
    }

    double *conv = NULL;

    if (!(conv = malloc(img_w * (long)img_h * sizeof(double)))) {
        printf("Error allocating convoluted image\n");
        return EXIT_FAILURE;
    }

    memset(conv, 0, img_w * (long)img_h);

    // Start timer
    gettimeofday(&start, NULL);
    gettimeofday(&start2, NULL);

    // Compute histogram of the image
    conv2d(image, img_w, img_h, conv);

    // Stop timer
    gettimeofday(&end, NULL);

    #ifdef DEBUG
        savePGM32("output.pgm", conv, img_w, img_h);
    #endif

    // Free memory
    free(image);
    free(conv);

    // Calculate the elapsed time
    double overhead = (start2.tv_sec - start.tv_sec) + (start2.tv_usec - start.tv_usec) / 1e6;
    double time = (end.tv_sec - start2.tv_sec) + (end.tv_usec - start2.tv_usec) / 1e6 - overhead;

    // Print the results
    printf("PAE | Time: %f\n", time);

    return EXIT_SUCCESS;

}

int conv2d (float *image, int img_w, int img_h, double* conv) {

    int ker_w = 3, ker_h = 3;
    float ker[] = {0, -1, 0, -1, 5, -1, 0, -1, 0};

    int pad_w = ker_w / 2;
    int pad_h = ker_h / 2;

    // Aplicar convolución con padding de ceros
    for (int y = 0; y < img_h; y++) {

        for (int x = 0; x < img_w; x++) {

            double sum = 0.0;

            for (int ky = 0; ky < ker_h; ky++) {

                for (int kx = 0; kx < ker_w; kx++) {

                    int img_x = x + kx - pad_w;
                    int img_y = y + ky - pad_h;

                    if (img_x >= 0 && img_x < img_w && img_y >= 0 && img_y < img_h) {
                        sum += image[img_y * img_w + img_x] * ker[ky * ker_w + kx];
                    }

                }

            }

            conv[y * img_w + x] = sum;

        }

    }

    return EXIT_SUCCESS;

}
