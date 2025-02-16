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

// The file loadPGM.h will be used for defining load and export functions
#include "pgmio.h"

/**
 * @brief Calculate 2d shaerpen convolution
 *
 * @param image
 * @param image_widht
 * @param img_h
 * @param conv
 * @return int Exit status
 */
int conv2d (float *image, int img_w, int img_h, double* conv);

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
        printf("Use %s image.pgm\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    struct timeval start;
    struct timeval start2;
    struct timeval end;
    struct timeval memoryStart;
    struct timeval memoryEnd;

    int img_w;
    int img_h;

    float* image = loadPGM32(argv[1], &img_w, &img_h);

    if (image == NULL) {
        printf("Error: Image could not be loaded\n");
        return EXIT_FAILURE;
    }

    double *conv = NULL;

    if (!(conv = malloc(img_w * (long)img_h * sizeof(double)))) {
        printf("Error allocating convoluted image\n");
        return EXIT_FAILURE;
    }

    gettimeofday(&memoryStart, NULL);

    memset(conv, 0, img_w * (long)img_h);

    gettimeofday(&memoryEnd, NULL);

    gettimeofday(&start, NULL);
    gettimeofday(&start2, NULL);

    conv2d(image, img_w, img_h, conv);

    gettimeofday(&end, NULL);

    #ifdef DEBUG
        savePGM32("output.pgm", conv, img_w, img_h);
    #endif

    free(image);
    free(conv);

    double overhead    = (start2.tv_sec - start.tv_sec) +                      \
                         (start2.tv_usec - start.tv_usec) / 1e6;

    double time        = (end.tv_sec - start2.tv_sec) +                        \
                         (end.tv_usec - start2.tv_usec) / 1e6 - overhead;

    double memory_time = (memoryEnd.tv_sec - memoryStart.tv_sec) +             \
                         (memoryEnd.tv_usec - memoryStart.tv_usec) /           \
                         1e6 - overhead;

    printf("PAE | Time: %f\n", time);

    return EXIT_SUCCESS;

}

int conv2d (float *image, int img_w, int img_h, double* conv) {

    int ker_w = 3, ker_h = 3;
    float ker[] = {0, -1, 0, -1, 5, -1, 0, -1, 0};

    int pad_w = ker_w / 2;
    int pad_h = ker_h / 2;

    for (int y = 0; y < img_h; y++) {

        for (int x = 0; x < img_w; x++) {

            double sum = 0.0;

            for (int ky = 0; ky < ker_h; ky++) {

                for (int kx = 0; kx < ker_w; kx++) {

                    int img_x = x + kx - pad_w;
                    int img_y = y + ky - pad_h;

                    if (img_x >= 0 && img_x < img_w &&                         \
                        img_y >= 0 && img_y < img_h) {

                        sum += image[img_y * img_w + img_x] *                  \
                                 ker[ky    * ker_w + kx];
                    }
                }
            }

            conv[y * img_w + x] = sum;

        }
    }

    return EXIT_SUCCESS;

}
