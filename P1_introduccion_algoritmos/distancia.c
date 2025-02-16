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

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

#define DEFAULT_N ((2.5 * 1024 * 1024 * 1024) / sizeof(float))

/**
 *
 * @brief Function that performs euclidean distance between two vectors
 * @param n Size of the vectors
 * @param x Vector x
 * @param y Vector y
 * @return void
 *
 */
float euclidean_distance(size_t n, float *x, float *y);

/**
 *
 * @brief Main function
 * @param argc Number of arguments
 * @param argv Array of arguments
 * @return int
 *
 */
int main(int argc, char *argv[]) {

    if (argc > 2) {
        printf("Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t n = (argc > 1) ? atoll(argv[1]) : DEFAULT_N;

    struct timeval start;
    struct timeval start2;
    struct timeval end;
    struct timeval memoryStart;
    struct timeval memoryEnd;

    gettimeofday(&memoryStart, NULL);

    float *x = (float *)malloc(n * sizeof(float));
    float *y = (float *)malloc(n * sizeof(float));

    gettimeofday(&memoryEnd, NULL);

    if (x == NULL || y == NULL) {
        printf("Error: Memory could not be allocated\n");
        return EXIT_FAILURE;
    }

    srand(time(NULL));

    for (size_t i = 0; i < n; i++) {

        x[i] = rand() % 10;
        y[i] = rand() % 10;

    }

    gettimeofday(&start, NULL);
    gettimeofday(&start2, NULL);

    float result = euclidean_distance(n, x, y);

    gettimeofday(&end, NULL);

    free(x);
    free(y);

    double overhead    = (start2.tv_sec - start.tv_sec) +                      \
                         (start2.tv_usec - start.tv_usec) / 1e6;
    
    double time        = (end.tv_sec - start2.tv_sec) +                        \
                         (end.tv_usec - start2.tv_usec) / 1e6 - overhead;
   
    double memory_time = (memoryEnd.tv_sec - memoryStart.tv_sec) +             \
                         (memoryEnd.tv_usec - memoryStart.tv_usec) /           \
                         1e6 - overhead;

    printf("PAE,%f,%f,%f,%ld\n", time, memory_time, result, n);

    return EXIT_SUCCESS;

}

float euclidean_distance(size_t n, float *x, float *y) {

    float result = 0;

    for (size_t i = 0; i < n; i++) {

        float temp = x[i] - y[i];
        result += temp * temp;

    }

    return sqrt(result);
}