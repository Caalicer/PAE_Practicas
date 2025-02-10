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

    // Check if the number of arguments is correct
    if (argc > 2) {
        printf("Usage: %s [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Get the arguments or set the default values
    size_t n = (argc > 1) ? atoll(argv[1]) : DEFAULT_N;

    // Allocate memory for the vectors
    float *x = (float *)malloc(n * sizeof(float));
    float *y = (float *)malloc(n * sizeof(float));

    // Check if the memory has been allocated
    if (x == NULL || y == NULL) {
        printf("Error: Memory could not be allocated\n");
        return EXIT_FAILURE;
    }

    // Initialize the timers
    struct timeval start;
    struct timeval start2;
    struct timeval end;

    // Initialize the seed
    srand(time(NULL));

    // Initialize the vectors
    for (size_t i = 0; i < n; i++) {

        x[i] = rand() % 10;
        y[i] = rand() % 10;

    }

    // Start timer
    gettimeofday(&start, NULL);
    gettimeofday(&start2, NULL);

    // Call the function
    float result = euclidean_distance(n, x, y);

    // Stop timer
    gettimeofday(&end, NULL);

    // Free memory
    free(x);
    free(y);

    // Calculate the elapsed time
    double overhead = (start2.tv_sec - start.tv_sec) + (start2.tv_usec - start.tv_usec) / 1e6;
    double time = (end.tv_sec - start2.tv_sec) + (end.tv_usec - start2.tv_usec) / 1e6 - overhead;

    // Print the results
    printf("PAE | Time: %f | Result: %f | N: %ld \n", time, result, n);

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