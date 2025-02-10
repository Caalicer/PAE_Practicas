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

#define DEFAULT_ALPHA 2.0
#define DEFAULT_N ((6L * 1024 * 1024 * 1024) / sizeof(double))

/**
 *
 * @brief Function that performs the operation y = a*x + y
 * @param n Size of the vectors
 * @param a Scalar value
 * @param x Vector x
 * @param y Vector y
 * @return void
 *
 */
void daxpy(size_t n, double a, double *x, double *y);

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
    if (argc > 3) {
        printf("Usage: %s [alpha] [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    // Get the arguments or set the default values
    double alpha = (argc > 1) ? atof(argv[1]) : DEFAULT_ALPHA;
    size_t n = (argc > 2) ? atoll(argv[2]) : DEFAULT_N;

    // Allocate memory for the vectors
    double *x = (double *)malloc(n * sizeof(double));
    double *y = (double *)malloc(n * sizeof(double));

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
    daxpy(n, alpha, x, y);

    // Stop timer
    gettimeofday(&end, NULL);

    // Calculate the elapsed time
    double overhead = (start2.tv_sec - start.tv_sec) + (start2.tv_usec - start.tv_usec) / 1e6;
    double time = (end.tv_sec - start2.tv_sec) + (end.tv_usec - start2.tv_usec) / 1e6 - overhead;

    // Print the results
    printf("PAE | Time: %f | Result: %f | Alpha: %f | N: %ld \n", time, y[0], alpha, n);

    // Free memory
    free(x);
    free(y);

    return EXIT_SUCCESS;

}

void daxpy(size_t n, double a, double *x, double *y) {

    for (size_t i = 0; i < n; i++) {

        y[i] += a * x[i];

    }

}