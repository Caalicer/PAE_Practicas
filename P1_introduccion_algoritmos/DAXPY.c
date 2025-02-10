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

#define DEFAULT_N ((6L * 1024 * 1024 * 1024) / sizeof(double))
#define DEFAULT_ALPHA 2

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

    if (argc > 3) {
        printf("Usage: %s [alpha] [n]\n", argv[0]);
        return EXIT_FAILURE;
    }

    size_t n = DEFAULT_N;
    double alpha = DEFAULT_ALPHA;

    // Si se pasan argumentos, se usan los valores proporcionados
    if (argc > 1) {
        alpha = atof(argv[1]); // Primer argumento: valor de α
    }

    if (argc > 2) {
        n = atoll(argv[2]); // Segundo argumento: tamaño del array
    }

    // Allocate memory for the vectors
    double *x = (double *)malloc(n * sizeof(double));
    double *y = (double *)malloc(n * sizeof(double));

    // Check if the memory has been allocated
    if (x == NULL || y == NULL) {
        printf("Error: Memory could not be allocated\n");
        return EXIT_FAILURE;
    }

    struct timeval start;
    struct timeval start2;
    struct timeval end;

    // Initialize the seed
    srand(time(NULL));

    // Initialize the vectors
    for (int i = 0; i < n; i++) {

        x[i] = rand() % 10;
        y[i] = rand() % 10;

    }

    // start timer
    gettimeofday(&start, NULL);
    gettimeofday(&start2, NULL);

    // call the function
    daxpy(n, alpha, x, y);

    // stop timer
    gettimeofday(&end, NULL);

    free(x);
    free(y);

    // Calculate the elapsed time
    double overhead = (start2.tv_sec - start.tv_sec) + (start2.tv_usec - start.tv_usec) / 1e6;
    double time = (end.tv_sec - start2.tv_sec) + (end.tv_usec - start2.tv_usec) / 1e6 - overhead;

    // Print the results
    printf("PAE | Time: %f\n", time);

    return EXIT_SUCCESS;

}

void daxpy(size_t n, double a, double *x, double *y) {

    for (int i = 0; i < n; i++) {

        y[i] += a * x[i];

    }

}