#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "pgmio.h"

#define NUM_SCHEDULES 4
#define NUM_CHUNK_SIZES 10
#define NUM_METHODS 2 // Número de métodos de paralelización

// Definir los nombres de los métodos para la salida
const char* method_names[] = {
    "rows-only",   // Solo paralelizar filas
    "collapse(2)", // Paralelizar con collapse(2)
};

const omp_sched_t scheduling_types[] = {omp_sched_static, omp_sched_dynamic,
                                        omp_sched_guided, omp_sched_auto};

const int chunk_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

// Método secuencial
int conv2d_sequential(float* image, int img_w, int img_h, float* conv) {
    int ker_w = 3, ker_h = 3;
    float ker[] = {0, -1, 0, -1, 5, -1, 0, -1, 0};

    int pad_w = ker_w / 2;
    int pad_h = ker_h / 2;

    for (int y = 0; y < img_h; y++) {
        for (int x = 0; x < img_w; x++) {
            float sum = 0.0;

            for (int ky = 0; ky < ker_h; ky++) {
                for (int kx = 0; kx < ker_w; kx++) {
                    int img_x = x + kx - pad_w;
                    int img_y = y + ky - pad_h;

                    if (img_x >= 0 && img_x < img_w && img_y >= 0 &&
                        img_y < img_h) {
                        sum +=
                            image[img_y * img_w + img_x] * ker[ky * ker_w + kx];
                    }
                }
            }

            conv[y * img_w + x] = sum;
        }
    }

    return EXIT_SUCCESS;
}

// Método 1: Solo paralelizar filas
int conv2d_parallel_rows(float* image, int img_w, int img_h, float* conv) {
    int ker_w = 3, ker_h = 3;
    float ker[] = {0, -1, 0, -1, 5, -1, 0, -1, 0};

    int pad_w = ker_w / 2;
    int pad_h = ker_h / 2;

#pragma omp parallel for
    for (int y = 0; y < img_h; y++) {
        for (int x = 0; x < img_w; x++) {
            float sum = 0.0;

            for (int ky = 0; ky < ker_h; ky++) {
                for (int kx = 0; kx < ker_w; kx++) {
                    int img_x = x + kx - pad_w;
                    int img_y = y + ky - pad_h;

                    if (img_x >= 0 && img_x < img_w && img_y >= 0 &&
                        img_y < img_h) {
                        sum +=
                            image[img_y * img_w + img_x] * ker[ky * ker_w + kx];
                    }
                }
            }

            conv[y * img_w + x] = sum;
        }
    }

    return EXIT_SUCCESS;
}

// Método 2: Paralelizar con collapse(2)
int conv2d_parallel_collapse(float* image, int img_w, int img_h, float* conv) {
    int ker_w = 3, ker_h = 3;
    float ker[] = {0, -1, 0, -1, 5, -1, 0, -1, 0};

    int pad_w = ker_w / 2;
    int pad_h = ker_h / 2;

#pragma omp parallel for collapse(2)
    for (int y = 0; y < img_h; y++) {
        for (int x = 0; x < img_w; x++) {
            float sum = 0.0;

            for (int ky = 0; ky < ker_h; ky++) {
                for (int kx = 0; kx < ker_w; kx++) {
                    int img_x = x + kx - pad_w;
                    int img_y = y + ky - pad_h;

                    if (img_x >= 0 && img_x < img_w && img_y >= 0 &&
                        img_y < img_h) {
                        sum +=
                            image[img_y * img_w + img_x] * ker[ky * ker_w + kx];
                    }
                }
            }

            conv[y * img_w + x] = sum;
        }
    }

    return EXIT_SUCCESS;
}

typedef int (*conv_function)(float*, int, int, float*);

// Array de punteros a las funciones de convolución
conv_function conv_methods[] = {conv2d_parallel_rows, conv2d_parallel_collapse};

int check_results(float* result1, float* result2, int size) {
    for (int i = 0; i < size; i++) {
        if (fabs(result1[i] - result2[i]) > 1e-6) {
            return 0;
        }
    }
    return 1;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        printf("Usage: %s image.pgm numThreads\n", argv[0]);
        exit(EXIT_FAILURE);
    }

    char* image_filename = argv[1];
    int num_threads = atoi(argv[2]);

    omp_set_dynamic(0);
    omp_set_num_threads(num_threads);

    int img_w, img_h;

    // Overhead calculation
    double overhead_start = omp_get_wtime();
    double overhead_end = omp_get_wtime();
    double overhead_time = overhead_end - overhead_start;

    // Load image
    double load_start = omp_get_wtime();
    float* image = loadPGM32(image_filename, &img_w, &img_h);
    double load_end = omp_get_wtime();

    size_t imageSize = img_w * (size_t)img_h;

    if (image == NULL) {
        printf("Error: Image could not be loaded\n");
        return EXIT_FAILURE;
    }

    // Allocate result images
    double malloc_start = omp_get_wtime();
    float* conv_seq = malloc(img_w * (long)img_h * sizeof(float));
    float* conv_par = malloc(img_w * (long)img_h * sizeof(float));
    double malloc_end = omp_get_wtime();

    if (conv_seq == NULL || conv_par == NULL) {
        printf("Error allocating result images\n");
        free(image);
        return EXIT_FAILURE;
    }

    // Sequential execution
    double seq_start = omp_get_wtime();
    conv2d_sequential(image, img_w, img_h, conv_seq);
    double seq_end = omp_get_wtime();

    // Calculate times without overhead
    double load_time = load_end - load_start - overhead_time;
    double malloc_time = malloc_end - malloc_start - overhead_time;
    double seq_time = seq_end - seq_start - overhead_time;

    // Try each parallelization method
    for (int method = 0; method < NUM_METHODS; method++) {
        // Try different scheduling strategies and chunk sizes
        for (int i = 0; i < NUM_SCHEDULES; i++) {
            for (int j = 0; j < NUM_CHUNK_SIZES; j++) {
                omp_set_schedule(scheduling_types[i], chunk_sizes[j]);

                // Parallel execution
                double par_start = omp_get_wtime();
                conv_methods[method](image, img_w, img_h, conv_par);
                double par_end = omp_get_wtime();

                // Check results
                int results_match =
                    check_results(conv_seq, conv_par, img_w * img_h);
                if (!results_match) {
                    printf("Error: Parallel results for method %s do not match "
                           "sequential results\n",
                           method_names[method]);
                    continue;
                }

                double par_time = par_end - par_start - overhead_time;
                double speedup = seq_time / par_time;
                double total_time =
                    load_time + malloc_time + seq_time + par_time;

                // Print results including the method name
                printf("\nPAE,%s,%d,%zu,%d,%d,%s,%f,%f,%f,%f,%f,%f,%f,PAE\n",
                       image_filename, num_threads, imageSize,
                       scheduling_types[i], chunk_sizes[j],
                       method_names[method], overhead_time, load_time,
                       malloc_time, par_time, seq_time, speedup, total_time);
            }
        }
    }

#ifdef DEBUG
    savePGM32("output.pgm", conv_par, img_w, img_h);
#endif

    // Free memory
    free(image);
    free(conv_seq);
    free(conv_par);

    return EXIT_SUCCESS;
}