#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

#define DEFAULT_N ((2.5 * 1024 * 1024 * 1024) / sizeof(float))

#define NUM_SCHEDULES 4
#define NUM_CHUNK_SIZES 10

const omp_sched_t scheduling_types[] = {
	omp_sched_static,
	omp_sched_dynamic,
	omp_sched_guided,
	omp_sched_auto
};

const int chunk_sizes[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};

void init_vectors(float *A, float *B, int size) {

	for (int i = 0; i < size; i++) {

		A[i] = i;
		B[i] = i;

	}

}

float euclidean_distance_automatic_reduction(float *A, float *B, int size) {

	float sum = 0.0;

	#pragma omp parallel for reduction(+:sum)
	for (int i = 0; i < size; i++) {

		float diff = A[i] - B[i];
		sum += diff * diff;

	}

	return sqrt(sum);

}

float euclidean_distance_manual_reduction(float *A, float *B, size_t size) {

	float total_sum = 0.0;

	#pragma omp parallel
	{

		float local_sum = 0.0;

		#pragma omp for nowait
		for (size_t i = 0; i < size; i++) {

			float diff = A[i] - B[i];
			local_sum += diff * diff;

		}

		#pragma omp atomic
		total_sum += local_sum;

	}

	return sqrtf(total_sum);

}

float euclidean_distance_secuential(float *A, float *B, int size) {

	float sum = 0.0;

	for (int i = 0; i < size; i++) {

		float diff = A[i] - B[i];
		sum += diff * diff;

	}

	return sqrt(sum);

}

int main(int argc, char *argv[]) {

	if (argc > 3 || argc < 2) {

		printf("Error: Invalid number of arguments\n");
		printf("Usage: %s numThreads [n]\n", argv[0]);
		return EXIT_FAILURE;

	}

	int num_threads = atoi(argv[1]);

	omp_set_dynamic(0);
	omp_set_num_threads(num_threads);

	size_t n = (argc > 2) ? atoi(argv[2]) : DEFAULT_N;

	// Alloc 
	double malloc_start = omp_get_wtime();
	float *A = (float *)malloc(n * sizeof(float));
	float *B = (float *)malloc(n * sizeof(float));
	double malloc_end = omp_get_wtime();

	if (A == NULL || B == NULL) {

		printf("Error: Memory could not be allocated\n");
		return EXIT_FAILURE;

	}

	// Overhead
	double overhead_1 = omp_get_wtime();
	double overhead_2 = omp_get_wtime();

	// Initialization
	double init_start = omp_get_wtime();
	init_vectors(A, B, n);
	double init_end = omp_get_wtime();

	// Sequential
	double secuential_start = omp_get_wtime();
	double dist_secuential = euclidean_distance_secuential(A, B, n);
	double secuential_end = omp_get_wtime();

	// Time calculations
	double overhead_time = overhead_2 - overhead_1;

	double malloc_time = malloc_end - malloc_start - overhead_time;
	double init_time = init_end - init_start - overhead_time;
	double secuential_time = secuential_end - secuential_start - overhead_time;

	// Paralel options
	for (int i = 0; i < NUM_SCHEDULES; i++) {

		for (int j = 0; j < NUM_CHUNK_SIZES; j++) {

			omp_set_schedule(scheduling_types[i], chunk_sizes[j]);

			// Automatic reduction
			double automatic_start = omp_get_wtime();
			double dist_reduction = euclidean_distance_automatic_reduction(A, B, n);
			double automatic_end = omp_get_wtime();

			// Manual reduction
			double manual_start = omp_get_wtime();
			double dist_manual = euclidean_distance_manual_reduction(A, B, n);
			double manual_end = omp_get_wtime();

			if (fabs(dist_reduction - dist_manual) > 1e-6 || fabs(dist_reduction - dist_secuential) > 1e-6) {

				printf("Error: Inconsistent distance calculation \n");
				return EXIT_FAILURE;

			}

			double automatic_time = automatic_end - automatic_start - overhead_time;
			double manual_time = manual_end - manual_start - overhead_time;

			double total_time = malloc_time + init_time + secuential_time + automatic_time + manual_time;

			printf("PAE,%d,%ld,%d,%d,%f,%f,%f,%f,%f,%f,%f,PAE\n", num_threads, n, scheduling_types[i], chunk_sizes[j], overhead_time, malloc_time, init_time, automatic_time, manual_time, secuential_time, total_time);

		}

	}

	// Free memory
	free(A);
	free(B);

	return EXIT_SUCCESS;

}
