#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define M 16384  // (M × M) ≈ 1 GiB (con sizeof(int) = 4 bytes)

int matrix[M][M];

void init_matrix_parallel_for(int schedule_type, int chunk_size) {

	double start = omp_get_wtime();

	#pragma omp parallel for schedule(runtime)
	for (int i = 0; i < M; i++) {

		for (int j = 0; j < M; j++) {

			matrix[i][j] = i * M + j;

		}

	}

	double end = omp_get_wtime();
	printf("Parallel for (schedule=%d, chunk=%d) time: %f seconds\n", schedule_type, chunk_size, end - start);

}

void init_matrix_task() {

	double start = omp_get_wtime();

	#pragma omp parallel
	{

		#pragma omp single
		{

			for (int i = 0; i < M; i++) {

				#pragma omp task
				{

					for (int j = 0; j < M; j++) {

						matrix[i][j] = i * M + j;

					}

				}

			}

		}

	}

	double end = omp_get_wtime();
	printf("Parallel task time: %f seconds\n", end - start);

}

void init_matrix_taskloop() {

	double start = omp_get_wtime();

	#pragma omp parallel
	{

		#pragma omp single
		{

			#pragma omp taskloop
			for (int i = 0; i < M; i++) {

				for (int j = 0; j < M; j++) {

					matrix[i][j] = i * M + j;

				}

			}

		}

	}

	double end = omp_get_wtime();
	printf("Parallel taskloop time: %f seconds\n", end - start);

}

int main() {

	omp_set_num_threads(8); // Ajustar número de hilos

	printf("Testing parallel for with different scheduling policies\n");

	// Probar con política STATIC
	printf("Testing STATIC\n");
	omp_set_schedule(omp_sched_static, 256);
	init_matrix_parallel_for(0, 256);

	// Probar con política DYNAMIC
	printf("Testing DYNAMIC\n");
	omp_set_schedule(omp_sched_dynamic, 256);
	init_matrix_parallel_for(1, 256);

	// Probar con política GUIDED
	printf("Testing GUIDED\n");
	omp_set_schedule(omp_sched_guided, 256);
	init_matrix_parallel_for(2, 256);

	// Ejecutar con tareas
	printf("\nTesting parallel task\n");
	init_matrix_task();

	// Ejecutar con taskloop
	printf("\nTesting parallel taskloop\n");
	init_matrix_taskloop();

	return 0;

}
