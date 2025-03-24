#include <stdio.h>
#include <omp.h>

void example_taskloop() {

	int N = 10;

	printf("\nTaskloop Example:\n");

	#pragma omp parallel
	{

		#pragma omp single
		{

			#pragma omp taskloop
			for (int i = 0; i < N; i++) {
				printf("Task %d executed by thread %d\n", i, omp_get_thread_num());
			}

		}

	}

}

int main() {

	// Configurar número de hilos y deshabilitar ajuste dinámico
	omp_set_dynamic(0);
	omp_set_num_threads(4);

	printf("Dynamic threads: %d\n", omp_get_dynamic());

	// Uso de omp_get_num_threads
	#pragma omp parallel
	{
		#pragma omp single
		printf("Number of threads: %d\n", omp_get_num_threads());
	}

	// Uso de la cláusula reduction
	int sum = 0;
	#pragma omp parallel for reduction(+:sum)
	for (int i = 1; i <= 10; i++) {

		sum += i;

	}

	printf("Sum using reduction: %d\n", sum);

	// Uso de las cláusulas collapse y linear
	int A[3][3], count = 0;

	#pragma omp parallel for collapse(2) linear(count:1)
	for (int i = 0; i < 3; i++) {

		for (int j = 0; j < 3; j++) {

			A[i][j] = count++;

		}

	}

	printf("Matrix using collapse and linear:\n");

	for (int i = 0; i < 3; i++) {

		for (int j = 0; j < 3; j++) {

			printf("%d ", A[i][j]);

		}

		printf("\n");

	}

	// Uso de task
	printf("\nTask Example:\n");
	#pragma omp parallel
	{

		#pragma omp single
		{

			for (int i = 0; i < 5; i++) {

				#pragma omp task
				printf("Executing task %d in thread %d\n", i, omp_get_thread_num());

			}

		}

	}

	// Uso de taskloop
	example_taskloop();

	return 0;

}
