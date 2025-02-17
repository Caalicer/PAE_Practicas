#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>

#define N 1000000

void inicializar_vectores(float *a, float *b, int n) {

	for (int i = 0; i < n; i++) {

		a[i] = (float)rand() / RAND_MAX;
		b[i] = (float)rand() / RAND_MAX;

	}

}

void suma_vectores(float *a, float *b, float *c, int n) {

	// Directiva para paralelizar el bucle en GPU
	#pragma acc kernels copyin(a[0:n], b[0:n]) copyout(c[0:n])
	for (int i = 0; i < n; i++) {

		c[i] = a[i] + b[i];

	}

}

void verificar_resultado(float *a, float *b, float *c, int n) {

	float epsilon = 1.0e-5;

	for (int i = 0; i < n; i++) {

		float expected = a[i] + b[i];

		if (abs(c[i] - expected) > epsilon) {

			printf("Error en la posición %d: %.6f != %.6f\n", i, c[i], expected);
			return;

		}

	}

	printf("NICE!\n");

}

int main() {

	float *a = (float*)malloc(N * sizeof(float));
	float *b = (float*)malloc(N * sizeof(float));
	float *c = (float*)malloc(N * sizeof(float));

	// Inicializar vectores con valores aleatorios
	inicializar_vectores(a, b, N);

	// Realizar la suma de vectores usando OpenACC
	suma_vectores(a, b, c, N);

	// Verificar el resultado
	verificar_resultado(a, b, c, N);

	// Liberar memoria
	free(a);
	free(b);
	free(c);

	return 0;

}