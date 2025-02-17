#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 262144

// Kernel para la suma de dos vectores

__global__ void vectorAdd(float *a, float *b, float *c, int n) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < n) {

		c[idx] = a[idx] + b[idx];

	}

}

int main(int argc, char **argv) {

	// Punteros para los vectores en el host (CPU)
	float *h_a, *h_b, *h_c;

	// Punteros para los vectores en el device (GPU)
	float *d_a, *d_b, *d_c;

	size_t size = N * sizeof(float);

	// Reserva de memoria en el host
	h_a = (float *)malloc(size);
	h_b = (float *)malloc(size);
	h_c = (float *)malloc(size);

	// Inicialización secuencial de los vectores en el host
	for (int i = 0; i < N; i++) {

		h_a[i] = (float)i;
		h_b[i] = (float)(N - i);

	}

	// Reserva de memoria en la GPU
	cudaMalloc((void **)&d_a, size);
	cudaMalloc((void **)&d_b, size);
	cudaMalloc((void **)&d_c, size);

	// Copia de datos desde el host a la GPU
	cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

	dim3 dimBlock(256);
	dim3 dimGrid((N + dimBlock.x - 1) / dimBlock.x);

	// Lanzamiento del kernel
	vectorAdd<<<dimGrid, dimBlock>>>(d_a, d_b, d_c, N);

	// Copiar el resultado de la GPU al host
	cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

	// Verificar resultados
	for (int i = 0; i < N; i++) {

		if (h_c[i] != (h_a[i] + h_b[i])) {

			printf("Error en la posición %d: %f != %f\n", i, h_c[i], h_a[i] + h_b[i]);
			break;

		}

	}

	// Liberar memoria en GPU
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	// Liberar memoria en CPU
	free(h_a);
	free(h_b);
	free(h_c);

	printf("Suma de vectores completada correctamente.\n");

	return 0;

}
