#include <stdio.h>
#include <cuda_runtime.h>

#define SIZE 8192
#define BYTES SIZE * sizeof(int)

#define BLOCK_SIZE 256
#define GRID_SIZE (SIZE + BLOCK_SIZE - 1) / BLOCK_SIZE

// Error checking function
void checkCUDAError(const char* msg) {

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {

		fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);

	}

}

// Kernel para inicializar el array
__global__ void initArray(int* x, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {

		x[idx] = idx;

	}

}

// Kernel para invertir el array
__global__ void invertArray(int* x, int* y, int size) {

	int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < size) {

		y[size - 1 - idx] = x[idx];

	}

}

// Function to verify result
bool verifyResult(int* y, int size) {

	for (int i = 0; i < size; i++) {

		if (y[i] != size - 1 - i) {

			printf("Error en la posición %d: esperado %d, obtenido %d\n", i, size - 1 - i, y[i]);
			return false;

		}

	}

	return true;

}

int main() {

	// Allocate host memory
	int* h_y = (int*)malloc(BYTES);

	// Allocate device memory
	int *d_x, *d_y;

	cudaMalloc(&d_x, BYTES);
	cudaMalloc(&d_y, BYTES);

	checkCUDAError("cudaMalloc failed");

	// Initialize array on GPU
	initArray<<<GRID_SIZE, BLOCK_SIZE>>>(d_x, SIZE);
	cudaDeviceSynchronize();

	checkCUDAError("Kernel initialization failed");

	// Create CUDA events for timing
	cudaEvent_t start;
	cudaEvent_t stop;

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Record start time
	cudaEventRecord(start);

	// Invert array
	invertArray<<<GRID_SIZE, BLOCK_SIZE>>>(d_x, d_y, SIZE);
	cudaDeviceSynchronize();

	// Record stop time
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	// Calculate elapsed time
	float milliseconds = 0;

	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("Execution time: %.3f ms\n", milliseconds);

	// Check for errors
	checkCUDAError("Kernel execution failed");

	// Copy result back to host
	cudaMemcpy(h_y, d_y, BYTES, cudaMemcpyDeviceToHost);

	checkCUDAError("cudaMemcpy failed");

	// Verify result

	if (verifyResult(h_y, SIZE)) {

		printf("Nice!\n");

	} else {

		printf("F!\n");

	}

	// Cleanup
	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	cudaFree(d_x);
	cudaFree(d_y);

	checkCUDAError("cudaFree failed");

	free(h_y);

	return EXIT_SUCCESS;

}