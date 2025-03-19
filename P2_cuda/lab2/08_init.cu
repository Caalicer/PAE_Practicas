#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Error checking function
void checkCUDAError(const char* msg) {

	cudaError_t err = cudaGetLastError();

	if (err != cudaSuccess) {

		fprintf(stderr, "CUDA error: %s: %s\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);

	}

}

// Kernel to initialize matrix
__global__ void initMatrix(int* matrix, size_t M, size_t N) {

	// Calculate global thread indices
	size_t col = blockIdx.x * blockDim.x + threadIdx.x;
	size_t row = blockIdx.y * blockDim.y + threadIdx.y;

	// Check if thread is within matrix bounds
	if (row < M && col < N) {

		size_t idx = row * N + col;
		matrix[idx] = row * N + col;

	}

}

int main() {

	// Print device properties
	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, 0);

	printf("Device: %s\n", prop.name);
	printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
	printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
	printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
	printf("Number of SMs: %d\n\n", prop.multiProcessorCount);
	printf("Warp size: %d\n", prop.warpSize);
	printf("Max warps per SM: %d\n", prop.maxThreadsPerMultiProcessor / prop.warpSize);

	// Calculate matrix dimensions for 1 GiB matrix
	// 1 GiB = 2^30 bytes
	const size_t ONE_GiB = 1 << 30;
	const size_t TYPE_SIZE = sizeof(int);
	const size_t TOTAL_ELEMENTS = ONE_GiB / TYPE_SIZE;

	// Make matrix square
	const size_t M = sqrt(TOTAL_ELEMENTS);
	const size_t N = M;

	printf("Matrix dimensions: M = %zu, N = %zu\n", M, N);

	// Allocate device memory
	int* d_matrix;

	cudaMalloc(&d_matrix, M * N * TYPE_SIZE);

	checkCUDAError("cudaMalloc failed");

	// Test different block sizes optimized for A100
	// A100 has max 1024 threads per block and 2048 threads per SM
	dim3 blockSizes[] = {
		dim3(32, 1),	// warp size (32 threads)
		dim3(32, 32),	// 1024 threads (maximum square)
		dim3(128, 8),	// 1024 threads (rectangular)
		dim3(256, 4),	// 1024 threads (wide)
		dim3(64, 16),	// 1024 threads (balanced)
		dim3(16, 32)	// 512 threads (half-maximum, might be better for occupancy)
	};

	for (dim3 blockSize : blockSizes) {

		// Calculate grid dimensions
		dim3 gridSize(
			ceil(N / blockSize.x),
			ceil(M / blockSize.y)
		);

		printf("\nTesting block size: %d x %d (Total threads per block: %d)\n", blockSize.x, blockSize.y, blockSize.x * blockSize.y);
		printf("Grid size: %d x %d\n", gridSize.x, gridSize.y);

		// Record start time
		cudaEvent_t start;
		cudaEvent_t stop;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);

		// Warm-up run
		initMatrix<<<gridSize, blockSize>>>(d_matrix, M, N);
		cudaDeviceSynchronize();

		// Timed run
		cudaEventRecord(start);

		initMatrix<<<gridSize, blockSize>>>(d_matrix, M, N);
		cudaDeviceSynchronize();

		cudaEventRecord(stop);
		cudaEventSynchronize(stop);

		float milliseconds = 0;

		cudaEventElapsedTime(&milliseconds, start, stop);

		printf("Execution time: %.3f ms\n", milliseconds);

		// Get occupancy information
		int maxBlocksPerSM;

		cudaOccupancyMaxActiveBlocksPerMultiprocessor(
			&maxBlocksPerSM,
			initMatrix,
			blockSize.x * blockSize.y,
			0
		);

		float occupancy = (float)(maxBlocksPerSM * blockSize.x * blockSize.y) / prop.maxThreadsPerMultiProcessor;

		printf("Blocks per SM: %d\n", maxBlocksPerSM);
		printf("Theoretical occupancy: %.2f\n", occupancy);

		cudaEventDestroy(start);
		cudaEventDestroy(stop);

	}

	// Cleanup
	cudaFree(d_matrix);

	checkCUDAError("cudaFree failed");

	return EXIT_SUCCESS;

}