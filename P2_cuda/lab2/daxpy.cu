#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

#define DEFAULT_ALPHA 2.0
#define DEFAULT_N ((6L * 1024 * 1024 * 1024) / sizeof(double))
#define MAX_BLOCKS_PER_SM 16

__global__ void init(double* x, double* y, size_t N) {

	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N) {

		x[idx] = idx;
		y[idx] = idx;

	}

}

__global__ void daxpy(double* x, double* y, double a, size_t N) {

	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N) {

		y[idx] = a * x[idx] + y[idx];

	}

}

int main(int argc, char** argv) {

	if (argc < 3) {

		fprintf(stderr, "Usage: %s <blockSizeX> <blockSizeY> [alpha] [n]\n", argv[0]);
		return EXIT_FAILURE;

	}

	int blockSizeX = atoi(argv[1]);
	int blockSizeY = atoi(argv[2]);

	double alpha = (argc > 3) ? atof(argv[3]) : DEFAULT_ALPHA;
	size_t n = (argc > 4) ? atoll(argv[4]) : DEFAULT_N;

	double* x;
	double* y;

	struct timeval startAlloc, startAlloc2, endAlloc;
	struct timeval startInit, endInit;
	struct timeval startDaxpy, endDaxpy;

	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, 0);

	gettimeofday(&startAlloc, NULL);
	gettimeofday(&startAlloc2, NULL);

	cudaMallocManaged(&x, n * sizeof(double));
	cudaMallocManaged(&y, n * sizeof(double));

	gettimeofday(&endAlloc, NULL);

	dim3 blockSize(blockSizeX, blockSizeY);
	dim3 grid((n + blockSize.x - 1) / blockSize.x, 1);

	gettimeofday(&startInit, NULL);

	init<<<grid, blockSize>>>(x, y, n);
	cudaDeviceSynchronize();

	gettimeofday(&endInit, NULL);

	gettimeofday(&startDaxpy, NULL);

	daxpy<<<grid, blockSize>>>(x, y, alpha, n);
	cudaDeviceSynchronize();

	gettimeofday(&endDaxpy, NULL);

	int maxBlocksPerSM;

	cudaOccupancyMaxActiveBlocksPerMultiprocessor(
		&maxBlocksPerSM,
		init,
		blockSize.x * blockSize.y,
		0
	);

	float occupancy = (float)(maxBlocksPerSM * blockSize.x * blockSize.y) / prop.maxThreadsPerMultiProcessor;

	double overhead = (startAlloc2.tv_sec - startAlloc.tv_sec) + (startAlloc2.tv_usec - startAlloc.tv_usec) / 1e6;
	double alloc_time = (endAlloc.tv_sec - startAlloc2.tv_sec) + (endAlloc.tv_usec - startAlloc2.tv_usec) / 1e6 - overhead;
	double init_time = (endInit.tv_sec - startInit.tv_sec) + (endInit.tv_usec - startInit.tv_usec) / 1e6;
	double daxpy_time = (endDaxpy.tv_sec - startDaxpy.tv_sec) + (endDaxpy.tv_usec - startDaxpy.tv_usec) / 1e6;

	printf("PAE,%d,%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%ld,%f\n", blockSize.x, blockSize.y, grid.x, grid.y, blockSize.x * blockSize.y, maxBlocksPerSM, occupancy, overhead, alloc_time, init_time, daxpy_time, n, alpha);

	cudaFree(x);
	cudaFree(y);

	return EXIT_SUCCESS;

}