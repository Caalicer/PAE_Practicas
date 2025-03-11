#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

#define DEFAULT_ALPHA 2.0
#define DEFAULT_N ((6L * 1024 * 1024 * 1024) / sizeof(double))

__global__ void init(double* x, double* y, size_t N) {

	size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < N) {

		x[idx] = idx;
		y[idx] = idx;

	}

}

__global__ void daxpy(double* x, double* y, double a, size_t N) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int stride = gridDim.x * blockDim.x;

	for (; i < N; i += stride) {

		y[i] = a * x[i] + y[i];

	}

}

int main(int argc, char** argv) {

	if (argc < 2) {

		fprintf(stderr, "Usage: %s <threadsPerBlock> [alpha] [n]\n", argv[0]);
		return EXIT_FAILURE;

	}

	int threadsPerBlock = atoi(argv[1]);

	double alpha = (argc > 2) ? atof(argv[2]) : DEFAULT_ALPHA;
	size_t n = (argc > 3) ? atoll(argv[3]) : DEFAULT_N;

	int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

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

	gettimeofday(&startInit, NULL);

	init<<<blocksPerGrid, threadsPerBlock>>>(x, y, n);

	gettimeofday(&endInit, NULL);

	gettimeofday(&startDaxpy, NULL);

	daxpy<<<blocksPerGrid, threadsPerBlock>>>(x, y, alpha, n);

	gettimeofday(&endDaxpy, NULL);

	int maxBlocksPerSM;

	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, init, threadsPerBlock, 0);

	float occupancy = (float)(maxBlocksPerSM * threadsPerBlock) / prop.maxThreadsPerMultiProcessor;

	double overhead = (startAlloc2.tv_sec - startAlloc.tv_sec) + (startAlloc2.tv_usec - startAlloc.tv_usec) / 1e6;
	double alloc_time = (endAlloc.tv_sec - startAlloc2.tv_sec) + (endAlloc.tv_usec - startAlloc2.tv_usec) / 1e6 - overhead;
	double init_time = (endInit.tv_sec - startInit.tv_sec) + (endInit.tv_usec - startInit.tv_usec) / 1e6  - overhead;
	double daxpy_time = (endDaxpy.tv_sec - startDaxpy.tv_sec) + (endDaxpy.tv_usec - startDaxpy.tv_usec) / 1e6 - overhead;
	double total_time = alloc_time + init_time + daxpy_time;

	printf("PAE,%d,%d,%d,%f,%f,%f,%f,%f,%f,%ld,%f,PAE\n", threadsPerBlock, blocksPerGrid, maxBlocksPerSM, occupancy, overhead, alloc_time, init_time, daxpy_time, total_time, n, alpha);

	cudaFree(x);
	cudaFree(y);

	return EXIT_SUCCESS;

}