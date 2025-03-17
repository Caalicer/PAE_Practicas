#include <stdio.h>
#include <cuda_runtime.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>

#define DEFAULT_ALPHA 2.0
#define DEFAULT_N ((6L * 1024 * 1024 * 1024) / sizeof(double))

double get_time() {

	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return ts.tv_sec + ts.tv_nsec / 1.0e9;

}

void checkCUDAError(const char *msg) {

	cudaError_t err = cudaGetLastError();

	if (cudaSuccess != err) {

		fprintf(stderr, "Cuda error: %s: %s in %s at line %d.\n", msg, cudaGetErrorString(err), __FILE__, __LINE__);
		exit(EXIT_FAILURE);

	}

}

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

void cpu_daxpy(double* x, double* y, double a, size_t N) {

	for (size_t i = 0; i < N; i++) {

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

	double* cpu_x;
	double* cpu_y;

	cudaDeviceProp prop;

	cudaGetDeviceProperties(&prop, 0);

	checkCUDAError("Getting device properties");

	double start_alloc, start_alloc2, end_alloc;
	double start_init, end_init;
	double start_daxpy, end_daxpy;
	double start_dh, end_dh;
	double overhead;

	start_alloc = get_time();
	start_alloc2 = get_time();

	cudaMallocManaged(&x, n * sizeof(double));
	cudaMallocManaged(&y, n * sizeof(double));

	cpu_x = (double*)malloc(n * sizeof(double));
	cpu_y = (double*)malloc(n * sizeof(double));

	if (cpu_x == NULL || cpu_y == NULL) {

		fprintf(stderr, "Failed to allocate host memory\n");
		return EXIT_FAILURE;

	}

	end_alloc = get_time();

	start_init = get_time();

	init<<<blocksPerGrid, threadsPerBlock>>>(x, y, n);

	cudaDeviceSynchronize();

	end_init = get_time();

	checkCUDAError("Init");

	start_daxpy = get_time();

	daxpy<<<blocksPerGrid, threadsPerBlock>>>(x, y, alpha, n);

	cudaDeviceSynchronize();

	end_daxpy = get_time();

	checkCUDAError("DAXPY");

	for (size_t i = 0; i < n; i++) {

		cpu_x[i] = i;
		cpu_y[i] = i;

	}

	start_dh = get_time();

	cudaMemPrefetchAsync(y, n * sizeof(double), cudaCpuDeviceId);
	cudaDeviceSynchronize();

	end_dh = get_time();

	cpu_daxpy(cpu_x, cpu_y, alpha, n);

	int errors = 0;

	for (size_t i = 0; i < n; i++) {

		double diff = fabs(y[i] - cpu_y[i]);

		if (diff > 1e-10) {

			errors++;

		}

	}

	int maxBlocksPerSM;

	cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocksPerSM, init, threadsPerBlock, 0);

	float occupancy = (float)(maxBlocksPerSM * threadsPerBlock) / prop.maxThreadsPerMultiProcessor;

	overhead = start_alloc2 - start_alloc;
	double alloc_time = end_alloc - start_alloc - overhead;
	double init_time = end_init - start_init - overhead;
	double daxpy_time = end_daxpy - start_daxpy - overhead;
	double dh_time = end_dh - start_dh - overhead;
	double total_time = alloc_time + init_time + daxpy_time + dh_time;

	printf("PAE,%d,%d,%d,%f,%.12f,%.12f,%.12f,%.12f,%.12f,%.12f,%ld,%f,PAE\n", threadsPerBlock, blocksPerGrid, maxBlocksPerSM, occupancy, overhead, alloc_time, init_time, daxpy_time, dh_time, total_time, n, alpha);

	if (errors == 0) {

		printf("Verification PASSED! CPU and GPU results match.\n");

	} else {

		printf("Verification FAILED! %d errors found.\n", errors);

	}

	cudaFree(x);
	checkCUDAError("Freeing x");

	cudaFree(y);
	checkCUDAError("Freeing y");

	free(cpu_x);
	free(cpu_y);

	return EXIT_SUCCESS;

}