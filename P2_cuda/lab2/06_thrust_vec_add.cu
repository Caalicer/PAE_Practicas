#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <iostream>

__global__ void vectorAdd(float *a, float *b, float *c, int n) {

	int idx = threadIdx.x + blockIdx.x * blockDim.x;

	if (idx < n) {

		c[idx] = a[idx] + b[idx];

	}

}

int main(void) {

	// pointer and dimension for host memory
	int n, dim = 262144;
	size_t size = dim * sizeof(float);

	// pointers for host memory
	thrust::host_vector<float> h_a(dim);
	thrust::host_vector<float> h_b(dim);
	thrust::host_vector<float> h_c(dim);

	// initialize input data in host
	for (n = 0; n < dim; n++) {
		h_a[n] = (float)n;
		h_b[n] = (float)n;
	}

	// allocate and initialize device memory
	thrust::device_vector<float> d_A(dim);
	thrust::device_vector<float> d_B(dim);
	thrust::device_vector<float> d_C(dim);

	// ############################################################
	// New Code

	// Use explicit cudaMemcpy to copy data from host to device
	cudaMemcpy(thrust::raw_pointer_cast(d_A.data()), h_a.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(thrust::raw_pointer_cast(d_B.data()), h_b.data(), size, cudaMemcpyHostToDevice);

	// Launch initializeKernel kernel on the GPU
	int threadsPerBlock = 256;
	int blocksPerGrid = (dim + threadsPerBlock - 1) / threadsPerBlock;

	// Launch initializeKernel kernel on the GPU instead of using transform
	vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(
		thrust::raw_pointer_cast(d_A.data()),
		thrust::raw_pointer_cast(d_B.data()),
		thrust::raw_pointer_cast(d_C.data()),
		dim
	);

	// transfer data back to host
	cudaMemcpy(h_c.data(), thrust::raw_pointer_cast(d_C.data()), size, cudaMemcpyDeviceToHost);

	// ############################################################

	// ############################################################
	// Old Code
	// d_A = h_a;
	// d_B = h_b;
	// thrust::transform(d_A.begin(), d_A.end(), d_B.begin(), d_C.begin(), thrust::plus<float>());
	// h_c = d_C;
	// ############################################################

	// verify the data on the host is correct
	for (n = 0; n < dim; n++) {
		assert(h_c[n] == h_a[n] + h_b[n]);
	}

	// free host memory pointers
	// done by Thrust

	// If the program makes it this far, then the results are correct and
	// there are no run-time errors.  Good work!
	printf("Correct!\n");

}