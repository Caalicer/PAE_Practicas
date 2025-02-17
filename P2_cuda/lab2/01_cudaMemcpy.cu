// includes, system
#include <stdio.h>
#include <assert.h>

///////////////////////////////////////////////////////////////////////////////
// Program main
///////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) {

	// pointer and dimension for host memory
	int n, dim;
	float *h_a, *h_b;

	// pointers for device memory
	float *d_a, *d_b;

	// allocate initialize host memory
	dim = 262144;

	size_t mem_size = dim * sizeof(float);

	h_a = (float *) malloc(mem_size);
	h_b = (float *) malloc(mem_size);

	// initialize input data in host
	for (n=0; n<dim; n++) {
		h_a[n] = (float) n;
	}

	// allocate device memory
	cudaMalloc((void **) &d_a, mem_size);
	cudaMalloc((void **) &d_b, mem_size);

	// host to device memory copy d_a = h_a
	cudaMemcpy(d_a, h_a, mem_size, cudaMemcpyHostToDevice);

	// device to device memory copy d_b = d_a
	cudaMemcpy(d_b, d_a, mem_size, cudaMemcpyDeviceToDevice);

	// device to host copy h_b = d_b
	cudaMemcpy(h_b, d_b, mem_size, cudaMemcpyDeviceToHost);

	// verify the data on the host is correct
	for (n=0; n<dim; n++) {
		assert(h_b[n] == h_a[n]);
	}

	// free device memory pointers d_a and d_b
	cudaFree(d_a);
	cudaFree(d_b);

	// free host memory pointers
	free(h_a);
	free(h_b);

	// If the program makes it this far, then the results are correct and
	// there are no run-time errors.  Good work!
	printf("Correct!\n");

	return 0;

}

// nvcc 01_cudaMemcpy.cu - 01_cudaMemcpy
