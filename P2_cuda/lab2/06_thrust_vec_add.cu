/*
 * Programación de Arquitecturas Emerxentes [G4011452] Labs
 * Last update: 14/03/2022
 * 
 * Author: Pablo Quesada Barriuso
 * 
 * Thrust is a parallel algorithms library which resembles the C++ Standard 
 * Template Library (STL). Thrust's high-level interface greatly enhances 
 * programmer productivity while enabling performance portability between GPUs
 * and multicore CPUs. Interoperability with established technologies (such as 
 * CUDA, TBB, and OpenMP) facilitates integration with existing software. 
 * Develop high-performance applications rapidly with Thrust!
 *
 * This work is licensed under a Creative Commons
 * Attribution-NonCommercial-ShareAlike 4.0 International.
 * http:// https://creativecommons.org/licenses/by-nc-sa/4.0/
 * 
 * THE AUTHOR MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR
 * IMPLIED WARRANTY OF ANY KIND.  THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL,
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE
 * OR PERFORMANCE OF THIS SOURCE CODE.
 */
#include <thrust/device_vector.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <iostream>

int main(void)
{
    // pointer and dimension for host memory
    int n, dim = 262144;
    
    // pointers for host memory
    thrust::host_vector<float> h_a(dim);
    thrust::host_vector<float> h_b(dim);
    thrust::host_vector<float> h_c(dim);
    
    // initialize input data in host
    for (n=0; n<dim; n++)
    {
        h_a[n] = (float) n;
        h_b[n] = (float) n;
    }
    
    // allocate and initialize device memory
    thrust::device_vector<float> d_A(dim);
    thrust::device_vector<float> d_B(dim);   
    thrust::device_vector<float> d_C(dim);
    
    // transfer data to the device
    d_A = h_a;
    d_B = h_b;

    // compute vector sum C = A+B
    thrust::transform(d_A.begin(), d_A.end(),
        d_B.begin(),
        d_C.begin(),
        thrust::plus<float>());
       
    
    // transfer data back to host
    h_c = d_C;

    // verify the data on the host is correct
    for (n=0; n<dim; n++)
    {
        assert(h_c[n] == h_a[n] + h_b[n]);
    }

    // free host memory pointers
    // done by Thrust

    // If the program makes it this far, then the results are correct and
    // there are no run-time errors.  Good work!
    printf("Correct!\n");
}

// nvcc 06_thrust_vec_add.cu -o 06_thrust_vec_add
