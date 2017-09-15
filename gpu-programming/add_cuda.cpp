#include <iostream>
#include <math.h>

// Adds two arrays with millions elements each on Nvidia's CUDA

// compile on unix
// $ clang++ add_cuda.cpp -o add_cuda

// function add the elements of two arrays
// add global specifier and CUDA C++ compiler to run on GPU and can be called on CPU
__global__
void add(int n, float *x, float *y)
{
    // y will contain the sum of each elements in both arrays
    // stride through array with parallel threads
    int index = threadIdx.x;
    int stride = blockDim.x;

    for (int i = 0; i < n; i+= stride) {
      y[i] = x[i] + y[i];
    }
}

int main(void)
{
  int N = 1<<20; // 1M Elements, int array space for million elements

  // pointers for both of our arrays with space allocated for both of them
  // through our previously allocated array
  float *x = new float[N];
  float *y = new float[N];

  // ALlocate Unified Mmeory - accessible from CPU and GPU and provides a pointer
  // to access from CPU or GPU
  cudaMallocManaged(&x, N*sizeof(float));
  cudaMallocManaged(&y, N*sizeof(float));

  // initialize x and y arrays on the host
  // fill each array with float values
  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  // Run kernel on 1M elements on the CPU
  // add kernel by using triple anchor syntax
  // Two params, second one is the number of threads, blocks of thread in 32 size
  add<<<1, 256>>>(N,x,y);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();

  // Free memory
  cudaFree(x);
  cudaFree(y);

  return 0;
}
