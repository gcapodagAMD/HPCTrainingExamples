#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define BLOCKDIM 64

void usage()
{
  printf("Usage: ./interop <nstreams> <blockSize (optional, default=64)>\n");
  exit(1);
  return;
}

void hip_host_malloc(double * pointer, const size_t bytes);

void hip_malloc(double * pointer, const size_t bytes);

void hip_host_free(double * pointer);

void hip_free(double * pointer);

void hip_cube(const int gridSizePerStream, const int blockSize, double *input, double *output, const int offset,
              const int elements_per_stream, hipStream_t stream);

int main( int argc, char* argv[] )
{
    // Read input arg
    if (argc < 2)
      usage();
    int num_streams = atoi(argv[1]);

    // Number of threads in each thread block
    int blockSize = BLOCKDIM;
    if (argc == 3)
      blockSize = atoi(argv[2]);

    // Size of vectors
    int n = 100000000;
    int elements_per_stream = n/num_streams;
    int bytes_per_stream = elements_per_stream*sizeof(double);
    printf("bytes/stream: %f (MB)\n", bytes_per_stream*1.e-6);


    // Host input vectors
    double *h_input1;
    //Host output vector
    double *h_output1;
    //Host output vector for verification
    double *h_verify1;
 
    // Device input vectors
    double *d_input1;
    //Device output vector
    double *d_output1;

    //Creating streams

    hipStream_t streams[num_streams];
    for (int i = 0; i < num_streams; ++i)
    {
       HIP_ASSERT(hipStreamCreate(&streams[i]));	    
    }

   
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
  
    // Allocate page locked memory for these vectors on host
    hip_host_malloc(&h_input1, bytes);
    hip_host_malloc(&h_output1, bytes);
   
    h_verify1 = (double*)malloc(bytes);
   
   printf("Finished allocating vectors on the CPU\n");     
    // Allocate memory for each vector on GPU
   hip_malloc(&d_input1, bytes);
   hip_malloc(&d_output1, bytes);
 
   printf("Finished allocating vectors on the GPU\n");

    int i;
    // Initialize vectors on host
    for( i = 0; i < n; i++ ) {
        h_input1[i] = sin(i);
    }

#pragma omp target teams loop map(tofrom:h_input1[0:n])
    for( i = 0; i < n; i++ ) {
        h_input1[i] = sin(i);
    }

    // Number of thread blocks in grid
#ifdef MI300A_RUN
    int gridSizePerStream = 228; //(int)ceil((float)elements_per_stream/blockSize);
#else
    int gridSizePerStream = 104; //(int)ceil((float)elements_per_stream/blockSize);
#endif
    printf("gridSizePerStream: %d\n", gridSizePerStream);

    hipEventRecord(start);
    // Copy and execute in loop
   
#ifndef SPLIT_DATACOPY_KERNEL_CALLS
    for(int i = 0; i < num_streams; i++)
    {
      int offset = i * elements_per_stream;
      HIP_ASSERT(hipMemcpyAsync(&d_input1[offset],&h_input1[offset],bytes_per_stream,hipMemcpyHostToDevice,streams[i]));
      hip_cube(gridSizePerStream,blockSize,d_input1,d_output1,offset,elements_per_stream,streams[i]);
      HIP_ASSERT(hipMemcpyAsync(&h_output1[offset],&d_output1[offset], bytes_per_stream, hipMemcpyDeviceToHost,streams[i]));
    }
#else
    // split H2D copies and kernel calls into separate loops
    for(int i = 0; i < num_streams; i++)
    {
      int offset = i * elements_per_stream;
      HIP_ASSERT(hipMemcpyAsync(&d_input1[offset],&h_input1[offset],bytes_per_stream,hipMemcpyHostToDevice,streams[i]));
    }
    for(int i = 0; i < num_streams; i++)
    {
      int offset = i * elements_per_stream;
      hip_cube(gridSizePerStream,blockSize,d_input1,d_output1,offset,elements_per_stream,streams[i]);
    }
    for(int i = 0; i < num_streams; i++)
    {
      int offset = i * elements_per_stream;
      HIP_ASSERT(hipMemcpyAsync(&h_output1[offset],&d_output1[offset], bytes_per_stream, hipMemcpyDeviceToHost,streams[i]));
    }
#endif //SPLIT_DATACOPY_KERNEL_CALLS

    hipEventRecord(stop);
    hipEventSynchronize(stop);

    float milliseconds = 0;
    hipEventElapsedTime(&milliseconds, start, stop);
    printf("Time required total (ms) %f\n", milliseconds);
    printf("Finished copying the output vector from the GPU to the CPU\n");

   //Compute for CPU 
    for(i=0; i <n; i++)
    {
    h_verify1[i] = h_input1[i] * h_input1[i] * h_input1[i];
    }


    //Verfiy results
    for(i=0; i <n; i++)
    {
    if (abs(h_verify1[i] - h_output1[i]) > 1e-5) 
     {
     printf("Error at position i %d, Expected: %f, Found: %f \n", i, h_verify1[i], h_output1[i]);
     }  
    }

    printf("Releasing GPU memory\n");
     
    // Release device memory
    HIP_ASSERT(hipFree(d_input1));
    HIP_ASSERT(hipFree(d_output1));
    
    // Release host memory
    printf("Releasing CPU memory\n");
    HIP_ASSERT(hipHostFree(h_input1));
    HIP_ASSERT(hipHostFree(h_output1));
    free(h_verify1);
  
    return 0;
}

