#include <hip/hip_runtime.h>

/* Macro for checking GPU API return values */
#define hip_check(call)                                                                        \
do{                                                                                             \
    hipError_t gpuErr = call;                                                                   \
    if(hipSuccess != gpuErr){                                                                   \
        printf("GPU API Error - %s:%d: '%s'\n", __FILE__, __LINE__, hipGetErrorString(gpuErr)); \
        exit(1);                                                                                \
    }                                                                                           \
}while(0)


__global__ void cube(double *input, double *output,int offset, int elements_per_stream)
{
  size_t tid = blockIdx.x*blockDim.x+threadIdx.x;
  size_t gstride = blockDim.x * gridDim.x;

  // Span all elements assigned to this stream
  for (size_t id = tid + offset; id < offset + elements_per_stream; id += gstride)
    for (size_t i = 0; i < 1000; ++i)
      output[id] = input[id]*input[id]*input[id];
}


void hip_cube(const int gridSizePerStream, const int blockSize, double *input, double *output, const int offset, 
              const int elements_per_stream, hipStream_t stream) {
   cube<<<gridSizePerStream, blockSize, 0, stream>>>(input, output, offset, elements_per_stream);
}

void hip_create_streams(){}

void hip_host_malloc(double* pointer, const size_t bytes){
  hip_check( hipHostMalloc(&pointer, bytes) );
}

void hip_malloc(double* pointer, const size_t bytes){
  hip_check( hipMalloc(&pointer, bytes) );
}

void hip_host_free(double* pointer){
  hip_check( hipHostFree(pointer) );
}

void hip_free(double* pointer){
  hip_check( hipFree(pointer) );
}

void hip_map_to(){

}

void hip_map_from(){

}
