#include <iostream>
#include <chrono>
#include "cuda_runtime.h"
#include "cuda.h"
#include "device_launch_parameters.h"
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



__device__ float polynomial (float x, float* poly, int degree) {
  float out = 0.;
  float xtothepowerof = 1.;
  for (int i=0; i<=degree; ++i) {
    out += xtothepowerof*poly[i];
    xtothepowerof *= x;
   // printf("Out values are %f\n",out);
  }
  return out;
}

__global__ void polynomial_expansion (float* poly, int degree,
                           int n, float* array,int iter)
{
    int j;
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<n){
    array[i] = polynomial (array[i], poly, degree);}

}





inline  void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
int main (int argc, char* argv[]) {

  if(argc!=5)
  {
        printf("Not enough arguments");
        return 0;
  }

/* Take command line arguments*/

  int n = atoi(argv[1]); //TODO: atoi is an unsafe function
  printf("The size of array is %d\n",n);
  int degree = atoi(argv[2]);
  printf("The degree is %d\n",degree);
  int nbiter = atoi(argv[3]);
  printf("The iterations are %d\n",nbiter);
  int threadsperblock=atoi(argv[4]);
  printf("Threads per block are %d\n",threadsperblock);
  int totalblocks= (int) n/threadsperblock;

  if(n%threadsperblock!=0)
  {
        totalblocks= totalblocks+1;

  }
   printf("Total blocks are %d\n",totalblocks);


  int size= sizeof(int);
 /* Initialize and allocate memory for device*/
 int *d_n;
 int *d_degree;
 int *d_nbiter;
 HANDLE_ERROR( cudaMalloc((void **)&d_n, size));
 HANDLE_ERROR( cudaMalloc((void **)&d_degree, size));
 HANDLE_ERROR( cudaMalloc((void **)&d_nbiter, size));
 
/* Declare arrays*/

  float* array = new float[n];
  if(array==NULL)
        {
                printf("Malloc failed");
                exit(0);
        }

  float* poly = new float[degree+1];

  for (int i=0; i<n; ++i)
    array[i] = 1.;

  for (int i=0; i<degree+1; ++i)
    poly[i] = 1.;


/* Initialize and allocate memory for arrays on device*/

  float* d_array;
  float* d_poly;

  HANDLE_ERROR( cudaMalloc( (void**)&d_array, n * sizeof(float) ) );
  HANDLE_ERROR( cudaMalloc( (void**)&d_poly, (degree+1) * sizeof(float) ) );

  cudaDeviceSynchronize();
/* Copy the memory from Host to Device*/
   std::chrono::time_point<std::chrono::system_clock> begin, end;


   HANDLE_ERROR( cudaMemcpy((void**)  d_array,array , n * sizeof(float),cudaMemcpyHostToDevice ) );
   HANDLE_ERROR(  cudaMemcpy((void**)  d_n, &n,size,cudaMemcpyHostToDevice ) );
   HANDLE_ERROR(  cudaMemcpy((void**)  d_degree, &degree,size,cudaMemcpyHostToDevice ) );
   HANDLE_ERROR(  cudaMemcpy((void**)  d_nbiter, &nbiter,size,cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy((void**)  d_array,array , n * sizeof(float),cudaMemcpyHostToDevice ) );
   HANDLE_ERROR( cudaMemcpy((void**)   d_poly ,poly , (degree+1) * sizeof(float),cudaMemcpyHostToDevice ) );

   cudaDeviceSynchronize();

   begin = std::chrono::system_clock::now();
   for(int k=0 ;k<nbiter; k++){
   polynomial_expansion<<<totalblocks,threadsperblock>>>(d_poly,degree,n,d_array,nbiter);// We are supposed to call the function ibter times
   }
   cudaDeviceSynchronize();
   end = std::chrono::system_clock::now();

    HANDLE_ERROR( cudaMemcpy((void**) array,d_array , n * sizeof(float),cudaMemcpyDeviceToHost ) );

  std::chrono::duration<double> totaltime = (end-begin);

  std::cerr<<array[0]<<std::endl;
  std::cout<<std::fixed<<" For array size " << n <<" The total time required is "<<(totaltime.count()/nbiter)<<std::endl;

  cudaFree(d_array);
  cudaFree(d_poly);

  return 0;
}
                     
