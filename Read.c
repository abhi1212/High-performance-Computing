#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <sys/time.h>
#include <omp.h>
#include <chrono>
#include<iostream>
using namespace std;


  __m256 read_memory_avx(float* array, int size)
{


        //__m256* varray = (__m256*) array;
        uint32_t j;

    __m256 accum = _mm256_set_ps(0,0,0,0,0,0,0,0);
    __m256 val = _mm256_set_ps(0,0,0,0,0,0,0,0);

   // printf("array values are %f\n%f\n", array[0], array[5]);

   for(j=0; j<(size/8); j=j+8)
     {
      val= _mm256_loadu_ps(&array[j]);
      accum= _mm256_add_ps(val, accum);
     }

   accum=  _mm256_hadd_ps ( accum,accum);
   accum=  _mm256_hadd_ps ( accum, accum);
   accum=  _mm256_hadd_ps ( accum, accum);


  return accum;

}
int main(void)
 {

   #pragma omp parallel
   {
    int j;
    float *array;
    uint32_t size=5000000;
    uint32_t size_bytes= size *4;
     array =(float*) malloc(size*sizeof(float));
    double bandwidth= 0;
    double calculation=0;

    for(j=0; j<size;j++)
   {
       array[j]= 1;

   }
    uint32_t i;
    __m256 val= _mm256_set_ps(0,0,0,0,0,0,0,0);

    auto start_time = chrono::high_resolution_clock::now();
      for(i=0; i<100; i++)
        {
            val= read_memory_avx(array,size);

        }

          // printf("value of arrays are %f\n%f\n%f\n", array[1],array[2],array[3]);

   auto end_time = chrono::high_resolution_clock::now();
  // float* val1 = (float*)&val;
   // printf("Values are %f\n %f\n %f\n %f\n %f\n %f\n %f\n %f\n",
   //val1[0], val1[1], val1[2], val1[3], val1[4], val1[5], val1[6], val1[7]);
  // auto end_time = chrono::high_resolution_clock::now();

  //int printval= val[0];
   //cout<<printval;
    cout <<"The time in microseconds is "<< chrono::duration_cast<chrono::microseconds>(end_time - start_time).count()<<endl ;
  //  cout<<"Hii"<<chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
     calculation=( chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() * 10);
   //  cout<<"Calculation is " <<calculation;
     bandwidth= (size_bytes/calculation);
    cout<<" The total bandwidth is"<< bandwidth<<"GBs";
  }

return 0
}


