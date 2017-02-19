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


   for(j=0; j<(size/8); j++)
     {
      val= _mm256_loadu_ps(&array[j]);
      accum= _mm256_add_ps(val, accum);
     }

   accum=   _mm256_hadd_ps ( accum,accum);
   accum=   _mm256_hadd_ps ( accum, accum);
   accum=   _mm256_hadd_ps ( accum, accum);


  return accum;

}
int main(void)
 {

   #pragma omp parallel
   {
    int j;
    uint32_t size=250;
    uint32_t size_bytes= size *4;
   static  float array[10000];
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
            val= read_memory_avx(array,250);

        }
   auto end_time = chrono::high_resolution_clock::now();
  
    cout <<"The time in microseconds is "<< chrono::duration_cast<chrono::microseconds>(end_time - start_time).count()<<endl ;
     calculation=( chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() * 10);
     bandwidth= (size_bytes/calculation);
    cout<<" The total bandwidth is"<< bandwidth<<"GBs";


}
        return 0;

 }
