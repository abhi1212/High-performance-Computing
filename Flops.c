
#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <sys/time.h>
#include <omp.h>
#include <chrono>
#include <iostream>
using namespace std;

int main(void)
 {

        #pragma omp parallel
        {
                uint32_t i;

    __m256 evens = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0,6.0,7.0,8.0);
    __m256 odds = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0,6.0,7.0,8.0);
    __m256 vecc= _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0,6.0,7.0,8.0);


   // cout << chrono::high_resolution_clock::period::den << endl;
        auto start_time = chrono::high_resolution_clock::now();

    for (i=0; i < 100000000; i++)
    {

           __m256 result = _mm256_fmaddsub_ps(evens, odds, vecc);
          __m256 result1 = _mm256_fmaddsub_ps( result, odds, vecc);
          __m256 result2= _mm256_fmaddsub_ps(evens,  result1, vecc);
          __m256 result3 = _mm256_fmaddsub_ps(evens,  result2, vecc);
          __m256 result4 = _mm256_fmaddsub_ps(evens,  result3, vecc);


  if(i==100)
{
    int* res = (int*)&result;
     printf("%d %d %d %d %d %d %d %d\n",
     res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]);

    int* res1 = (int*)&result1;
    printf("%d %d %d %d %d %d %d %d\n",
    res1[0], res1[1], res1[2], res1[3], res1[4], res1[5], res1[6], res1[7]);

    int* res2 = (int*)&result2;
    printf("%d %d %d %d %d %d %d %d\n",
    res2[0], res2[1], res2[2], res2[3], res2[4], res2[5], res2[6], res2[7]);

    int* res3 = (int*)&result;
    printf("%d %d %d %d %d %d %d %d\n",
    res3[0], res3[1], res3[2], res3[3], res3[4], res3[5], res3[6], res3[7]);
}
     asm("");
    }

        auto end_time = chrono::high_resolution_clock::now();
        cout <<"The time in microseconds is"<< chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() << ":";
}
        return 0;


}
