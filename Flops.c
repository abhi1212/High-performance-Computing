#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <sys/time.h>
#include <omp.h>
#include <chrono>
#include<iostream>
using namespace std;

int main(void) {


    __m256 a = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0,6.0,7.0,8.0);
    __m256 b = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0,6.0,7.0,8.0);
    __m256 c = _mm256_set_ps(1.0, 2.0, 3.0, 4.0, 5.0,6.0,7.0,8.0);
     __m256 d;
     __m256 e;
   volatile long long i;
#pragma omp parallel
{
 // cout << chrono::high_resolution_clock::period::den << endl;
        auto start_time = chrono::high_resolution_clock::now();

    for (i=0; i < 10000000; i++)
    {
      d= _mm256_fmadd_ps(a, b, c);
      e=  d+d;
      }

    auto end_time = chrono::high_resolution_clock::now();
cout <<"The time in microseconds is"<< chrono::duration_cast<chrono::microseconds>(end_time - start_time).count()<<endl;

        }
                return 0;

        }