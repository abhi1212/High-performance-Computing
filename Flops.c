#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

long timediff(clock_t t1, clock_t t2)
 {
    long elapsed;
    elapsed = ((double)t2 - t1) / CLOCKS_PER_SEC * 1000;
    return elapsed;
}




int main(void)

{
    clock_t t1, t2;
    int i;
    float x = 2.7182;
    long elapsed;

 __m256 evens = _mm256_set_ps(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
 __m256 odds = _mm256_set_ps(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);
 __m256 vecc = _mm256_set_ps(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);

   t1 =clock();


    #pragma omp parallel
{

        for(i=0;i<10000000;i++)
        {
        /* Alternately subtract and add the third vector
 *      from the product of the first and second vectors */
  __m256 result = _mm256_fmaddsub_ps(evens, odds, vecc);

    }
}
    t2 = clock();


    elapsed = timediff(t1, t2);


    printf("elapsed: %ld ms\n", elapsed);
