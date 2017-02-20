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
    __m256 result= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result1=_mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result2= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result3= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result4= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result5= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result6= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result7= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result8= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result9= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result10= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result11=_mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result12= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result13= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result14= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result15= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result16= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result17= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result18= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result19= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result20= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result21= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result22= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result23= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result24= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result25= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result26= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);
    __m256 result27= _mm256_set_ps(0.0, 0.0, 0.0, 0.0, 0.0,0.0,0.0,0.0);


   // cout << chrono::high_resolution_clock::period::den << endl;
        auto start_time = chrono::high_resolution_clock::now();

    for (i=0; i < 100000000; i++)
    {


           result   = _mm256_fmadd_ps(result2, odds, vecc);
           result1  = _mm256_fmadd_ps(result3, odds, vecc);
           result2  = _mm256_fmadd_ps(evens,  result, vecc);
           result3  = _mm256_fmadd_ps(evens,  result1, vecc);
           result4  = _mm256_fmadd_ps(evens,  result2, vecc);
           result5  = _mm256_fmadd_ps(evens,  result1, vecc);
           result6  = _mm256_fmadd_ps(evens,  result5, vecc);
           result7  = _mm256_fmadd_ps(evens,  result6, vecc);
           result8  = _mm256_fmadd_ps(evens,  result4, vecc);
           result9  = _mm256_fmadd_ps(evens,  odds, vecc);
           result10   = _mm256_fmadd_ps(result2, odds, vecc);
           result11  = _mm256_fmadd_ps(result3, odds, vecc);
           result12  = _mm256_fmadd_ps(evens,  result, vecc);
           result13  = _mm256_fmadd_ps(evens,  result1, vecc);
           result14  = _mm256_fmadd_ps(evens,  result2, vecc);
           result15  = _mm256_fmadd_ps(evens,  result1, vecc);
           result16  = _mm256_fmadd_ps(evens,  result5, vecc);
           result17  = _mm256_fmadd_ps(evens,  result6, vecc);
           result18  = _mm256_fmadd_ps(evens,  result4, vecc);
           result19  = _mm256_fmadd_ps(evens,  odds, vecc);
           result20  = _mm256_fmadd_ps(evens,  result1, vecc);
           result21  = _mm256_fmadd_ps(evens,  result5, vecc);
           result22  = _mm256_fmadd_ps(evens,  result6, vecc);
           result23  = _mm256_fmadd_ps(evens,  result4, vecc);
           result24  = _mm256_fmadd_ps(evens,  odds, vecc);
           result25  = _mm256_fmadd_ps(evens,  result5, vecc);
           result26  = _mm256_fmadd_ps(evens,  result6, vecc);
           result27  = _mm256_fmadd_ps(evens,  result4, vecc);
           result28  = _mm256_fmadd_ps(evens,  odds, vecc);



     asm("");
    }
      auto end_time = chrono::high_resolution_clock::now();


    int* res = (int*)&result;
    printf("%d %d %d %d %d %d %d %d\n",
    res[0], res[1], res[2], res[3], res[4], res[5], res[6], res[7]);

    int* res1 = (int*)&result1;
    printf("%d %d %d %d %d %d %d %d\n",
    res1[0], res1[1], res1[2], res1[3], res1[4], res1[5], res1[6], res1[7]);

    int* res2 = (int*)&result2;
    printf("%d %d %d %d %d %d %d %d\n",
    res2[0], res2[1], res2[2], res2[3], res2[4], res2[5], res2[6], res2[7]);

    int* res3 = (int*)&result3;
    printf("%d %d %d %d %d %d %d %d\n",
    res3[0], res3[1], res3[2], res3[3], res3[4], res3[5], res3[6], res3[7]);

    int* res4 = (int*)&result4;
    printf("%d %d %d %d %d %d %d %d\n",
    res4[0], res4[1], res4[2], res4[3], res4[4], res4[5], res4[6], res4[7]);

    int* res5 = (int*)&result5;
    printf("%d %d %d %d %d %d %d %d\n",
    res5[0], res5[1], res5[2], res5[3], res5[4], res5[5], res5[6], res5[7]);

    int* res6 = (int*)&result6;
    printf("%d %d %d %d %d %d %d %d\n",
    res6[0], res6[1], res6[2], res6[3], res6[4], res6[5], res6[6], res6[7]);

    int* res7 = (int*)&result7;
    printf("%d %d %d %d %d %d %d %d\n",
    res7[0], res7[1], res7[2], res7[3], res7[4], res7[5], res7[6], res7[7]);

    int* res8 = (int*)&result8;
    printf("%d %d %d %d %d %d %d %d\n",
    res8[0], res8[1], res8[2], res8[3], res8[4], res8[5], res8[6], res8[7]);

    int* res9 = (int*)&result9;
    printf("%d %d %d %d %d %d %d %d\n",
    res9[0], res9[1], res9[2], res9[3], res9[4], res9[5], res9[6], res9[7]);

    int* res10 = (int*)&result10;
    printf("%d %d %d %d %d %d %d %d\n",
    res10[0], res10[1], res10[2], res10[3], res10[4], res10[5], res10[6], res10[7]);

    int* res11 = (int*)&result11;
    printf("%d %d %d %d %d %d %d %d\n",
    res11[0], res11[1], res11[2], res11[3], res11[4], res11[5], res11[6], res11[7]);

    int* res12 = (int*)&result12;
    printf("%d %d %d %d %d %d %d %d\n",
    res12[0], res12[1], res12[2], res12[3], res12[4], res12[5], res12[6], res12[7]);

    int* res13 = (int*)&result13;
    printf("%d %d %d %d %d %d %d %d\n",
    res13[0], res13[1], res13[2], res13[3], res13[4], res13[5], res13[6], res13[7]);

    int* res14 = (int*)&result14;
    printf("%d %d %d %d %d %d %d %d\n",
    res14[0], res14[1], res14[2], res14[3], res14[4], res14[5], res14[6], res14[7]);

    int* res15 = (int*)&result15;
    printf("%d %d %d %d %d %d %d %d\n",
    res15[0], res15[1], res15[2], res15[3], res15[4], res15[5], res15[6], res15[7]);

    int* res16 = (int*)&result16;
    printf("%d %d %d %d %d %d %d %d\n",
    res16[0], res16[1], res16[2], res16[3], res16[4], res16[5], res16[6], res16[7]);

     int* res17 = (int*)&result17;
    printf("%d %d %d %d %d %d %d %d\n",
    res17[0], res17[1], res17[2], res17[3], res17[4], res17[5], res17[6], res17[7]);

    int* res18 = (int*)&result18;
    printf("%d %d %d %d %d %d %d %d\n",
    res18[0], res18[1], res18[2], res18[3], res18[4], res18[5], res18[6], res18[7]);

    int* res19 = (int*)&result19;
    printf("%d %d %d %d %d %d %d %d\n",
    res19[0], res19[1], res19[2], res19[3], res19[4], res19[5], res19[6], res19[7]);

    int* res20 = (int*)&result20;
    printf("%d %d %d %d %d %d %d %d\n",
    res20[0], res20[1], res20[2], res20[3], res20[4], res20[5], res20[6], res20[7]);

    int* res21 = (int*)&result21;
    printf("%d %d %d %d %d %d %d %d\n",
    res21[0], res21[1], res21[2], res21[3], res21[4], res21[5], res21[6], res21[7]);

     int* res22 = (int*)&result22;
    printf("%d %d %d %d %d %d %d %d\n",
    res22[0], res22[1], res22[2], res22[3], res22[4], res22[5], res22[6], res22[7]);

    int* res23 = (int*)&result23;
    printf("%d %d %d %d %d %d %d %d\n",
    res23[0], res23[1], res23[2], res23[3], res23[4], res23[5], res23[6], res23[7]);

    int* res24 = (int*)&result24;
    printf("%d %d %d %d %d %d %d %d\n",
    res24[0], res24[1], res24[2], res24[3], res24[4], res24[5], res24[6], res24[7]);

    int* res25 = (int*)&result25;
    printf("%d %d %d %d %d %d %d %d\n",
    res25[0], res25[1], res25[2], res25[3], res25[4], res25[5], res25[6], res25[7]);

    int* res26 = (int*)&result26;
    printf("%d %d %d %d %d %d %d %d\n",
    res26[0], res26[1], res26[2], res26[3], res26[4], res26[5], res26[6], res26[7]);

    int* res27 = (int*)&result27;
    printf("%d %d %d %d %d %d %d %d\n",
    res27[0], res27[1], res27[2], res27[3], res27[4], res27[5], res27[6], res27[7]);

    int* res28 = (int*)&result28;
    printf("%d %d %d %d %d %d %d %d\n",
    res28[0], res28[1], res28[2], res28[3], res28[4], res28[5], res28[6], res28[7]);






//      auto end_time = chrono::high_resolution_clock::now();
        cout <<"The time in microseconds is"<< chrono::duration_cast<chrono::microseconds>(end_time - start_time).count() << ":"<<endl;
        double j= chrono::duration_cast<chrono::microseconds>(end_time - start_time).count();
        double k= (j/1000000);
        double output = 7700 * (1/(10*k));
        cout<<"Total number of Gflops"<<output<<endl;

}
        return 0;


}



                                                                                             
                                                                                     
