#include <stdio.h>
#include <time.h>
#include <immintrin.h>
#include <sys/time.h>
#include <omp.h>
#include <chrono>
#include <iostream>
using namespace std;

int main(int argc, char** argv)
 {

        if (argc!= 5){
                cout<<"Not enough parameters";
                return 0;
        }


        uint32_t m= atoi(argv[1]);
        cout<<"m is "<<m<<endl<<endl;

        uint32_t n = atoi(argv[2]);
        cout<<"n is "<< n << endl<<endl;

        uint32_t k =atoi(argv[3]);
        cout<<"K is"<<k<<endl<<endl;

        uint32_t block =atoi(argv[4]);
         cout<<"Block_Size is"<<block<<endl<<endl;

        //int image[];

        //int rc=  posix_memalign((void**)&image, 32, m*sizeof(int));
        int *image[m];
        int *kernel[k];
        int *output[m];
        int i,j,c,d,mm,nn,ii,jj;
        int kernelcenterX= k/2;
        int kernelcenterY= k/2;
        float operation_time=0;
        float mem_time=0;
        int temp_rows=0;
        int temp_columns=0;
		 for (i=0; i<m; i++)
        {
                 image[i] = (int *)malloc(n * sizeof(int));
        }

        for (i=0; i<m; i++)
        {
                 output[i] = (int *)malloc(n * sizeof(int));
        }


        for (i=0; i<k; i++)
        {
                 kernel[i] = (int *)malloc(k * sizeof(int));
        }


        for(i=0;i<m;i++)
        {
                for(j=0;j<n;j++)
                {
                        image[i][j]=1;

                }
        }


         for(i=0;i<m;i++)
        {
                for(j=0;j<n;j++)
                {
                        output[i][j]=1;

                }
        }



        for(i=0;i<k;i++)
        {
                for(j=0;j<k;j++)
                {
                        kernel[i][j]=1;
                }
        }

		 int counter=0;
        omp_set_dynamic(0);     // Explicitly disable dynamic teams
        omp_set_num_threads(6); // Use 4 threads for all consecutive parallel region

        auto start_time_1 = chrono::high_resolution_clock::now();


for(i=0;i<100;i++)
{

        for(int blockx=block;blockx<m; blockx=blockx+block)        // Using blocks of images so that they would be cached
        {
                for(int blocky=block;blocky<n; blocky=blocky+block)
                {
                        for(int rows=temp_rows;rows<blockx;rows++)
                        {
                                for(int columns=temp_columns;columns<blocky;columns++)
                                {

                                        for(int krows=0; krows<k; krows++)
                                        {
                                                int mm = k - 1 - krows;

                                                for(int kcolumns=0;kcolumns<k;kcolumns++)
                                                {
                                                        int nn = k - 1 - kcolumns;  // column index of flipped kernel
                                                        int ii = rows + (krows - kernelcenterY);
                                                        int jj = columns + (kcolumns - kernelcenterX);
                                                        if( ii >= 0 && ii < m && jj >= 0 && jj < n )
                                                        {
                                                                output[rows][columns] += image[ii][jj] * kernel[mm][nn]; //Fma
                                                                cout<<"Values of ii is"<<ii<<endl;
                                                                cout<<"Values of jj is"<<jj<<endl;
                                                                cout<<"Values of mm is"<<mm<<endl;
                                                                cout<<"Values of nn is"<<nn<<endl;
                                                                cout<<"Values of image[ii][jj] is"<<image[ii][jj]<<endl;
                                                                cout<<"Values of  kernel[mm][nn]"<< kernel[mm][nn]<<endl;
                                                                cout<<"Values of rows is"<<rows<<endl;
                                                                cout<<"Values of columns is"<<columns<<endl;
                                                                cout<<"Values of  output[rows][columns] is"<< output[rows][columns]<<endl<<endl;
                                                                counter++;
														                               counter++;
                                                        }
                                                }

                                        }
                                }

                        }
                        temp_columns=blocky;
                }
                temp_rows=blockx;
        }
}




        auto end_time_1 = chrono::high_resolution_clock::now();
        cout <<"The time in microseconds "<< chrono::duration_cast<chrono::microseconds>(end_time_1 - start_time_1).count()<<endl ;

        operation_time= (819/(k*k));
        mem_time= ((34*m*n)/ ((2*m*n)+ (k*k)));
        float performance= ((m*n)/(chrono::duration_cast<chrono::microseconds>(end_time_1 - start_time_1).count()*0.1 ));

        cout<<"Performance for floating point operations "<<operation_time<<" Gigapixels/sec"<<endl;
        cout<<"Performance for Memory Transfer "<<mem_time<<" Gigapixels/sec"<<endl;
        cout<<"Actual performance is "<<performance<<" Gigapixels/sec"<<endl;


        for(i=0;i<m;i++)
        {
                for(j=0;j<n;j++)
                {
                       cout<<"The output elements are"<< output[i][j]<<endl;

                }
        }



        cout<<"Counter values is"<<counter;


		
		
		
		
 }		
		
		
		
		
		
		
		