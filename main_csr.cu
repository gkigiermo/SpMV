//=====================================================================================================================
// CUDA playground 
//---------------------------------------------------------------------------------------------------------------------
//
// Purpose: compute SpMV with ELLPACK on CUDA GPU b=Ax 
//
//=====================================================================================================================
#include <cuda.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include <fstream>
#include <iostream>
#include <sys/time.h>
#include <sys/times.h>

#include "extras.h"
using namespace std;

#define MAXCHAR 100
#define N 10


__global__ void cudaDcsrspmv(int num_rows, int *rowIndA, int* colIndA,double* valA,double* x, double* y)
{
    int row =blockDim.x*blockIdx.x +threadIdx.x;
    if( row<num_rows)
    {
        double sum =0;
        int row_start= rowIndA[row];
        int row_end= rowIndA[row+1];
        for(int j=row_start;j<row_end;j++)
            sum+=valA[j]*x[colIndA[j]];
        y[row]=sum;
    }
}

int main(int argc, char **argv){

    if(argc-1 != 1) {crash("a.out <Mesh_file>  ");return 0;}
    char*  matname=argv[1];

   

    cout<<"Reading matrix ..."<<endl;
    // Reading matrix info
    int nnz,num_cols,num_rows,blocks,threads;
    char filename[MAXCHAR];
    sprintf(filename,"%s.csr",matname);
 
    FILE *fp;

    fp= fopen(filename,"r");

    int sizes[5];

    for(int i=0;i<5;i++)
        fscanf(fp," %d",&sizes[i]);
    fscanf(fp," \n");

    nnz=sizes[0];
    num_cols=sizes[1];
    num_rows=sizes[2];

    double* csrValA=new double[nnz];
    int* csrColIndA=new int[nnz];
    int* csrRowIndA= new int[num_rows+1];
    double *x    = new double[num_cols];
    double *b    = new double[num_rows];
    double *hb    = new double[num_rows];
   

    for(int i=0;i<nnz;i++)
        fscanf(fp," %lf",&csrValA[i]);
    fscanf(fp," \n");
    for(int i=0;i<nnz;i++)
        fscanf(fp," %d",&csrColIndA[i]);
    fscanf(fp," \n");
    for(int i=0;i<num_rows+1;i++)
        fscanf(fp," %d",&csrRowIndA[i]);
    fscanf(fp," \n");

    fclose(fp);




    double end,begin,cpuTime;
    float gpuTime;

    begin=getTime();
    for(int i=0;i<N;i++)
      cpuCSROMP(num_rows, csrValA,csrColIndA,csrRowIndA, x, hb );
    end=getTime();
    
    cpuTime=(end-begin)/N;



    /* Define the id of the GPU device*/
    cudaSetDevice(0);

    /* References for the memory adresses in GPU memory  */
    double *dcsrValA,*dx,*db;
    int *dcsrColIndA,*dcsrRowIndA;


    /* Creating events for calculating the execution time in GPU */
    cudaEvent_t start_t;
    cudaEvent_t stop_t;
    cudaEventCreate(&start_t);
    cudaEventCreate(&stop_t);


    /* Allocating memory in the GPU */
    cudaMalloc((void**)&dcsrValA,nnz*sizeof(double));
    cudaMalloc((void**)&dcsrColIndA,nnz*sizeof(int));
    cudaMalloc((void**)&dcsrRowIndA,(num_rows+1)*sizeof(int));
    cudaMalloc((void**)&dx,num_cols*sizeof(double));
    cudaMalloc((void**)&db,num_rows*sizeof(double));
 
    
    /* Transfering memory from Host to Device */
    cudaMemcpy(dcsrValA,csrValA,nnz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dcsrColIndA,csrColIndA,nnz*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dcsrRowIndA,csrRowIndA,(num_rows+1)*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dx,x,num_cols*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(db,b,num_rows*sizeof(double),cudaMemcpyHostToDevice);

    /* Selecting the number of threads per each block*/    
    threads=512;
    /* In the last block probably some threads will not perform calculations ==> negligible effect*/
    blocks=(num_rows+(threads-1))/threads;
        
    /*create the execution grid which is input of the CUDA kernel*/
    dim3 dimGrid(blocks,1,1);
    dim3 dimBlock(threads,1,1);

    /* Set maximum cache posible for improve the reuse of the multiplying vector */    
    cudaFuncSetCacheConfig( cudaDcsrspmv, cudaFuncCachePreferL1 );

    /* Sets a flag in the device to obtain the starting time using default stream 0 */
    cudaEventRecord(start_t,0);
    begin=getTime();
    for(int i=0;i<N;i++)
            cudaDcsrspmv<<<dimGrid,dimBlock>>>(num_rows,dcsrRowIndA,dcsrColIndA,dcsrValA,dx,db);


    /* Sets a flag in the device to obtain the ending time */
      cudaEventRecord(stop_t,0);
    
      /* The CPU waits until the execution in the GPU has ended using default stream 0 */
         cudaEventSynchronize(stop_t);
        end=getTime();

    cudaEventElapsedTime(&gpuTime,start_t,stop_t);

    gpuTime=gpuTime/(N*1000);
    /* Transfers the result back in the CPU */
    cudaMemcpy(b,db,num_rows*sizeof(double),cudaMemcpyDeviceToHost);



    /* checking difference */
    double err=0.0;
    for(int i=0; i<num_rows; i++) err += fabs(hb[i]-b[i]);

    printf("Error %g \n", err);
    cout<<" Time elapsed CPU "<<cpuTime<<endl;
    cout<<" Time elapsed GPU "<<gpuTime<<" new test "<<(end-begin)/N<<endl;
    cout<<" Final Speedup    "<<cpuTime/gpuTime<<endl;
    
    delete[] csrValA; delete[] csrColIndA; delete[] csrRowIndA; delete[] x; delete[] b,delete[] hb;
       
    cudaFree(dcsrValA); cudaFree(dcsrColIndA); cudaFree(dcsrRowIndA), cudaFree(dx); cudaFree(db);

    return 0;

}
