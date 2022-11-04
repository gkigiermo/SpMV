//=====================================================================================================================
// CUDA playground 
//---------------------------------------------------------------------------------------------------------------------
//
// Purpose: compute SpMV with ELLPACK on CUDA GPU b=Ax 
//
//=====================================================================================================================
#include<cuda.h>
#include<cuda_runtime.h>

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
#define N 1000

__global__ void ell(const int num_rows,const int num_cols_per_row,const int* indices, const double* data,  double* x,double* y)
{
    int offset=blockDim.x*blockIdx.x*num_cols_per_row;
    int row=blockDim.x*blockIdx.x+threadIdx.x;
    if (row<num_rows){
 
        double dot=0;
        for(int n=0; n< num_cols_per_row; n++)
        {
            int col=indices[threadIdx.x + offset+ n*blockDim.x];
            double val=data[threadIdx.x + offset+ n*blockDim.x];
            if(col!=-1)
                dot+=val*x[col];
        }
       y[row]=dot;
    }
}

void gpuELL(double *x, double *target,double *nvals,int *cols, int num_rows,int num_cprow, int blocks,int th_per_block)
{

         cudaFuncSetCacheConfig( ell, cudaFuncCachePreferL1 );

        dim3 dimGrid(blocks,1,1);
        dim3 dimBlock(th_per_block,1,1);// (1024,1024,64)


        ell<<<dimGrid,dimBlock>>>(num_rows,num_cprow,cols,nvals,x,target);
}



int main(int argc, char **argv){

    if(argc-1 != 1) {crash("a.out <Mesh_file>  ");return 0;}
    char*  matname=argv[1];

   

    cout<<"Reading matrix ..."<<endl;
    // Reading matrix info
    int nnz,csize,rsize,chunksize,blocks,threads;
    char filename[MAXCHAR];
    sprintf(filename,"%s-0.ell",matname);
    
    FILE *fp;
    fp= fopen(filename,"r");
    int sizes[6];
    fread(&sizes[0],sizeof(int),6,fp);
 
    nnz=sizes[0];
    csize=sizes[1];
    rsize=sizes[2];
    chunksize=sizes[3];
    blocks=sizes[4];
    threads=sizes[5];

    double *vals    = new double[nnz];
    int *cols    = new int[nnz];
    double *x    = new double[csize];
    double *b    = new double[csize];
    double *hb    = new double[csize];
    
    for(int i=0; i<csize; i++)
    {
        x[i]=cos(i)+cos(10*i)+cos(100*i)+cos(1000*i);
        b[i]=0.0;
        hb[i]=0.0;
    }
  
    fread(vals,sizeof(double),nnz,fp);
    fread(cols,sizeof(int),nnz,fp);

    fclose(fp);





    double end,begin,cpuTime;
    float gpuTime;

    begin=getTime();
    for(int i=0;i<N;i++)
      cpuELL(rsize,chunksize,threads, blocks, vals,cols, x, hb );
    end=getTime();
    
    cpuTime=(end-begin)/N;



    /* Define the id of the GPU device*/
    cudaSetDevice(0);

    /* References for the memory adresses in GPU memory  */
    double *dvals,*dx,*db;
    int *dcols;


    /* Creating events for calculating the execution time in GPU */
    cudaEvent_t start_t;
    cudaEvent_t stop_t;
    cudaEventCreate(&start_t);
    cudaEventCreate(&stop_t);


    /* Allocating memory in the GPU */
    cudaMalloc((void**)&dvals,nnz*sizeof(double));
    cudaMalloc((void**)&dcols,nnz*sizeof(int));
    cudaMalloc((void**)&dx,csize*sizeof(double));
    cudaMalloc((void**)&db,csize*sizeof(double));
 
    
    /* Transfering memory from Host to Device */
    cudaMemcpy(dvals,vals,nnz*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dcols,cols,nnz*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(dx,x,csize*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(db,b,csize*sizeof(double),cudaMemcpyHostToDevice);
	
        


    /* Sets a flag in the device to obtain the starting time using default stream 0 */
    cudaEventRecord(start_t,0);
    begin=getTime();
    for(int i=0;i<N;i++)
        gpuELL(dx,db,dvals,dcols,rsize,chunksize,blocks,threads);

    /* Sets a flag in the device to obtain the ending time */
      cudaEventRecord(stop_t,0);
    
      /* The CPU waits until the execution in the GPU has ended using default stream 0 */
         cudaEventSynchronize(stop_t);
        end=getTime();

    cudaEventElapsedTime(&gpuTime,start_t,stop_t);

    gpuTime=gpuTime/(N*1000);
    /* Transfers the result back in the CPU */
    cudaMemcpy(b,db,rsize*sizeof(double),cudaMemcpyDeviceToHost);



    /* checking difference */
    double err=0.0;
    for(int i=0; i<rsize; i++) err += fabs(hb[i]-b[i]);

    printf("TEST DONE. ERROR %g \n", err);
    cout<<" Time elapsed CPU "<<cpuTime<<endl;
    cout<<" Time elapsed GPU "<<gpuTime<<" new test "<<(end-begin)/N<<endl;
    cout<<" Final Speedup    "<<cpuTime/gpuTime<<endl;
    
    delete[] vals; delete[] cols; delete[] x; delete[] b,delete[] hb;
       
    cudaFree(dvals); cudaFree(dcols); cudaFree(dx); cudaFree(db);

    return 0;

}
