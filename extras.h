
void crash(const char *fmt,...){ //program abort
    FILE *f=NULL;
    va_list ap;
    int i;
 
    for(i=0; i<2; i++){ /* treiem el missatge per la sortida d'errors i per la sortida estandar */
        switch(i){
        case 0: f=stdout; break;
        case 1: f=stderr; break;
        //case 2: f=afile;  break;
        }
        fprintf(f,"crash: ");
        va_start(ap,fmt);
        vfprintf(f,fmt,ap);
        va_end(ap);
        fprintf(f,"\n");
        fflush(f);
    }
    exit(0);
}



double getTime()
{

        struct timeval now;

            if (gettimeofday(&now, (struct timezone *) 0)) exit(1);

                return(((double) now.tv_sec + (double) now.tv_usec / 1000000) +1.0);
}

void cpuELL(int rows,int chunksize,int threads,int blocks, double *vals,int *cols, double *x,double *b )
{

    int goffset=0;
    int threadId=0;
    int gth=0;
    for(int i=0;i<blocks;i++)
    {
        for(int j=0;j<threads;j++)
        {
            threadId= j + i*threads;
            if(threadId < rows)
            {
                double sum=0;

                for(int k=0;k<chunksize;k++)
                {
                    gth=j+k*threads+goffset;
                    int col=cols[gth];
                    if(col!=-1)
                        sum+=x[col]*vals[gth];

                }
                b[threadId]=sum;
            }
        }
        goffset+=threads*chunksize;
    }
}

//Ax=b
void cpuCSROMP(int num_rows, double *csrValA,int *csrColIndA,int *csrRowIndA, double *x,double *b )
{
    double sum;
    int i,j,jb,je;

//    omp_set_num_threads(4);

// #pragma omp parallel for schedule(static) private(j,jb,je,sum)
    for (i=0;i<num_rows;i++) {
        sum = 0;
        jb = csrRowIndA[i];
        je = csrRowIndA[i+1];
        for (j=jb;j<je;j++)
              sum+=x[csrColIndA[j]]*csrValA[j];
       b[i] = sum;
    }
}


