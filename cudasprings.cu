#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
//#include "potentials.h"
#include <time.h>
#include <stdint.h>
#ifdef _WIN32
#include <conio.h>
//for now only for 1 chain(!)
//CUDA
#endif
#include <curand.h>

#include <cuda_runtime_api.h>
#define NMax 16384
float SIZE[3]={100,100,100};// SIMULATION CELL SIZE
const float OBR_SIZE[3]={0.01,0.01,0.01};
 
 
 float *cudaSize;
 float *cudaObrSize;
 
curandGenerator_t gen;
   

double coords[3*NMax];
double *cudaCoords;
float testCoords[3*NMax];
double *cudaRands;
double hostRands[5*NMax];
double Energies[NMax];
double *cudaEnergies;
double *cudaPowCoeffs;
double hostPowCoeffs[NMax];
double hostdE[2];
double *cudadE;
int *cudaAccRate;
int *cudaPart;
int hostPart[2];
int AccRate[2];
int *cudaHistPart;
int hostHistPart[NMax];
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define NOT_POSSIBLE -10000;
#define NUMSTEPS NMax
#define EXCLVOL
#define TPERBLOCK 1024

void ReadPosition();
void outputGradVrml(char* filename);


 __global__ void calcEnergiesBond(double *coor, double *DEnergies, double* cudaRands, double* coeffsPow, double* dE, int rr, int* accrate, int* Particle, int* Hist)
 {
	 int x =  blockIdx.x*blockDim.x+threadIdx.x;
	 __shared__ int part;
	 __shared__ double partshift[3];
	 __shared__ double partcoor[3];
	 __shared__ double sdata[TPERBLOCK];
	 unsigned int tid = threadIdx.x;
	double mySum = 0;
	 //__shared__ int randsCounter = 0;
	 //__shared__ float PowCoeffs[NMAX];
	 //PowCoeffs[x] = coeffsPow[x];
	 
	 //int part;
	 //float partshift[3];
	 
	double prob;
	 //__syncthreads();
	 
		
		
		if(threadIdx.x == 0)
		{
		 
		 part = floor(cudaRands[5*rr]*NMax);
		 
		 partshift[0] = 0.3 - 0.6*(double)cudaRands[5*rr+1];
		 partshift[1] = 0.3 - 0.6*(double)cudaRands[5*rr+2];
		 partshift[2] = 0.3 - 0.6*(double)cudaRands[5*rr+3];
		 partcoor[0] = coor[3*part+0];
		 partcoor[1] = coor[3*part+1];
		 partcoor[2] = coor[3*part+2];
		 //randsCounter++;
		}
		if(x == 0)
		{
			dE[0] = 0;
			prob = cudaRands[5*rr+4];
			Particle[0] = part;
			Hist[part]++;
		}
	 	__syncthreads();
		
		double distNew = 0;
		double distOld = 0;
		if(x!=part)
		{	
		#pragma unroll
		for(int k = 0; k < 3; k++)
		{
		
		distNew+= ((coor[3*x+k]-partcoor[k]-partshift[k])*(coor[3*x+k]-partcoor[k] - partshift[k]));
	    distOld+= ((coor[3*x+k]-coor[3*part+k])*(coor[3*x+k]-coor[3*part+k]));
		}
		}
		sdata[tid] = 0;
		//__syncthreads();
		//DEnergies[x] = 0;
		
		#ifdef EXCLVOL
		if(distNew < 1 && x!=part)
		{
		//	DEnergies[x] += NOT_POSSIBLE;
			sdata[tid] += 10*(1 - sqrt(distNew))*(1-sqrt(distNew)) - 10*(1-sqrt(distOld))*(1-sqrt(distOld));
		}
		#endif
		//DEnergies[x] += coeffsPow[abs(x-part)]*(distNew-distOld);
		sdata[tid] += coeffsPow[abs(x-part)]*(distNew-distOld);
		DEnergies[x] = sdata[tid];
		__syncthreads();	
			

		//unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
		
		//sdata[tid] = DEnergies[i]

		//__syncthreads();
		/*
		for (unsigned int s = 1; s < blockDim.x; s *= 2) {
			if (tid % (2 * s) == 0) {
            sdata[tid] += sdata[tid + s];
			}

			__syncthreads();
		}
		*/
		//REDUCE
	if (tid < 512)
        sdata[tid] += sdata[tid + 512];
    __syncthreads();
	
	if (tid < 256)
        sdata[tid] += sdata[tid + 256];
    __syncthreads();

    if (tid < 128)
        sdata[tid] +=  sdata[tid + 128];
     __syncthreads();

    if (tid <  64)
       sdata[tid] += sdata[tid +  64];
    __syncthreads();
	
	if ( tid < 32 )
    {
        // Fetch final intermediate sum from 2nd warp
        mySum = sdata[tid]+ sdata[tid + 32];
        // Reduce final warp using shuffle
        for (int offset = warpSize/2; offset > 0; offset /= 2) 
            mySum += __shfl_down(mySum, offset);
    }
    if(tid == 0)
	{
	sdata[0]=mySum;
	}
	__syncthreads();
	//endreduce
		if(tid == 0)
		{
			atomicAdd(&dE[0],sdata[tid]);
		}
		__threadfence_system();
		if (x == 0) 
		{
			atomicAdd(&dE[0],(double)0.0);
			__threadfence_system();
		//	if(dE[0] > -1000)
		//	{
				if(exp(-dE[0]) > prob)
				{
					coor[3*part+0] = coor[3*part+0] + partshift[0]; 
					coor[3*part+1] = coor[3*part+1] + partshift[1]; 
					coor[3*part+2] = coor[3*part+2] + partshift[2]; 
					accrate[0]++;
				}
			
		//	}
		//dE[0] = 0;
		}
	
	  __syncthreads();
	 
	 
	 
 }   
 


int main()
{
	 
	setvbuf(stdout,NULL,_IONBF,0);
  gpuErrchk( cudaMalloc((void**)&cudaCoords, NMax*3*sizeof(double)));
   gpuErrchk( cudaMalloc((void**)&cudaSize, 3*sizeof(float)));
   gpuErrchk( cudaMalloc((void**)&cudaRands, NUMSTEPS*5*sizeof(double)));
   gpuErrchk( cudaMalloc((void**)&cudaPowCoeffs, NMax*sizeof(double)));
   gpuErrchk( cudaMalloc((void**)&cudaEnergies, NMax*sizeof(double)));
   gpuErrchk( cudaMalloc((void**)&cudaHistPart, NMax*sizeof(int)));

      gpuErrchk( cudaMalloc((void**)&cudadE, 2*sizeof(double)));
      gpuErrchk( cudaMalloc((void**)&cudaAccRate, 2*sizeof(int)));
      gpuErrchk( cudaMalloc((void**)&cudaPart, 2*sizeof(int)));
   
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed ( gen, time(NULL) ) ;
//   gpuErrchk( cudaMalloc((void**)&DEnerg, NMax*sizeof(float)));
//      ReadPosition();
	for(int j = 0; j < 3*NMax; j++)
	{
		coords[j] = ((double)rand()/(double)RAND_MAX)*40.0;
	}
	 dim3 gridSize = dim3(NMax/1024, 1, 1);
	 dim3 blockSize = dim3(1024, 1, 1);
	for(int j = 1; j < NMax; j++)
	{
		hostPowCoeffs[j] = (20.0/pow( (double)min(j,NMax-j),2.05));
		printf("%e\n",hostPowCoeffs[j]);
		hostHistPart[j] = 0;
		//hostPowCoeffs[j] = 0.01;
	}
	hostPowCoeffs[0] = 0;
	AccRate[0] = 0; 
	AccRate[1] = 0;
	gpuErrchk( cudaMemcpy(cudaCoords,coords, NMax*3*sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(cudaPowCoeffs,hostPowCoeffs, NMax*sizeof(double), cudaMemcpyHostToDevice));
    gpuErrchk( cudaMemcpy(cudaAccRate,AccRate, 2*sizeof(int), cudaMemcpyHostToDevice));
	int oldAccRate = 0;
	int seconds = clock()/CLOCKS_PER_SEC;
	for(int i = 0; i < 1500000; i++)
	 {		 
      curandStatus_t err = curandGenerateUniformDouble(gen, cudaRands, NUMSTEPS*5);
	//cudaMemcpy(hostRands,cudaRands,NMax*5*sizeof(double),cudaMemcpyDeviceToHost);
	//for(int z = 0; z < NMax; z++)
	//{
	//printf("%lf\n",hostRands[z]);
	//}
	//exit(0);
	cudaDeviceSynchronize();
		//	  			gpuErrchk( cudaMemcpy(testRands,cudaRands, NMax*3*sizeof(float), cudaMemcpyDeviceToHost));
	if(i%10000 == 0)
	{
		printf("%i steps %i elapsed\n",i,clock()/CLOCKS_PER_SEC-seconds);
		cudaMemcpy(AccRate,cudaAccRate,2*sizeof(int),cudaMemcpyDeviceToHost);
		cudaMemcpy(hostPart,cudaPart,2*sizeof(int),cudaMemcpyDeviceToHost);
		
		printf("%i accsteps %i part\n ",AccRate[0], hostPart[0]);
		char ff[50];
		sprintf(ff,"conf%i.vrml",i);
		outputGradVrml(ff);
		cudaMemcpy(hostdE,cudadE,2*sizeof(double),cudaMemcpyDeviceToHost);
	cudaMemcpy(Energies, cudaEnergies, NMax*sizeof(double), cudaMemcpyDeviceToHost);
	    cudaDeviceSynchronize();
	    double tempE = 0;
	    
	// printf("%f hostssum energies %f dE\n",tempE,hostdE[0]);
	cudaMemcpy(hostHistPart,cudaHistPart,NMax*sizeof(int),cudaMemcpyDeviceToHost);
		FILE *fhist = fopen("hist.txt","w");
		for(int tt = 0; tt < NMax; tt++)
	    {
		  fprintf(fhist,"%i %i\n",tt,hostHistPart[tt]);
		tempE+=Energies[tt];
	    }
		fclose(fhist);
		printf("%e dEcu %e dEhost\n",hostdE[0],tempE);
		
	}
	for(int gg = 0; gg < NUMSTEPS; gg++)
	  {
	 
	 calcEnergiesBond <<<gridSize,blockSize>>> (cudaCoords,cudaEnergies,cudaRands,cudaPowCoeffs,cudadE,gg,cudaAccRate,cudaPart,cudaHistPart);
	 gpuErrchk(cudaPeekAtLastError());
	// cudaDeviceSynchronize();
 	// cudaMemcpy(AccRate,cudaAccRate,2*sizeof(int),cudaMemcpyDeviceToHost);
	 /*if(gg==0)
	 {			
	// oldAccRate = AccRate[0];
	 cudaMemcpy(Energies,cudaEnergies,NMax*sizeof(float),cudaMemcpyDeviceToHost);
 	 cudaMemcpy(hostdE,cudadE,2*sizeof(float),cudaMemcpyDeviceToHost);
	 cudaDeviceSynchronize();
	 float tempE = 0;
	 for(int tt = 0; tt < NMax; tt++)
	 {
		 tempE += Energies[tt];
	 }
	 printf("%f hostssum energies %f dE\n",tempE,hostdE[0]);
	 }
	 //exit(0);
	//cudaDeviceSynchronize();
	  }*/
	  //}
     }
	 cudaMemcpy(coords,cudaCoords,NMax*3*sizeof(double),cudaMemcpyDeviceToHost);
	 //printf("%f %f %f first part\n",coords[0],coords[1],coords[2]);
	 }
	 cudaMemcpy(AccRate,cudaAccRate,2*sizeof(int),cudaMemcpyDeviceToHost);
	 printf("%i accsteps\n ",AccRate[0]);
	 char ff[] = "testout.vrml";
	 outputGradVrml(ff);
	 
	return 0;
}


 
struct rgb{
    double r;       // percent
    double g;       // percent
    double b;       // percent
} ;
 struct hsv{
    double h;       // angle in degrees
    double s;       // percent
    double v;       // percent
} ;

hsv   rgb2hsv(rgb in);
rgb   hsv2rgb(hsv in);

hsv rgb2hsv(rgb in)
{
    hsv         out;
    double      min, max, delta;

    min = in.r < in.g ? in.r : in.g;
    min = min  < in.b ? min  : in.b;

    max = in.r > in.g ? in.r : in.g;
    max = max  > in.b ? max  : in.b;

    out.v = max;                                // v
    delta = max - min;
    if (delta < 0.00001)
    {
        out.s = 0;
        out.h = 0; // undefined, maybe nan?
        return out;
    }
    if( max > 0.0 ) { // NOTE: if Max is == 0, this divide would cause a crash
        out.s = (delta / max);                  // s
    } else {
        // if max is 0, then r = g = b = 0              
            // s = 0, v is undefined
        out.s = 0.0;
        out.h = 0;                            // its now undefined
        return out;
    }
    if( in.r >= max )                           // > is bogus, just keeps compilor happy
        out.h = ( in.g - in.b ) / delta;        // between yellow & magenta
    else
    if( in.g >= max )
        out.h = 2.0 + ( in.b - in.r ) / delta;  // between cyan & yellow
    else
        out.h = 4.0 + ( in.r - in.g ) / delta;  // between magenta & cyan

    out.h *= 60.0;                              // degrees

    if( out.h < 0.0 )
        out.h += 360.0;

    return out;
}


rgb hsv2rgb(hsv in)
{
    double      hh, p, q, t, ff;
    long        i;
    rgb         out;

    if(in.s <= 0.0) {       // < is bogus, just shuts up warnings
        out.r = in.v;
        out.g = in.v;
        out.b = in.v;
        return out;
    }
    hh = in.h;
    if(hh >= 360.0) hh = 0.0;
    hh /= 60.0;
    i = (long)hh;
    ff = hh - i;
    p = in.v * (1.0 - in.s);
    q = in.v * (1.0 - (in.s * ff));
    t = in.v * (1.0 - (in.s * (1.0 - ff)));

    switch(i) {
    case 0:
        out.r = in.v;
        out.g = t;
        out.b = p;
        break;
    case 1:
        out.r = q;
        out.g = in.v;
        out.b = p;
        break;
    case 2:
        out.r = p;
        out.g = in.v;
        out.b = t;
        break;

    case 3:
        out.r = p;
        out.g = q;
        out.b = in.v;
        break;
    case 4:
        out.r = t;
        out.g = p;
        out.b = in.v;
        break;
    case 5:
    default:
        out.r = in.v;
        out.g = p;
        out.b = q;
        break;
    }
    return out;     
}
 
 
void ReadPosition()
{
int bloat;
int a,b,c,d;
double x,y,z;
FILE *f = fopen("positionStart.txt","r");
    int temp;

        for(int j = 0; j < NMax ; j++)
        {
            fscanf(f,"%lf %lf %lf %i %i %i %i\n",&x,&y,&z,&a,&b,&c,&d);
           		 coords[3*j+0] = x;
			coords[3*j+1] = y;
			coords[3*j+2] = z;
			//  printf("%i %i %f %f %f %i\n",j,C[i].M[j].Typ,C[i].M[j].XReal.x[0],C[i].M[j].XReal.x[1],C[i].M[j].XReal.x[2],C[i].M[j].NumBonds);
            //getch();


        }

    fclose(f);
}


void outputGradVrml(char* filename)
{
    #ifndef _WIN32
char str[1000] = "cp vrml_header.txt ";
strcat(str,filename);
system(str);
#endif

#ifdef _WIN32
char str[1000] = "copy vrml_header.txt ";
strcat(str,filename);
system(str);
#endif

//system("y");
FILE *F = fopen(filename,"a");
//printf("%lld",mAcc);
//getch();

for(int j = 0; j < NMax; j ++)
{




	hsv color;
 color.h = ( (double) j/ (double) NMax )*360.0;
 color.s = 0.9;
 color.v = 0.8;
 rgb clrrgb = hsv2rgb(color);
 
 
 fprintf(F,"Transform { \n translation %lf %lf %lf  children [ Shape {geometry Sphere {radius 0.5} \n appearance Appearance {material Material {diffuseColor %lf %lf %lf}}}]}\n",coords[3*j+0], coords[3*j+1], coords[3*j+2], clrrgb.r , clrrgb.g, clrrgb.b );

	
	




}



fprintf(F,"Shape {appearance Appearance {material Material {emissiveColor 0 0 0}}geometry IndexedLineSet {coord Coordinate {point [");
fprintf(F,"0 0 %f\n",SIZE[2]);
fprintf(F,"0 %f %f \n",SIZE[1],SIZE[2]);
fprintf(F,"0 %f 0\n",SIZE[1]);
fprintf(F,"0 0 0\n");
fprintf(F,"%f 0 0\n",SIZE[0]);
fprintf(F,"%f 0 %f\n",SIZE[0],SIZE[2]);
fprintf(F,"%f %f %f\n",SIZE[0],SIZE[1],SIZE[2]);
fprintf(F,"%f %f 0\n",SIZE[0],SIZE[1]);
fprintf(F,"]}coordIndex [");


fprintf(F,"0,1,-1\n");
fprintf(F,"1,2,-1\n");
fprintf(F,"2,3,-1\n");
fprintf(F,"0,3,-1\n");

fprintf(F,"3,4,-1\n");
fprintf(F,"1,6,-1\n");
fprintf(F,"2,7,-1\n");
fprintf(F,"0,5,-1\n");

fprintf(F,"4,5,-1\n");
fprintf(F,"5,6,-1\n");
fprintf(F,"6,7,-1\n");
fprintf(F,"7,4,-1\n");



fprintf(F,"]}}");



fclose(F);





}
