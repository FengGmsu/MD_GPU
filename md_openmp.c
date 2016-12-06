/*
	This code is using Molecular dynamics to study the evolution of the electron bunch form the photo cathod in Ultrafast electron diffraction experiment.

	Goals:
	1. [done] make this md c code working
	2. [done] implement OpenMP with this code 
	3. check with the fortran version
	4. implement CUDA to compare the result with OpenMP
	5. try to get PPPM working with GPU

	origian Fortran code version: MD_1121.f90

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

//simulation setup
#define N 1000				//number of particles
#define Ntime 1000			//number of iternations
#define newR 46022.8	//initial radius of bunch
#define cutoff 20			//lower limit of the initial distance between electrons

//parameters
#define m 5.4858e-4		//elelctron mass
#define vc 5.85e3			//speed of light
#define PI 3.141592653589793	//  \pi

double inline getrand(){ return (double)rand()/(double)RAND_MAX; }

double getPE(double r[N][3]){
	int i,j,k;
	double rel,vij;
	double PE_c = 0.0;
	for(j=0; j<(N-1); j++) {
		for(i=j+1; i<N; i++) {
			vij = 0.0;
			for(k=0; k<3; k++) {
				rel = r[i][k] - r[j][k];
				vij += pow(rel,2.0); //vij = r^2 for now
			}
			PE_c += pow(vij,-0.5); // r^(-1)
		}
	}
	return PE_c;
}

double getKE(double v[N][3]){
	int i,j;
	double KE_c = 0.0; //sum(v**2)
	for (i=0; i<N; i++){
		for (j=0; j<3; j++) {
			KE_c += pow(v[i][j],2.0);
		}
	}
	KE_c = 0.5*m*KE_c;
	return KE_c;
}

void getForce(double f[N][3],double r[N][3]) {
	int i,j,k;
	double rel[3], rel_c, fij[3];
	//use openmp here with (rel[3],rel_c,fij) private
	#pragma omp parallel for private(i,j,k,rel,rel_c,fij)
	for (j=0; j<(N-1); j++) {
		for (i=j+1; i<N; i++) {
			rel_c = 0.0;
			for (k=0; k<3; k++) {
				rel[k] = r[i][k] - r[j][k];
				rel_c += pow(rel[k], 2.0 );
			}
			rel_c = pow(rel_c, -1.5 );
			for (k=0; k<3; k++) {
				fij[k] = rel[k]*rel_c;
				//atomic operation here
				#pragma omp atomic
				f[j][k] -= fij[k];
				#pragma omp atomic
				f[i][k] += fij[k];
			}
		}
	}
}

void verlet(double r[N][3],double v[N][3],double f[N][3], double dt){
	int i,j;
	double hdtm = 0.5*dt/m; 
	for (i=0; i<N; i++) {
		for (j=0; j<3; j++) {
			v[i][j] += f[i][j]*hdtm;
			r[i][j] += v[i][j]*dt;
			f[i][j] = 0.0; //setup for force calculation
		}
	}
	getForce(f,r);
	for (i=0; i<N; i++) {
		for (j=0; j<3; j++) {
			v[i][j] += f[i][j]*hdtm;
		}
	}
}

int main() {
	double R[N][3] = {{0.0}}, V[N][3] = {{1.0}}, F[N][3]={{0.0}};
	int i, iter;
	int numb, check;
	double dt = 1.0, realt = 0.0;
	int plotstride = 20;
	double r0,r1,r2,rel0,rel1,rel2;
	double KE,PE;
	FILE *RVo,*To,*initR;

	RVo = fopen("./RandV.dat","w+");
	To = fopen("./time.dat","w+");
	initR = fopen("./initR.xyz","w+");

	srand(time(NULL));
	numb = 0;
	while (numb < N) {
		r0 = newR*(2*getrand()-1);
		r1 = newR*(2*getrand()-1);
		r2 = newR*(2*getrand()-1);
		check = 0;
		if (sqrt(r0*r0+r1*r1+r2*r2) < newR) {
			for (i = 0; i < numb; i++){
				rel0 = R[i][0] - r0;
				rel1 = R[i][1] - r1;
				rel2 = R[i][2] - r2;
				if (sqrt(rel0*rel0+rel1*rel1+rel2*rel2) < cutoff) {
					check = 1;
					break;
				}				
			}
			if (check == 0) {
				R[numb][0] = r0;
				R[numb][1] = r1;
				R[numb][2] = r2;
				numb += 1;
			}
		}
	}
	
	
	//Tonight 
	// number of box after mesh
	int numbox = 0;
	// mesh size
	int mesh = 0;
	//store every atom belongs to which box
	int boxid[N] = 0;
	//store which box contains which atom
	int box[numbox][N/2] = 0;
	//calculate every atom belongs to which box, mesh build at the same time
	for (i=0;i<=N;i++){
		boxid[i] = floor(R[i][0]/mesh) + floor(numbox*R[i][1]/mesh) + floor(numbox*numbox*R[i][2]/mesh);
	}
	// calculate which box contains which atom, build cell list
	for (i=0;i<=numbox;i++)
	{
		k = 0;
		for(j=0;j<=N;j++)
	        {
                   if(boxid[j] ==i)
		   {
			box[i][k] = j;
		        k++;
		    }
	        }
	}
	//force calculation within cutoff cell,calculate the force in nearby 26 cells
	run getforce function
	//force calculation outside cutoff cell, particle-mesh
	//first we assign charges to vertex
	for (i=0;i<=meshpoint;i++)
	{
		detx = R[i][0] - ;
		dety = R[i][1] - ;
		dety = R[i][2] - ;
		ro[i] += ChargeAssign(detx,dety,detz,h);
		
	double ChargeAssign(double x,double y, double z, h)
		{
			return W = sin(1/2*x*h)/(1/2*x*h) * sin(1/2*y*h)/(1/2*y*h) * sin(1/2*z*h)/(1/2*z*h);
		}
	double ElectricField(rp)
	        {
		
	}
	
	//output initR for check
//	fprintf(initR,"%d \n", N);
//	fprintf(initR,"%d \n", 0);
//	for (i = 0; i<N; i++) {
//		fprintf(initR,"%5d %11.3f \t %11.3f \t %11.3f \n",1,R[i][0],R[i][1],R[i][2]);
//	}
	for (iter = 0; iter< Ntime; iter++) {
		verlet(R,V,F,dt);
		realt += dt;

		//output
		if ((iter % plotstride) == 1) {		
			PE = getPE(R);
			KE = getKE(V);
			printf("%7d \t %11.5f \t %11.5f \t %11.5f \n", (int)realt, PE, KE, PE+KE);
		}
		if (iter == 200) dt = 5.0;
		if (iter == 500) dt = 15.0;
		if (iter == 700) dt = 50.0;
		if (iter == 2000) plotstride = 100;
	}

	fclose(RVo);
	fclose(To);
	fclose(initR);

	return 0;
}
