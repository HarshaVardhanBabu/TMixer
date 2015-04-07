/*****************************************************************************************************************
This program predicts the fluid(Incompressible and compressible) behaviour in 3d TMixer using D3Q19 model
******************************************************************************************************************/
/*----------------------------------------------------------------------

This code Uses Pressure Boundary at the outlet and Velocity Boundary at the inlets
			Y ^
			|<---------------------length------------------->^
ny	6(G,G')	|------------------------------------------------| (F,F')7
	Inlet1  |		cw/2									 | Inlet2
	5(H,H')	|--------------------		---------------------| (E,E') 4
			|			3(D,D') ^		^ (C,C') 2			 |
			|					|		|					 |
			|					|		|					 |
			|					|		|				   width
			|					|		|					 |
			|				   mny		|					 |
			|					|		|					 |
			|					|		|					 |
			|					|		|					 |
			|					|		|					 |
			|					|<-mnx->|					 |
			|					V       V					 V
	(0,0) Z ----------------------------------------------------->  X
							(A,A')	(B,B')					(nx,0)
								0	   1
								Outlet
--------------------------------------------------------
1. Z axis is out of the plane.
2. plane (ABCDEFGH)' is z = nz plane.
-------------------------------------------------------*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp.h"

double *** allocate(int x,int y,int z)
{
	double ***temp;
	temp = (double ***)calloc(x,sizeof(double**));
    int i,j;
    #pragma omp parallel for collapse(1)
    for (i = 0; i< x; i++) 
        {         
	        temp[i] = (double **) calloc(y,sizeof(double *));
	    }
        #pragma omp parallel for collapse(2)
        for (i = 0; i< x; i++) 
        {         
            for (j = 0; j < y; j++) 
              {              
                temp[i][j] = (double *) calloc(z,sizeof(double));
              }
        }
	return temp;
}
void deallocate(double ***temp,int x,int y)
{
    int i,j;
    #pragma omp parallel for collapse(2)
    for(i=0;i<x;i++)
    {
    	for(j=0;j<y;j++)
    	{
    		free(temp[i][j]);
    	}
    }
    #pragma omp parallel for collapse(1)
    for(i=0;i<x;i++)
    {
        free(temp[i]);
    }
    free(temp);
}
// iterations to calculate the double summation in velocity profile generation
#define LIMIT 100

// Value of pi
#define pi 3.14159265358979323846   
#define NSc 3.0  // Schmidt Number

#define np 18	   //19-speed square lattice

#define tau 0.6  //Relaxation parameter momentum

#define taua (0.5*(((2.0*tau-1.0)/NSc)+1.0))  //Relaxation parameter A
#define taub (0.5*(((2.0*tau-1.0)/NSc)+1.0))  //Relaxation parameter B
#define taup (0.5*(((2.0*tau-1.0)/NSc)+1.0))  //Relaxation parameter C

#define Ca_in1 1.0
#define Ca_in2 0.0

#define Cb_in1 0.0
#define Cb_in2 1.0

#define Cp_in1 0.0
#define Cp_in2 0.0

#define rateconst 1.0

#define rho_out 3.0 //Outlet density 

#define rho0 3.0  //Initial density

#define C0 0.0 // Initial concentration

#define Umean 0.059761

 //Mean velocity at the inlet1
//0.09523810//0.0452//0.1282//0.00555556//0.0277778//(0.111111/4.0)
//0.01//0.16666667
///0.007167//0.047619//0.066667//0.00083333//0.005976096
//0.059761//0.139442//0.006640106//0.139442231 
//0.119521912//0.015//0.111553785 //0.15 

#define uxo 0
#define uyo 0
#define uzo 0

// Depth of the channel

#define nz 51
//61//11//101//141//71//53//45//19//25//91//81//61//45 //nz=no of z-grids
//35//41//7//21//25 //21  //11
/*-----------------------------------------------------------------------*/

// Inlet Channel Dimensions

/*-----------------------------------------------------------------------*/
#define inx 401
//481//101//301//421//211//161//1291//161//517//689//271//241//181//1291//861 //nx=no of x-grids (odd)
//(in this code number starts from 1 to nx)
//281//121//173  //161//201//401   //61
#define iny 51
//11//51//71//35//27//45//27//19//25//45//41//31//45//31 //ny=no of y-grids (odd)
//561//401//37 //321//401//201  //61
/*-----------------------------------------------------------------------*/
				// Mixing Channel Dimensions
/*-----------------------------------------------------------------------*/
#define mnx 101
//121//21//101//141//71//53//91//53//37//49//91//81//61//91 //nx=no of x-grids (odd)
//281//121//173  //161//201//401   //61
#define mny (800-50)
//(960-60)//(1400-70)//(271-45)//(533-27)//(108-18)//(144-24)//(361-45)//720//570//(226)//ny=no of y-grids (even)
//561//401//37 //321//401//201  //61 
/*-----------------------------------------------------------------------*/
#define time_step 1000
#define time_domaindata 5000
#define time 100000

#define length (inx)	  //length of the Horizontal Mixing Channel as shown in schematic
#define width  (mny+iny) // Width  as shown in schematic
#define depth  (nz)     //  Depth of the Mixer

#define esp 1.0 //Speed of the sound (in m/s)

// The following parameters are used in the velocity generation
/*
-----------------------------------
Cross-Sectional View of the Inlet
X-axis is into the plane.
-----------------------------------
		(Z)
		^
		|--------------|
		|    		   |
		|			   |
2.0*(d)	|			   |
		|			   |
		|			   |
		(X)----------------> (Y)
		2.0*(w)

*/
// Half of the depth of the inlet channel Cross section as shown in the above schematic
#define d ((nz-1.0)/2.0)
// Half of the Width of the inlet channel Cross section as shown in the above schematic
#define w ((iny-1.0)/2.0)
//Aspect Ratio of the Inlet cross-section
#define AR (1.0*w/d)

//Physical Parameters used for calculating conversion factors//

#define LENGTH (0.0008)//(0.0004)//(0.008600)(0.000800) // length in meters
#define DENSITY (1000.0) // Density in kg/m3
#define PRESSURE (101325.0) // Pressure in Kg/ms2
#define KVISCOSITY (0.000001004) // Kinematic viscosity in m2/s
// Constants for Converting the LBM units to Real/physical units //
#define L0 (LENGTH/(length-1))
#define T0 ((((2.0*tau-1.0)/6.0)/KVISCOSITY)*(L0*L0))
#define M0 ((PRESSURE/(rho0/3.0))*(L0*T0*T0))
#define M01 ((DENSITY/(rho0))*(L0*L0*L0))
#define V0 (L0/T0)
#define P0 ((M0)/(L0*T0*T0))
#define P01 ((M01)/(L0*T0*T0))
#define Pw0 ((L0*L0)/(T0*T0*T0))
#define Kv0 ((L0*L0)/(T0))
/*----------------------------------------------------------------------------
Declaring global variables
------------------------------------------------------------------------------*/

double ***f0 = NULL; //Local distribution function in the horizontal Inlet channel
double ***f1 = NULL;
double ***f2 = NULL;
double ***f3 = NULL;
double ***f4 = NULL;
double ***f5 = NULL;
double ***f6 = NULL;
double ***f7 = NULL;
double ***f8 = NULL;
double ***f9 = NULL;
double ***f10 = NULL;
double ***f11 = NULL;
double ***f12 = NULL;
double ***f13 = NULL;
double ***f14 = NULL;
double ***f15 = NULL;
double ***f16 = NULL;
double ***f17 = NULL;
double ***f18 = NULL;

double ***g0 = NULL; //Local distribution function in the horizontal Inlet channel
double ***g1 = NULL;
double ***g2 = NULL;
double ***g3 = NULL;
double ***g4 = NULL;
double ***g5 = NULL;
double ***g6 = NULL;
double ***g7 = NULL;
double ***g8 = NULL;
double ***g9 = NULL;
double ***g10 = NULL;
double ***g11 = NULL;
double ***g12 = NULL;
double ***g13 = NULL;
double ***g14 = NULL;
double ***g15 = NULL;
double ***g16 = NULL;
double ***g17 = NULL;
double ***g18 = NULL;

double ***h0 = NULL; //Local distribution function in the horizontal Inlet channel
double ***h1 = NULL;
double ***h2 = NULL;
double ***h3 = NULL;
double ***h4 = NULL;
double ***h5 = NULL;
double ***h6 = NULL;
double ***h7 = NULL;
double ***h8 = NULL;
double ***h9 = NULL;
double ***h10 = NULL;
double ***h11 = NULL;
double ***h12 = NULL;
double ***h13 = NULL;
double ***h14 = NULL;
double ***h15 = NULL;
double ***h16 = NULL;
double ***h17 = NULL;
double ***h18 = NULL;

double ***p0 = NULL; //Local distribution function in tpe porizontal Inlet cpannel
double ***p1 = NULL;
double ***p2 = NULL;
double ***p3 = NULL;
double ***p4 = NULL;
double ***p5 = NULL;
double ***p6 = NULL;
double ***p7 = NULL;
double ***p8 = NULL;
double ***p9 = NULL;
double ***p10 = NULL;
double ***p11 = NULL;
double ***p12 = NULL;
double ***p13 = NULL;
double ***p14 = NULL;
double ***p15 = NULL;
double ***p16 = NULL;
double ***p17 = NULL;
double ***p18 = NULL;

double ***fm0 = NULL; //Local distribution fmunction in the horizontal Inlet channel
double ***fm1 = NULL;
double ***fm2 = NULL;
double ***fm3 = NULL;
double ***fm4 = NULL;
double ***fm5 = NULL;
double ***fm6 = NULL;
double ***fm7 = NULL;
double ***fm8 = NULL;
double ***fm9 = NULL;
double ***fm10 = NULL;
double ***fm11 = NULL;
double ***fm12 = NULL;
double ***fm13 = NULL;
double ***fm14 = NULL;
double ***fm15 = NULL;
double ***fm16 = NULL;
double ***fm17 = NULL;
double ***fm18 = NULL;

double ***gm0 = NULL; //Local distribution gmunction in the horizontal Inlet channel
double ***gm1 = NULL;
double ***gm2 = NULL;
double ***gm3 = NULL;
double ***gm4 = NULL;
double ***gm5 = NULL;
double ***gm6 = NULL;
double ***gm7 = NULL;
double ***gm8 = NULL;
double ***gm9 = NULL;
double ***gm10 = NULL;
double ***gm11 = NULL;
double ***gm12 = NULL;
double ***gm13 = NULL;
double ***gm14 = NULL;
double ***gm15 = NULL;
double ***gm16 = NULL;
double ***gm17 = NULL;
double ***gm18 = NULL;

double ***hm0 = NULL; //Local distribution hmunction in the horizontal Inlet channel
double ***hm1 = NULL;
double ***hm2 = NULL;
double ***hm3 = NULL;
double ***hm4 = NULL;
double ***hm5 = NULL;
double ***hm6 = NULL;
double ***hm7 = NULL;
double ***hm8 = NULL;
double ***hm9 = NULL;
double ***hm10 = NULL;
double ***hm11 = NULL;
double ***hm12 = NULL;
double ***hm13 = NULL;
double ***hm14 = NULL;
double ***hm15 = NULL;
double ***hm16 = NULL;
double ***hm17 = NULL;
double ***hm18 = NULL;

double ***pm0 = NULL; //Local distribution pmunction in the horizontal Inlet channel
double ***pm1 = NULL;
double ***pm2 = NULL;
double ***pm3 = NULL;
double ***pm4 = NULL;
double ***pm5 = NULL;
double ***pm6 = NULL;
double ***pm7 = NULL;
double ***pm8 = NULL;
double ***pm9 = NULL;
double ***pm10 = NULL;
double ***pm11 = NULL;
double ***pm12 = NULL;
double ***pm13 = NULL;
double ***pm14 = NULL;
double ***pm15 = NULL;
double ***pm16 = NULL;
double ***pm17 = NULL;
double ***pm18 = NULL;

double ***feq0 = NULL; //Equilibrium distribution function for Horizontal Inlets chaNULLl
double ***feq1 = NULL;
double ***feq2 = NULL;
double ***feq3 = NULL;
double ***feq4 = NULL;
double ***feq5 = NULL;
double ***feq6 = NULL;
double ***feq7 = NULL;
double ***feq8 = NULL;
double ***feq9 = NULL;
double ***feq10 = NULL;
double ***feq11 = NULL;
double ***feq12 = NULL;
double ***feq13 = NULL;
double ***feq14 = NULL;
double ***feq15 = NULL;
double ***feq16 = NULL;
double ***feq17 = NULL;
double ***feq18 = NULL;

double ***geq0 = NULL; //Equilibrium distribution gunction gor Horizontal Inlets chaNULLl
double ***geq1 = NULL;
double ***geq2 = NULL;
double ***geq3 = NULL;
double ***geq4 = NULL;
double ***geq5 = NULL;
double ***geq6 = NULL;
double ***geq7 = NULL;
double ***geq8 = NULL;
double ***geq9 = NULL;
double ***geq10 = NULL;
double ***geq11 = NULL;
double ***geq12 = NULL;
double ***geq13 = NULL;
double ***geq14 = NULL;
double ***geq15 = NULL;
double ***geq16 = NULL;
double ***geq17 = NULL;
double ***geq18 = NULL;

double ***heq0 = NULL; //Equilibrium distribution hunction hor Horizontal Inlets chaNULLl
double ***heq1 = NULL;
double ***heq2 = NULL;
double ***heq3 = NULL;
double ***heq4 = NULL;
double ***heq5 = NULL;
double ***heq6 = NULL;
double ***heq7 = NULL;
double ***heq8 = NULL;
double ***heq9 = NULL;
double ***heq10 = NULL;
double ***heq11 = NULL;
double ***heq12 = NULL;
double ***heq13 = NULL;
double ***heq14 = NULL;
double ***heq15 = NULL;
double ***heq16 = NULL;
double ***heq17 = NULL;
double ***heq18 = NULL;

double ***peq0 = NULL; //Equilibrium distribution gunction gor Horizontal Inlets chaNULLl
double ***peq1 = NULL;
double ***peq2 = NULL;
double ***peq3 = NULL;
double ***peq4 = NULL;
double ***peq5 = NULL;
double ***peq6 = NULL;
double ***peq7 = NULL;
double ***peq8 = NULL;
double ***peq9 = NULL;
double ***peq10 = NULL;
double ***peq11 = NULL;
double ***peq12 = NULL;
double ***peq13 = NULL;
double ***peq14 = NULL;
double ***peq15 = NULL;
double ***peq16 = NULL;
double ***peq17 = NULL;
double ***peq18 = NULL;

double ***fmeq0 = NULL; //Equilibrium distribution function for Mixing channel
double ***fmeq1 = NULL;
double ***fmeq2 = NULL;
double ***fmeq3 = NULL;
double ***fmeq4 = NULL;
double ***fmeq5 = NULL;
double ***fmeq6 = NULL;
double ***fmeq7 = NULL;
double ***fmeq8 = NULL;
double ***fmeq9 = NULL;
double ***fmeq10 =NULL;
double ***fmeq11 =NULL;
double ***fmeq12 =NULL;
double ***fmeq13 =NULL;
double ***fmeq14 =NULL;
double ***fmeq15 =NULL;
double ***fmeq16 =NULL;
double ***fmeq17 =NULL;
double ***fmeq18 =NULL;

double ***gmeq0 = NULL; //Equilibrium distribution gunction gor Mixing channel
double ***gmeq1 = NULL;
double ***gmeq2 = NULL;
double ***gmeq3 = NULL;
double ***gmeq4 = NULL;
double ***gmeq5 = NULL;
double ***gmeq6 = NULL;
double ***gmeq7 = NULL;
double ***gmeq8 = NULL;
double ***gmeq9 = NULL;
double ***gmeq10 =NULL;
double ***gmeq11 =NULL;
double ***gmeq12 =NULL;
double ***gmeq13 =NULL;
double ***gmeq14 =NULL;
double ***gmeq15 =NULL;
double ***gmeq16 =NULL;
double ***gmeq17 =NULL;
double ***gmeq18 =NULL;

double ***hmeq0 = NULL; //Equilibrium distribution gunction gor Mixing channel
double ***hmeq1 = NULL;
double ***hmeq2 = NULL;
double ***hmeq3 = NULL;
double ***hmeq4 = NULL;
double ***hmeq5 = NULL;
double ***hmeq6 = NULL;
double ***hmeq7 = NULL;
double ***hmeq8 = NULL;
double ***hmeq9 = NULL;
double ***hmeq10 =NULL;
double ***hmeq11 =NULL;
double ***hmeq12 =NULL;
double ***hmeq13 =NULL;
double ***hmeq14 =NULL;
double ***hmeq15 =NULL;
double ***hmeq16 =NULL;
double ***hmeq17 =NULL;
double ***hmeq18 =NULL;

double ***pmeq0 = NULL; //Equilibrium distribution gunction gor Mixing channel
double ***pmeq1 = NULL;
double ***pmeq2 = NULL;
double ***pmeq3 = NULL;
double ***pmeq4 = NULL;
double ***pmeq5 = NULL;
double ***pmeq6 = NULL;
double ***pmeq7 = NULL;
double ***pmeq8 = NULL;
double ***pmeq9 = NULL;
double ***pmeq10 =NULL;
double ***pmeq11 =NULL;
double ***pmeq12 =NULL;
double ***pmeq13 =NULL;
double ***pmeq14 =NULL;
double ***pmeq15 =NULL;
double ***pmeq16 =NULL;
double ***pmeq17 =NULL;
double ***pmeq18 =NULL;

double ***fn0=NULL; // A step ahead euilibrium function for the horizontal inlets channel
double ***fn1=NULL;
double ***fn2=NULL;
double ***fn3=NULL;
double ***fn4=NULL;
double ***fn5=NULL;
double ***fn6=NULL;
double ***fn7=NULL;
double ***fn8=NULL;
double ***fn9=NULL;
double ***fn10=NULL;
double ***fn11=NULL;
double ***fn12=NULL;
double ***fn13=NULL;
double ***fn14=NULL;
double ***fn15=NULL;
double ***fn16=NULL;
double ***fn17=NULL;
double ***fn18=NULL;

double ***gn0=NULL; // A step ahead euilibrium gunction gor the horizontal inlets channel
double ***gn1=NULL;
double ***gn2=NULL;
double ***gn3=NULL;
double ***gn4=NULL;
double ***gn5=NULL;
double ***gn6=NULL;
double ***gn7=NULL;
double ***gn8=NULL;
double ***gn9=NULL;
double ***gn10=NULL;
double ***gn11=NULL;
double ***gn12=NULL;
double ***gn13=NULL;
double ***gn14=NULL;
double ***gn15=NULL;
double ***gn16=NULL;
double ***gn17=NULL;
double ***gn18=NULL;

double ***hn0=NULL; // A step ahead euilibrium gunction gor the horizontal inlets channel
double ***hn1=NULL;
double ***hn2=NULL;
double ***hn3=NULL;
double ***hn4=NULL;
double ***hn5=NULL;
double ***hn6=NULL;
double ***hn7=NULL;
double ***hn8=NULL;
double ***hn9=NULL;
double ***hn10=NULL;
double ***hn11=NULL;
double ***hn12=NULL;
double ***hn13=NULL;
double ***hn14=NULL;
double ***hn15=NULL;
double ***hn16=NULL;
double ***hn17=NULL;
double ***hn18=NULL;

double ***pn0=NULL; // A step ahead euilibrium gunction gor the horizontal inlets channel
double ***pn1=NULL;
double ***pn2=NULL;
double ***pn3=NULL;
double ***pn4=NULL;
double ***pn5=NULL;
double ***pn6=NULL;
double ***pn7=NULL;
double ***pn8=NULL;
double ***pn9=NULL;
double ***pn10=NULL;
double ***pn11=NULL;
double ***pn12=NULL;
double ***pn13=NULL;
double ***pn14=NULL;
double ***pn15=NULL;
double ***pn16=NULL;
double ***pn17=NULL;
double ***pn18=NULL;

double ***fmn0 = NULL; // A step ahead euilibrium function for the Mixing channel
double ***fmn1 = NULL;
double ***fmn2 = NULL;
double ***fmn3 = NULL;
double ***fmn4 = NULL;
double ***fmn5 = NULL;
double ***fmn6 = NULL;
double ***fmn7 = NULL;
double ***fmn8 = NULL;
double ***fmn9 = NULL;
double ***fmn10 = NULL;
double ***fmn11 = NULL;
double ***fmn12 = NULL;
double ***fmn13 = NULL;
double ***fmn14 = NULL;
double ***fmn15 = NULL;
double ***fmn16 = NULL;
double ***fmn17 = NULL;
double ***fmn18 = NULL;

double ***gmn0 = NULL; // A step ahead euilibrium gunction gor the Mixing channel
double ***gmn1 = NULL;
double ***gmn2 = NULL;
double ***gmn3 = NULL;
double ***gmn4 = NULL;
double ***gmn5 = NULL;
double ***gmn6 = NULL;
double ***gmn7 = NULL;
double ***gmn8 = NULL;
double ***gmn9 = NULL;
double ***gmn10 = NULL;
double ***gmn11 = NULL;
double ***gmn12 = NULL;
double ***gmn13 = NULL;
double ***gmn14 = NULL;
double ***gmn15 = NULL;
double ***gmn16 = NULL;
double ***gmn17 = NULL;
double ***gmn18 = NULL;

double ***hmn0 = NULL; // A step ahead euilibrium gunction gor the Mixing channel
double ***hmn1 = NULL;
double ***hmn2 = NULL;
double ***hmn3 = NULL;
double ***hmn4 = NULL;
double ***hmn5 = NULL;
double ***hmn6 = NULL;
double ***hmn7 = NULL;
double ***hmn8 = NULL;
double ***hmn9 = NULL;
double ***hmn10 = NULL;
double ***hmn11 = NULL;
double ***hmn12 = NULL;
double ***hmn13 = NULL;
double ***hmn14 = NULL;
double ***hmn15 = NULL;
double ***hmn16 = NULL;
double ***hmn17 = NULL;
double ***hmn18 = NULL;

double ***pmn0 = NULL; // A step ahead euilibrium gunction gor the Mixing channel
double ***pmn1 = NULL;
double ***pmn2 = NULL;
double ***pmn3 = NULL;
double ***pmn4 = NULL;
double ***pmn5 = NULL;
double ***pmn6 = NULL;
double ***pmn7 = NULL;
double ***pmn8 = NULL;
double ***pmn9 = NULL;
double ***pmn10 = NULL;
double ***pmn11 = NULL;
double ***pmn12 = NULL;
double ***pmn13 = NULL;
double ***pmn14 = NULL;
double ***pmn15 = NULL;
double ***pmn16 = NULL;
double ***pmn17 = NULL;
double ***pmn18 = NULL;

double ***rho = NULL; // Local density inlets horizontal channel
double ***mrho = NULL; // Local density mixing channel

double ***grho = NULL; // Local density inlets horizontal channel
double ***gmrho = NULL; // Local density mixing channel

double ***hrho = NULL; // Local density inlets horizontal channel
double ***hmrho = NULL; // Local density mixing channel

double ***prho = NULL; // Local density inlets horizontal channel
double ***pmrho = NULL; // Local density mixing channel

double ***ux = NULL; //Local velocity in x-direction in the inlet horizontal channel
double ***uy = NULL; //Local velocity in Y-direction in the inlet horizontal channel
double ***uz = NULL; //Local velocity in Z-direction in the inlet horizontal channel

double ***mux = NULL; //Local velocity in x-direction in the mixing channel
double ***muy = NULL; //Local velocity in Y-direction in the mixing channel
double ***muz = NULL; //Local velocity in Z-direction in the mixing channel


double Uin1[iny + 3][nz + 3] = { 0 }; // Velocity at the inlet1
double Uin2[iny + 3][nz + 3] = { 0 }; // Velocity at the inlet2

double weights[] = { 1.0/3.0,
					1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,1.0/18.0,
					1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,
					1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0,1.0/36.0};

void allocateMemory()
{	

			#pragma omp parallel sections
  			{
  				#pragma omp section
			    { 
					rho = allocate(inx+3,iny+3,nz+3);
					mrho = allocate(mnx+3,mny+3,nz+3);

					grho = allocate(inx+3,iny+3,nz+3);
					gmrho = allocate(mnx+3,mny+3,nz+3);

					hrho = allocate(inx+3,iny+3,nz+3);
					hmrho = allocate(mnx+3,mny+3,nz+3);

					prho = allocate(inx+3,iny+3,nz+3);
					pmrho = allocate(mnx+3,mny+3,nz+3);
				}
				#pragma omp section
			    {
					ux = allocate(inx+3,iny+3,nz+3);
					uy = allocate(inx+3,iny+3,nz+3);
					uz = allocate(inx+3,iny+3,nz+3);
				}
				#pragma omp section
			    {
					mux = allocate(mnx+3,mny+3,nz+3);
					muy = allocate(mnx+3,mny+3,nz+3);
					muz = allocate(mnx+3,mny+3,nz+3);
				}
				#pragma omp section
				{
					f0 = allocate(inx+3,iny+3,nz+3);		feq0 = allocate(inx+3,iny+3,nz+3);    
					f1 = allocate(inx+3,iny+3,nz+3);		feq1 = allocate(inx+3,iny+3,nz+3);
					f2 = allocate(inx+3,iny+3,nz+3);		feq2 = allocate(inx+3,iny+3,nz+3);
					f3 = allocate(inx+3,iny+3,nz+3);		feq3 = allocate(inx+3,iny+3,nz+3);
					f4 = allocate(inx+3,iny+3,nz+3);		feq4 = allocate(inx+3,iny+3,nz+3);
					f5 = allocate(inx+3,iny+3,nz+3);		feq5 = allocate(inx+3,iny+3,nz+3);
					f6 = allocate(inx+3,iny+3,nz+3);		feq6 = allocate(inx+3,iny+3,nz+3);
					f7 = allocate(inx+3,iny+3,nz+3);		feq7 = allocate(inx+3,iny+3,nz+3);
					f8 = allocate(inx+3,iny+3,nz+3);		feq8 = allocate(inx+3,iny+3,nz+3);
					f9 = allocate(inx+3,iny+3,nz+3);		feq9 = allocate(inx+3,iny+3,nz+3);
					f10 = allocate(inx+3,iny+3,nz+3);		feq10 = allocate(inx+3,iny+3,nz+3);
					f11 = allocate(inx+3,iny+3,nz+3);		feq11 = allocate(inx+3,iny+3,nz+3);
					f12 = allocate(inx+3,iny+3,nz+3);		feq12 = allocate(inx+3,iny+3,nz+3);
					f13 = allocate(inx+3,iny+3,nz+3);		feq13 = allocate(inx+3,iny+3,nz+3);
					f14 = allocate(inx+3,iny+3,nz+3);		feq14 = allocate(inx+3,iny+3,nz+3);
					f15 = allocate(inx+3,iny+3,nz+3);		feq15 = allocate(inx+3,iny+3,nz+3);
					f16 = allocate(inx+3,iny+3,nz+3);		feq16 = allocate(inx+3,iny+3,nz+3);
					f17 = allocate(inx+3,iny+3,nz+3);		feq17 = allocate(inx+3,iny+3,nz+3);
					f18 = allocate(inx+3,iny+3,nz+3);		feq18 = allocate(inx+3,iny+3,nz+3);

					g0 = allocate(inx+3,iny+3,nz+3);		geq0 = allocate(inx+3,iny+3,nz+3);    
					g1 = allocate(inx+3,iny+3,nz+3);		geq1 = allocate(inx+3,iny+3,nz+3);
					g2 = allocate(inx+3,iny+3,nz+3);		geq2 = allocate(inx+3,iny+3,nz+3);
					g3 = allocate(inx+3,iny+3,nz+3);		geq3 = allocate(inx+3,iny+3,nz+3);
					g4 = allocate(inx+3,iny+3,nz+3);		geq4 = allocate(inx+3,iny+3,nz+3);
					g5 = allocate(inx+3,iny+3,nz+3);		geq5 = allocate(inx+3,iny+3,nz+3);
					g6 = allocate(inx+3,iny+3,nz+3);		geq6 = allocate(inx+3,iny+3,nz+3);
					g7 = allocate(inx+3,iny+3,nz+3);		geq7 = allocate(inx+3,iny+3,nz+3);
					g8 = allocate(inx+3,iny+3,nz+3);		geq8 = allocate(inx+3,iny+3,nz+3);
					g9 = allocate(inx+3,iny+3,nz+3);		geq9 = allocate(inx+3,iny+3,nz+3);
					g10 = allocate(inx+3,iny+3,nz+3);		geq10 = allocate(inx+3,iny+3,nz+3);
					g11 = allocate(inx+3,iny+3,nz+3);		geq11 = allocate(inx+3,iny+3,nz+3);
					g12 = allocate(inx+3,iny+3,nz+3);		geq12 = allocate(inx+3,iny+3,nz+3);
					g13 = allocate(inx+3,iny+3,nz+3);		geq13 = allocate(inx+3,iny+3,nz+3);
					g14 = allocate(inx+3,iny+3,nz+3);		geq14 = allocate(inx+3,iny+3,nz+3);
					g15 = allocate(inx+3,iny+3,nz+3);		geq15 = allocate(inx+3,iny+3,nz+3);
					g16 = allocate(inx+3,iny+3,nz+3);		geq16 = allocate(inx+3,iny+3,nz+3);
					g17 = allocate(inx+3,iny+3,nz+3);		geq17 = allocate(inx+3,iny+3,nz+3);
					g18 = allocate(inx+3,iny+3,nz+3);		geq18 = allocate(inx+3,iny+3,nz+3);
				}
				#pragma omp section
				{
					h0 = allocate(inx+3,iny+3,nz+3);		heq0 = allocate(inx+3,iny+3,nz+3);    
					h1 = allocate(inx+3,iny+3,nz+3);		heq1 = allocate(inx+3,iny+3,nz+3);
					h2 = allocate(inx+3,iny+3,nz+3);		heq2 = allocate(inx+3,iny+3,nz+3);
					h3 = allocate(inx+3,iny+3,nz+3);		heq3 = allocate(inx+3,iny+3,nz+3);
					h4 = allocate(inx+3,iny+3,nz+3);		heq4 = allocate(inx+3,iny+3,nz+3);
					h5 = allocate(inx+3,iny+3,nz+3);		heq5 = allocate(inx+3,iny+3,nz+3);
					h6 = allocate(inx+3,iny+3,nz+3);		heq6 = allocate(inx+3,iny+3,nz+3);
					h7 = allocate(inx+3,iny+3,nz+3);		heq7 = allocate(inx+3,iny+3,nz+3);
					h8 = allocate(inx+3,iny+3,nz+3);		heq8 = allocate(inx+3,iny+3,nz+3);
					h9 = allocate(inx+3,iny+3,nz+3);		heq9 = allocate(inx+3,iny+3,nz+3);
					h10 = allocate(inx+3,iny+3,nz+3);		heq10 = allocate(inx+3,iny+3,nz+3);
					h11 = allocate(inx+3,iny+3,nz+3);		heq11 = allocate(inx+3,iny+3,nz+3);
					h12 = allocate(inx+3,iny+3,nz+3);		heq12 = allocate(inx+3,iny+3,nz+3);
					h13 = allocate(inx+3,iny+3,nz+3);		heq13 = allocate(inx+3,iny+3,nz+3);
					h14 = allocate(inx+3,iny+3,nz+3);		heq14 = allocate(inx+3,iny+3,nz+3);
					h15 = allocate(inx+3,iny+3,nz+3);		heq15 = allocate(inx+3,iny+3,nz+3);
					h16 = allocate(inx+3,iny+3,nz+3);		heq16 = allocate(inx+3,iny+3,nz+3);
					h17 = allocate(inx+3,iny+3,nz+3);		heq17 = allocate(inx+3,iny+3,nz+3);
					h18 = allocate(inx+3,iny+3,nz+3);		heq18 = allocate(inx+3,iny+3,nz+3);

					p0 = allocate(inx+3,iny+3,nz+3);		peq0 = allocate(inx+3,iny+3,nz+3);    
					p1 = allocate(inx+3,iny+3,nz+3);		peq1 = allocate(inx+3,iny+3,nz+3);
					p2 = allocate(inx+3,iny+3,nz+3);		peq2 = allocate(inx+3,iny+3,nz+3);
					p3 = allocate(inx+3,iny+3,nz+3);		peq3 = allocate(inx+3,iny+3,nz+3);
					p4 = allocate(inx+3,iny+3,nz+3);		peq4 = allocate(inx+3,iny+3,nz+3);
					p5 = allocate(inx+3,iny+3,nz+3);		peq5 = allocate(inx+3,iny+3,nz+3);
					p6 = allocate(inx+3,iny+3,nz+3);		peq6 = allocate(inx+3,iny+3,nz+3);
					p7 = allocate(inx+3,iny+3,nz+3);		peq7 = allocate(inx+3,iny+3,nz+3);
					p8 = allocate(inx+3,iny+3,nz+3);		peq8 = allocate(inx+3,iny+3,nz+3);
					p9 = allocate(inx+3,iny+3,nz+3);		peq9 = allocate(inx+3,iny+3,nz+3);
					p10 = allocate(inx+3,iny+3,nz+3);		peq10 = allocate(inx+3,iny+3,nz+3);
					p11 = allocate(inx+3,iny+3,nz+3);		peq11 = allocate(inx+3,iny+3,nz+3);
					p12 = allocate(inx+3,iny+3,nz+3);		peq12 = allocate(inx+3,iny+3,nz+3);
					p13 = allocate(inx+3,iny+3,nz+3);		peq13 = allocate(inx+3,iny+3,nz+3);
					p14 = allocate(inx+3,iny+3,nz+3);		peq14 = allocate(inx+3,iny+3,nz+3);
					p15 = allocate(inx+3,iny+3,nz+3);		peq15 = allocate(inx+3,iny+3,nz+3);
					p16 = allocate(inx+3,iny+3,nz+3);		peq16 = allocate(inx+3,iny+3,nz+3);
					p17 = allocate(inx+3,iny+3,nz+3);		peq17 = allocate(inx+3,iny+3,nz+3);
					p18 = allocate(inx+3,iny+3,nz+3);		peq18 = allocate(inx+3,iny+3,nz+3);
				}
				#pragma omp section
				{
					fn0 = allocate(inx+3,iny+3,nz+3);  
					fn1 = allocate(inx+3,iny+3,nz+3);
					fn2 = allocate(inx+3,iny+3,nz+3);
					fn3 = allocate(inx+3,iny+3,nz+3);
					fn4 = allocate(inx+3,iny+3,nz+3);
					fn5 = allocate(inx+3,iny+3,nz+3);
					fn6 = allocate(inx+3,iny+3,nz+3);
					fn7 = allocate(inx+3,iny+3,nz+3);
					fn8 = allocate(inx+3,iny+3,nz+3);
					fn9 = allocate(inx+3,iny+3,nz+3);
					fn10 = allocate(inx+3,iny+3,nz+3);
					fn11 = allocate(inx+3,iny+3,nz+3);
					fn12 = allocate(inx+3,iny+3,nz+3);
					fn13 = allocate(inx+3,iny+3,nz+3);
					fn14 = allocate(inx+3,iny+3,nz+3);
					fn15 = allocate(inx+3,iny+3,nz+3);
					fn16 = allocate(inx+3,iny+3,nz+3);
					fn17 = allocate(inx+3,iny+3,nz+3);
					fn18 = allocate(inx+3,iny+3,nz+3);

					gn0 = allocate(inx+3,iny+3,nz+3);  
					gn1 = allocate(inx+3,iny+3,nz+3);
					gn2 = allocate(inx+3,iny+3,nz+3);
					gn3 = allocate(inx+3,iny+3,nz+3);
					gn4 = allocate(inx+3,iny+3,nz+3);
					gn5 = allocate(inx+3,iny+3,nz+3);
					gn6 = allocate(inx+3,iny+3,nz+3);
					gn7 = allocate(inx+3,iny+3,nz+3);
					gn8 = allocate(inx+3,iny+3,nz+3);
					gn9 = allocate(inx+3,iny+3,nz+3);
					gn10 = allocate(inx+3,iny+3,nz+3);
					gn11 = allocate(inx+3,iny+3,nz+3);
					gn12 = allocate(inx+3,iny+3,nz+3);
					gn13 = allocate(inx+3,iny+3,nz+3);
					gn14 = allocate(inx+3,iny+3,nz+3);
					gn15 = allocate(inx+3,iny+3,nz+3);
					gn16 = allocate(inx+3,iny+3,nz+3);
					gn17 = allocate(inx+3,iny+3,nz+3);
					gn18 = allocate(inx+3,iny+3,nz+3);
				}
				#pragma omp section
				{
					hn0 = allocate(inx+3,iny+3,nz+3);  
					hn1 = allocate(inx+3,iny+3,nz+3);
					hn2 = allocate(inx+3,iny+3,nz+3);
					hn3 = allocate(inx+3,iny+3,nz+3);
					hn4 = allocate(inx+3,iny+3,nz+3);
					hn5 = allocate(inx+3,iny+3,nz+3);
					hn6 = allocate(inx+3,iny+3,nz+3);
					hn7 = allocate(inx+3,iny+3,nz+3);
					hn8 = allocate(inx+3,iny+3,nz+3);
					hn9 = allocate(inx+3,iny+3,nz+3);
					hn10 = allocate(inx+3,iny+3,nz+3);
					hn11 = allocate(inx+3,iny+3,nz+3);
					hn12 = allocate(inx+3,iny+3,nz+3);
					hn13 = allocate(inx+3,iny+3,nz+3);
					hn14 = allocate(inx+3,iny+3,nz+3);
					hn15 = allocate(inx+3,iny+3,nz+3);
					hn16 = allocate(inx+3,iny+3,nz+3);
					hn17 = allocate(inx+3,iny+3,nz+3);
					hn18 = allocate(inx+3,iny+3,nz+3);

					pn0 = allocate(inx+3,iny+3,nz+3);  
					pn1 = allocate(inx+3,iny+3,nz+3);
					pn2 = allocate(inx+3,iny+3,nz+3);
					pn3 = allocate(inx+3,iny+3,nz+3);
					pn4 = allocate(inx+3,iny+3,nz+3);
					pn5 = allocate(inx+3,iny+3,nz+3);
					pn6 = allocate(inx+3,iny+3,nz+3);
					pn7 = allocate(inx+3,iny+3,nz+3);
					pn8 = allocate(inx+3,iny+3,nz+3);
					pn9 = allocate(inx+3,iny+3,nz+3);
					pn10 = allocate(inx+3,iny+3,nz+3);
					pn11 = allocate(inx+3,iny+3,nz+3);
					pn12 = allocate(inx+3,iny+3,nz+3);
					pn13 = allocate(inx+3,iny+3,nz+3);
					pn14 = allocate(inx+3,iny+3,nz+3);
					pn15 = allocate(inx+3,iny+3,nz+3);
					pn16 = allocate(inx+3,iny+3,nz+3);
					pn17 = allocate(inx+3,iny+3,nz+3);
					pn18 = allocate(inx+3,iny+3,nz+3);
				}
				#pragma omp section
				{

					fm0 = allocate(mnx+3,mny+3,nz+3);		fmeq0 = allocate(mnx+3,mny+3,nz+3);    
					fm1 = allocate(mnx+3,mny+3,nz+3);		fmeq1 = allocate(mnx+3,mny+3,nz+3);
					fm2 = allocate(mnx+3,mny+3,nz+3);		fmeq2 = allocate(mnx+3,mny+3,nz+3);
					fm3 = allocate(mnx+3,mny+3,nz+3);		fmeq3 = allocate(mnx+3,mny+3,nz+3);
					fm4 = allocate(mnx+3,mny+3,nz+3);		fmeq4 = allocate(mnx+3,mny+3,nz+3);
					fm5 = allocate(mnx+3,mny+3,nz+3);		fmeq5 = allocate(mnx+3,mny+3,nz+3);
					fm6 = allocate(mnx+3,mny+3,nz+3);		fmeq6 = allocate(mnx+3,mny+3,nz+3);
					fm7 = allocate(mnx+3,mny+3,nz+3);		fmeq7 = allocate(mnx+3,mny+3,nz+3);
					fm8 = allocate(mnx+3,mny+3,nz+3);		fmeq8 = allocate(mnx+3,mny+3,nz+3);
					fm9 = allocate(mnx+3,mny+3,nz+3);		fmeq9 = allocate(mnx+3,mny+3,nz+3);
					fm10 = allocate(mnx+3,mny+3,nz+3);		fmeq10 = allocate(mnx+3,mny+3,nz+3);
					fm11 = allocate(mnx+3,mny+3,nz+3);		fmeq11 = allocate(mnx+3,mny+3,nz+3);
					fm12 = allocate(mnx+3,mny+3,nz+3);		fmeq12 = allocate(mnx+3,mny+3,nz+3);
					fm13 = allocate(mnx+3,mny+3,nz+3);		fmeq13 = allocate(mnx+3,mny+3,nz+3);
					fm14 = allocate(mnx+3,mny+3,nz+3);		fmeq14 = allocate(mnx+3,mny+3,nz+3);
					fm15 = allocate(mnx+3,mny+3,nz+3);		fmeq15 = allocate(mnx+3,mny+3,nz+3);
					fm16 = allocate(mnx+3,mny+3,nz+3);		fmeq16 = allocate(mnx+3,mny+3,nz+3);
					fm17 = allocate(mnx+3,mny+3,nz+3);		fmeq17 = allocate(mnx+3,mny+3,nz+3);
					fm18 = allocate(mnx+3,mny+3,nz+3);		fmeq18 = allocate(mnx+3,mny+3,nz+3);

					gm0 = allocate(mnx+3,mny+3,nz+3);		gmeq0 = allocate(mnx+3,mny+3,nz+3);    
					gm1 = allocate(mnx+3,mny+3,nz+3);		gmeq1 = allocate(mnx+3,mny+3,nz+3);
					gm2 = allocate(mnx+3,mny+3,nz+3);		gmeq2 = allocate(mnx+3,mny+3,nz+3);
					gm3 = allocate(mnx+3,mny+3,nz+3);		gmeq3 = allocate(mnx+3,mny+3,nz+3);
					gm4 = allocate(mnx+3,mny+3,nz+3);		gmeq4 = allocate(mnx+3,mny+3,nz+3);
					gm5 = allocate(mnx+3,mny+3,nz+3);		gmeq5 = allocate(mnx+3,mny+3,nz+3);
					gm6 = allocate(mnx+3,mny+3,nz+3);		gmeq6 = allocate(mnx+3,mny+3,nz+3);
					gm7 = allocate(mnx+3,mny+3,nz+3);		gmeq7 = allocate(mnx+3,mny+3,nz+3);
					gm8 = allocate(mnx+3,mny+3,nz+3);		gmeq8 = allocate(mnx+3,mny+3,nz+3);
					gm9 = allocate(mnx+3,mny+3,nz+3);		gmeq9 = allocate(mnx+3,mny+3,nz+3);
					gm10 = allocate(mnx+3,mny+3,nz+3);		gmeq10 = allocate(mnx+3,mny+3,nz+3);
					gm11 = allocate(mnx+3,mny+3,nz+3);		gmeq11 = allocate(mnx+3,mny+3,nz+3);
					gm12 = allocate(mnx+3,mny+3,nz+3);		gmeq12 = allocate(mnx+3,mny+3,nz+3);
					gm13 = allocate(mnx+3,mny+3,nz+3);		gmeq13 = allocate(mnx+3,mny+3,nz+3);
					gm14 = allocate(mnx+3,mny+3,nz+3);		gmeq14 = allocate(mnx+3,mny+3,nz+3);
					gm15 = allocate(mnx+3,mny+3,nz+3);		gmeq15 = allocate(mnx+3,mny+3,nz+3);
					gm16 = allocate(mnx+3,mny+3,nz+3);		gmeq16 = allocate(mnx+3,mny+3,nz+3);
					gm17 = allocate(mnx+3,mny+3,nz+3);		gmeq17 = allocate(mnx+3,mny+3,nz+3);
					gm18 = allocate(mnx+3,mny+3,nz+3);		gmeq18 = allocate(mnx+3,mny+3,nz+3);
				}
				#pragma omp section
				{

					hm0 = allocate(mnx+3,mny+3,nz+3);		hmeq0 = allocate(mnx+3,mny+3,nz+3);    
					hm1 = allocate(mnx+3,mny+3,nz+3);		hmeq1 = allocate(mnx+3,mny+3,nz+3);
					hm2 = allocate(mnx+3,mny+3,nz+3);		hmeq2 = allocate(mnx+3,mny+3,nz+3);
					hm3 = allocate(mnx+3,mny+3,nz+3);		hmeq3 = allocate(mnx+3,mny+3,nz+3);
					hm4 = allocate(mnx+3,mny+3,nz+3);		hmeq4 = allocate(mnx+3,mny+3,nz+3);
					hm5 = allocate(mnx+3,mny+3,nz+3);		hmeq5 = allocate(mnx+3,mny+3,nz+3);
					hm6 = allocate(mnx+3,mny+3,nz+3);		hmeq6 = allocate(mnx+3,mny+3,nz+3);
					hm7 = allocate(mnx+3,mny+3,nz+3);		hmeq7 = allocate(mnx+3,mny+3,nz+3);
					hm8 = allocate(mnx+3,mny+3,nz+3);		hmeq8 = allocate(mnx+3,mny+3,nz+3);
					hm9 = allocate(mnx+3,mny+3,nz+3);		hmeq9 = allocate(mnx+3,mny+3,nz+3);
					hm10 = allocate(mnx+3,mny+3,nz+3);		hmeq10 = allocate(mnx+3,mny+3,nz+3);
					hm11 = allocate(mnx+3,mny+3,nz+3);		hmeq11 = allocate(mnx+3,mny+3,nz+3);
					hm12 = allocate(mnx+3,mny+3,nz+3);		hmeq12 = allocate(mnx+3,mny+3,nz+3);
					hm13 = allocate(mnx+3,mny+3,nz+3);		hmeq13 = allocate(mnx+3,mny+3,nz+3);
					hm14 = allocate(mnx+3,mny+3,nz+3);		hmeq14 = allocate(mnx+3,mny+3,nz+3);
					hm15 = allocate(mnx+3,mny+3,nz+3);		hmeq15 = allocate(mnx+3,mny+3,nz+3);
					hm16 = allocate(mnx+3,mny+3,nz+3);		hmeq16 = allocate(mnx+3,mny+3,nz+3);
					hm17 = allocate(mnx+3,mny+3,nz+3);		hmeq17 = allocate(mnx+3,mny+3,nz+3);
					hm18 = allocate(mnx+3,mny+3,nz+3);		hmeq18 = allocate(mnx+3,mny+3,nz+3);

					pm0 = allocate(mnx+3,mny+3,nz+3);		pmeq0 = allocate(mnx+3,mny+3,nz+3);    
					pm1 = allocate(mnx+3,mny+3,nz+3);		pmeq1 = allocate(mnx+3,mny+3,nz+3);
					pm2 = allocate(mnx+3,mny+3,nz+3);		pmeq2 = allocate(mnx+3,mny+3,nz+3);
					pm3 = allocate(mnx+3,mny+3,nz+3);		pmeq3 = allocate(mnx+3,mny+3,nz+3);
					pm4 = allocate(mnx+3,mny+3,nz+3);		pmeq4 = allocate(mnx+3,mny+3,nz+3);
					pm5 = allocate(mnx+3,mny+3,nz+3);		pmeq5 = allocate(mnx+3,mny+3,nz+3);
					pm6 = allocate(mnx+3,mny+3,nz+3);		pmeq6 = allocate(mnx+3,mny+3,nz+3);
					pm7 = allocate(mnx+3,mny+3,nz+3);		pmeq7 = allocate(mnx+3,mny+3,nz+3);
					pm8 = allocate(mnx+3,mny+3,nz+3);		pmeq8 = allocate(mnx+3,mny+3,nz+3);
					pm9 = allocate(mnx+3,mny+3,nz+3);		pmeq9 = allocate(mnx+3,mny+3,nz+3);
					pm10 = allocate(mnx+3,mny+3,nz+3);		pmeq10 = allocate(mnx+3,mny+3,nz+3);
					pm11 = allocate(mnx+3,mny+3,nz+3);		pmeq11 = allocate(mnx+3,mny+3,nz+3);
					pm12 = allocate(mnx+3,mny+3,nz+3);		pmeq12 = allocate(mnx+3,mny+3,nz+3);
					pm13 = allocate(mnx+3,mny+3,nz+3);		pmeq13 = allocate(mnx+3,mny+3,nz+3);
					pm14 = allocate(mnx+3,mny+3,nz+3);		pmeq14 = allocate(mnx+3,mny+3,nz+3);
					pm15 = allocate(mnx+3,mny+3,nz+3);		pmeq15 = allocate(mnx+3,mny+3,nz+3);
					pm16 = allocate(mnx+3,mny+3,nz+3);		pmeq16 = allocate(mnx+3,mny+3,nz+3);
					pm17 = allocate(mnx+3,mny+3,nz+3);		pmeq17 = allocate(mnx+3,mny+3,nz+3);
					pm18 = allocate(mnx+3,mny+3,nz+3);		pmeq18 = allocate(mnx+3,mny+3,nz+3);
				}
				#pragma omp section
				{
					fmn0 = allocate(mnx+3,mny+3,nz+3);  
					fmn1 = allocate(mnx+3,mny+3,nz+3);
					fmn2 = allocate(mnx+3,mny+3,nz+3);
					fmn3 = allocate(mnx+3,mny+3,nz+3);
					fmn4 = allocate(mnx+3,mny+3,nz+3);
					fmn5 = allocate(mnx+3,mny+3,nz+3);
					fmn6 = allocate(mnx+3,mny+3,nz+3);
					fmn7 = allocate(mnx+3,mny+3,nz+3);
					fmn8 = allocate(mnx+3,mny+3,nz+3);
					fmn9 = allocate(mnx+3,mny+3,nz+3);
					fmn10 = allocate(mnx+3,mny+3,nz+3);
					fmn11 = allocate(mnx+3,mny+3,nz+3);
					fmn12 = allocate(mnx+3,mny+3,nz+3);
					fmn13 = allocate(mnx+3,mny+3,nz+3);
					fmn14 = allocate(mnx+3,mny+3,nz+3);
					fmn15 = allocate(mnx+3,mny+3,nz+3);
					fmn16 = allocate(mnx+3,mny+3,nz+3);
					fmn17 = allocate(mnx+3,mny+3,nz+3);
					fmn18 = allocate(mnx+3,mny+3,nz+3);

					gmn0 = allocate(mnx+3,mny+3,nz+3);  
					gmn1 = allocate(mnx+3,mny+3,nz+3);
					gmn2 = allocate(mnx+3,mny+3,nz+3);
					gmn3 = allocate(mnx+3,mny+3,nz+3);
					gmn4 = allocate(mnx+3,mny+3,nz+3);
					gmn5 = allocate(mnx+3,mny+3,nz+3);
					gmn6 = allocate(mnx+3,mny+3,nz+3);
					gmn7 = allocate(mnx+3,mny+3,nz+3);
					gmn8 = allocate(mnx+3,mny+3,nz+3);
					gmn9 = allocate(mnx+3,mny+3,nz+3);
					gmn10 = allocate(mnx+3,mny+3,nz+3);
					gmn11 = allocate(mnx+3,mny+3,nz+3);
					gmn12 = allocate(mnx+3,mny+3,nz+3);
					gmn13 = allocate(mnx+3,mny+3,nz+3);
					gmn14 = allocate(mnx+3,mny+3,nz+3);
					gmn15 = allocate(mnx+3,mny+3,nz+3);
					gmn16 = allocate(mnx+3,mny+3,nz+3);
					gmn17 = allocate(mnx+3,mny+3,nz+3);
					gmn18 = allocate(mnx+3,mny+3,nz+3);
				}
				#pragma omp section
				{
					hmn0 = allocate(mnx+3,mny+3,nz+3);  
					hmn1 = allocate(mnx+3,mny+3,nz+3);
					hmn2 = allocate(mnx+3,mny+3,nz+3);
					hmn3 = allocate(mnx+3,mny+3,nz+3);
					hmn4 = allocate(mnx+3,mny+3,nz+3);
					hmn5 = allocate(mnx+3,mny+3,nz+3);
					hmn6 = allocate(mnx+3,mny+3,nz+3);
					hmn7 = allocate(mnx+3,mny+3,nz+3);
					hmn8 = allocate(mnx+3,mny+3,nz+3);
					hmn9 = allocate(mnx+3,mny+3,nz+3);
					hmn10 = allocate(mnx+3,mny+3,nz+3);
					hmn11 = allocate(mnx+3,mny+3,nz+3);
					hmn12 = allocate(mnx+3,mny+3,nz+3);
					hmn13 = allocate(mnx+3,mny+3,nz+3);
					hmn14 = allocate(mnx+3,mny+3,nz+3);
					hmn15 = allocate(mnx+3,mny+3,nz+3);
					hmn16 = allocate(mnx+3,mny+3,nz+3);
					hmn17 = allocate(mnx+3,mny+3,nz+3);
					hmn18 = allocate(mnx+3,mny+3,nz+3);

					pmn0 = allocate(mnx+3,mny+3,nz+3);  
					pmn1 = allocate(mnx+3,mny+3,nz+3);
					pmn2 = allocate(mnx+3,mny+3,nz+3);
					pmn3 = allocate(mnx+3,mny+3,nz+3);
					pmn4 = allocate(mnx+3,mny+3,nz+3);
					pmn5 = allocate(mnx+3,mny+3,nz+3);
					pmn6 = allocate(mnx+3,mny+3,nz+3);
					pmn7 = allocate(mnx+3,mny+3,nz+3);
					pmn8 = allocate(mnx+3,mny+3,nz+3);
					pmn9 = allocate(mnx+3,mny+3,nz+3);
					pmn10 = allocate(mnx+3,mny+3,nz+3);
					pmn11 = allocate(mnx+3,mny+3,nz+3);
					pmn12 = allocate(mnx+3,mny+3,nz+3);
					pmn13 = allocate(mnx+3,mny+3,nz+3);
					pmn14 = allocate(mnx+3,mny+3,nz+3);
					pmn15 = allocate(mnx+3,mny+3,nz+3);
					pmn16 = allocate(mnx+3,mny+3,nz+3);
					pmn17 = allocate(mnx+3,mny+3,nz+3);
					pmn18 = allocate(mnx+3,mny+3,nz+3);
				}
			}
}

void deallocateMemory()
{

	#pragma omp parallel sections
	{
	#pragma omp section
    { 
		deallocate(rho,inx+3,iny+3);
		deallocate(mrho,mnx+3,mny+3);

		deallocate(hrho,inx+3,iny+3);
		deallocate(hmrho,mnx+3,mny+3);

		deallocate(prho,inx+3,iny+3);
		deallocate(pmrho,mnx+3,mny+3);

		deallocate(grho,inx+3,iny+3);
		deallocate(gmrho,mnx+3,mny+3);

		deallocate(ux,inx+3,iny+3);
		deallocate(uy,inx+3,iny+3);
		deallocate(uz,inx+3,iny+3);
	}
	#pragma omp section
    {

		deallocate(mux,mnx+3,mny+3);
		deallocate(muy,mnx+3,mny+3);
		deallocate(muz,mnx+3,mny+3);
	}
	#pragma omp section
    {
    	deallocate(f0,inx+3,iny+3); 	deallocate(fm0,mnx+3,mny+3);    
		deallocate(f1,inx+3,iny+3); 	deallocate(fm1,mnx+3,mny+3);
		deallocate(f2,inx+3,iny+3); 	deallocate(fm2,mnx+3,mny+3);
		deallocate(f3,inx+3,iny+3); 	deallocate(fm3,mnx+3,mny+3);
		deallocate(f4,inx+3,iny+3); 	deallocate(fm4,mnx+3,mny+3);
		deallocate(f5,inx+3,iny+3); 	deallocate(fm5,mnx+3,mny+3);
		deallocate(f6,inx+3,iny+3); 	deallocate(fm6,mnx+3,mny+3);
		deallocate(f7,inx+3,iny+3); 	deallocate(fm7,mnx+3,mny+3);
		deallocate(f8,inx+3,iny+3); 	deallocate(fm8,mnx+3,mny+3);
		deallocate(f9,inx+3,iny+3); 	deallocate(fm9,mnx+3,mny+3);
		deallocate(f10,inx+3,iny+3); 	deallocate(fm10,mnx+3,mny+3);
		deallocate(f11,inx+3,iny+3); 	deallocate(fm11,mnx+3,mny+3);
		deallocate(f12,inx+3,iny+3); 	deallocate(fm12,mnx+3,mny+3);
		deallocate(f13,inx+3,iny+3); 	deallocate(fm13,mnx+3,mny+3);
		deallocate(f14,inx+3,iny+3); 	deallocate(fm14,mnx+3,mny+3);
		deallocate(f15,inx+3,iny+3); 	deallocate(fm15,mnx+3,mny+3);
		deallocate(f16,inx+3,iny+3); 	deallocate(fm16,mnx+3,mny+3);
		deallocate(f17,inx+3,iny+3); 	deallocate(fm17,mnx+3,mny+3);
		deallocate(f18,inx+3,iny+3); 	deallocate(fm18,mnx+3,mny+3);

		deallocate(g0,inx+3,iny+3); 	deallocate(gm0,mnx+3,mny+3);    
		deallocate(g1,inx+3,iny+3); 	deallocate(gm1,mnx+3,mny+3);
		deallocate(g2,inx+3,iny+3); 	deallocate(gm2,mnx+3,mny+3);
		deallocate(g3,inx+3,iny+3); 	deallocate(gm3,mnx+3,mny+3);
		deallocate(g4,inx+3,iny+3); 	deallocate(gm4,mnx+3,mny+3);
		deallocate(g5,inx+3,iny+3); 	deallocate(gm5,mnx+3,mny+3);
		deallocate(g6,inx+3,iny+3); 	deallocate(gm6,mnx+3,mny+3);
		deallocate(g7,inx+3,iny+3); 	deallocate(gm7,mnx+3,mny+3);
		deallocate(g8,inx+3,iny+3); 	deallocate(gm8,mnx+3,mny+3);
		deallocate(g9,inx+3,iny+3); 	deallocate(gm9,mnx+3,mny+3);
		deallocate(g10,inx+3,iny+3); 	deallocate(gm10,mnx+3,mny+3);
		deallocate(g11,inx+3,iny+3); 	deallocate(gm11,mnx+3,mny+3);
		deallocate(g12,inx+3,iny+3); 	deallocate(gm12,mnx+3,mny+3);
		deallocate(g13,inx+3,iny+3); 	deallocate(gm13,mnx+3,mny+3);
		deallocate(g14,inx+3,iny+3); 	deallocate(gm14,mnx+3,mny+3);
		deallocate(g15,inx+3,iny+3); 	deallocate(gm15,mnx+3,mny+3);
		deallocate(g16,inx+3,iny+3); 	deallocate(gm16,mnx+3,mny+3);
		deallocate(g17,inx+3,iny+3); 	deallocate(gm17,mnx+3,mny+3);
		deallocate(g18,inx+3,iny+3); 	deallocate(gm18,mnx+3,mny+3);
    }
    #pragma omp section
    {
    	deallocate(h0,inx+3,iny+3); 	deallocate(hm0,mnx+3,mny+3);    
		deallocate(h1,inx+3,iny+3); 	deallocate(hm1,mnx+3,mny+3);
		deallocate(h2,inx+3,iny+3); 	deallocate(hm2,mnx+3,mny+3);
		deallocate(h3,inx+3,iny+3); 	deallocate(hm3,mnx+3,mny+3);
		deallocate(h4,inx+3,iny+3); 	deallocate(hm4,mnx+3,mny+3);
		deallocate(h5,inx+3,iny+3); 	deallocate(hm5,mnx+3,mny+3);
		deallocate(h6,inx+3,iny+3); 	deallocate(hm6,mnx+3,mny+3);
		deallocate(h7,inx+3,iny+3); 	deallocate(hm7,mnx+3,mny+3);
		deallocate(h8,inx+3,iny+3); 	deallocate(hm8,mnx+3,mny+3);
		deallocate(h9,inx+3,iny+3); 	deallocate(hm9,mnx+3,mny+3);
		deallocate(h10,inx+3,iny+3); 	deallocate(hm10,mnx+3,mny+3);
		deallocate(h11,inx+3,iny+3); 	deallocate(hm11,mnx+3,mny+3);
		deallocate(h12,inx+3,iny+3); 	deallocate(hm12,mnx+3,mny+3);
		deallocate(h13,inx+3,iny+3); 	deallocate(hm13,mnx+3,mny+3);
		deallocate(h14,inx+3,iny+3); 	deallocate(hm14,mnx+3,mny+3);
		deallocate(h15,inx+3,iny+3); 	deallocate(hm15,mnx+3,mny+3);
		deallocate(h16,inx+3,iny+3); 	deallocate(hm16,mnx+3,mny+3);
		deallocate(h17,inx+3,iny+3); 	deallocate(hm17,mnx+3,mny+3);
		deallocate(h18,inx+3,iny+3); 	deallocate(hm18,mnx+3,mny+3);

		deallocate(p0,inx+3,iny+3); 	deallocate(pm0,mnx+3,mny+3);    
		deallocate(p1,inx+3,iny+3); 	deallocate(pm1,mnx+3,mny+3);
		deallocate(p2,inx+3,iny+3); 	deallocate(pm2,mnx+3,mny+3);
		deallocate(p3,inx+3,iny+3); 	deallocate(pm3,mnx+3,mny+3);
		deallocate(p4,inx+3,iny+3); 	deallocate(pm4,mnx+3,mny+3);
		deallocate(p5,inx+3,iny+3); 	deallocate(pm5,mnx+3,mny+3);
		deallocate(p6,inx+3,iny+3); 	deallocate(pm6,mnx+3,mny+3);
		deallocate(p7,inx+3,iny+3); 	deallocate(pm7,mnx+3,mny+3);
		deallocate(p8,inx+3,iny+3); 	deallocate(pm8,mnx+3,mny+3);
		deallocate(p9,inx+3,iny+3); 	deallocate(pm9,mnx+3,mny+3);
		deallocate(p10,inx+3,iny+3); 	deallocate(pm10,mnx+3,mny+3);
		deallocate(p11,inx+3,iny+3); 	deallocate(pm11,mnx+3,mny+3);
		deallocate(p12,inx+3,iny+3); 	deallocate(pm12,mnx+3,mny+3);
		deallocate(p13,inx+3,iny+3); 	deallocate(pm13,mnx+3,mny+3);
		deallocate(p14,inx+3,iny+3); 	deallocate(pm14,mnx+3,mny+3);
		deallocate(p15,inx+3,iny+3); 	deallocate(pm15,mnx+3,mny+3);
		deallocate(p16,inx+3,iny+3); 	deallocate(pm16,mnx+3,mny+3);
		deallocate(g17,inx+3,iny+3); 	deallocate(gm17,mnx+3,mny+3);
		deallocate(g18,inx+3,iny+3); 	deallocate(gm18,mnx+3,mny+3);
    }
    #pragma omp section
    {
    	deallocate(feq0,inx+3,iny+3);	deallocate(fmeq0,mnx+3,mny+3);      
		deallocate(feq1,inx+3,iny+3);	deallocate(fmeq1,mnx+3,mny+3);
		deallocate(feq2,inx+3,iny+3);	deallocate(fmeq2,mnx+3,mny+3);
		deallocate(feq3,inx+3,iny+3);	deallocate(fmeq3,mnx+3,mny+3);
		deallocate(feq4,inx+3,iny+3);	deallocate(fmeq4,mnx+3,mny+3);
		deallocate(feq5,inx+3,iny+3);	deallocate(fmeq5,mnx+3,mny+3);
		deallocate(feq6,inx+3,iny+3);	deallocate(fmeq6,mnx+3,mny+3);
		deallocate(feq7,inx+3,iny+3);	deallocate(fmeq7,mnx+3,mny+3);
		deallocate(feq8,inx+3,iny+3);	deallocate(fmeq8,mnx+3,mny+3);
		deallocate(feq9,inx+3,iny+3);	deallocate(fmeq9,mnx+3,mny+3);
		deallocate(feq10,inx+3,iny+3);	deallocate(fmeq10,mnx+3,mny+3);
		deallocate(feq11,inx+3,iny+3);	deallocate(fmeq11,mnx+3,mny+3);
		deallocate(feq12,inx+3,iny+3);	deallocate(fmeq12,mnx+3,mny+3);
		deallocate(feq13,inx+3,iny+3);	deallocate(fmeq13,mnx+3,mny+3);
		deallocate(feq14,inx+3,iny+3);	deallocate(fmeq14,mnx+3,mny+3);
		deallocate(feq15,inx+3,iny+3);	deallocate(fmeq15,mnx+3,mny+3);
		deallocate(feq16,inx+3,iny+3);	deallocate(fmeq16,mnx+3,mny+3);
		deallocate(feq17,inx+3,iny+3);	deallocate(fmeq17,mnx+3,mny+3);
		deallocate(feq18,inx+3,iny+3);	deallocate(fmeq18,mnx+3,mny+3);

		deallocate(geq0,inx+3,iny+3);	deallocate(gmeq0,mnx+3,mny+3);      
		deallocate(geq1,inx+3,iny+3);	deallocate(gmeq1,mnx+3,mny+3);
		deallocate(geq2,inx+3,iny+3);	deallocate(gmeq2,mnx+3,mny+3);
		deallocate(geq3,inx+3,iny+3);	deallocate(gmeq3,mnx+3,mny+3);
		deallocate(geq4,inx+3,iny+3);	deallocate(gmeq4,mnx+3,mny+3);
		deallocate(geq5,inx+3,iny+3);	deallocate(gmeq5,mnx+3,mny+3);
		deallocate(geq6,inx+3,iny+3);	deallocate(gmeq6,mnx+3,mny+3);
		deallocate(geq7,inx+3,iny+3);	deallocate(gmeq7,mnx+3,mny+3);
		deallocate(geq8,inx+3,iny+3);	deallocate(gmeq8,mnx+3,mny+3);
		deallocate(geq9,inx+3,iny+3);	deallocate(gmeq9,mnx+3,mny+3);
		deallocate(geq10,inx+3,iny+3);	deallocate(gmeq10,mnx+3,mny+3);
		deallocate(geq11,inx+3,iny+3);	deallocate(gmeq11,mnx+3,mny+3);
		deallocate(geq12,inx+3,iny+3);	deallocate(gmeq12,mnx+3,mny+3);
		deallocate(geq13,inx+3,iny+3);	deallocate(gmeq13,mnx+3,mny+3);
		deallocate(geq14,inx+3,iny+3);	deallocate(gmeq14,mnx+3,mny+3);
		deallocate(geq15,inx+3,iny+3);	deallocate(gmeq15,mnx+3,mny+3);
		deallocate(geq16,inx+3,iny+3);	deallocate(gmeq16,mnx+3,mny+3);
		deallocate(geq17,inx+3,iny+3);	deallocate(gmeq17,mnx+3,mny+3);
		deallocate(geq18,inx+3,iny+3);	deallocate(gmeq18,mnx+3,mny+3);
    }
    #pragma omp section
    {
    	deallocate(heq0,inx+3,iny+3);	deallocate(hmeq0,mnx+3,mny+3);      
		deallocate(heq1,inx+3,iny+3);	deallocate(hmeq1,mnx+3,mny+3);
		deallocate(heq2,inx+3,iny+3);	deallocate(hmeq2,mnx+3,mny+3);
		deallocate(heq3,inx+3,iny+3);	deallocate(hmeq3,mnx+3,mny+3);
		deallocate(heq4,inx+3,iny+3);	deallocate(hmeq4,mnx+3,mny+3);
		deallocate(heq5,inx+3,iny+3);	deallocate(hmeq5,mnx+3,mny+3);
		deallocate(heq6,inx+3,iny+3);	deallocate(hmeq6,mnx+3,mny+3);
		deallocate(heq7,inx+3,iny+3);	deallocate(hmeq7,mnx+3,mny+3);
		deallocate(heq8,inx+3,iny+3);	deallocate(hmeq8,mnx+3,mny+3);
		deallocate(heq9,inx+3,iny+3);	deallocate(hmeq9,mnx+3,mny+3);
		deallocate(heq10,inx+3,iny+3);	deallocate(hmeq10,mnx+3,mny+3);
		deallocate(heq11,inx+3,iny+3);	deallocate(hmeq11,mnx+3,mny+3);
		deallocate(heq12,inx+3,iny+3);	deallocate(hmeq12,mnx+3,mny+3);
		deallocate(heq13,inx+3,iny+3);	deallocate(hmeq13,mnx+3,mny+3);
		deallocate(heq14,inx+3,iny+3);	deallocate(hmeq14,mnx+3,mny+3);
		deallocate(heq15,inx+3,iny+3);	deallocate(hmeq15,mnx+3,mny+3);
		deallocate(heq16,inx+3,iny+3);	deallocate(hmeq16,mnx+3,mny+3);
		deallocate(heq17,inx+3,iny+3);	deallocate(hmeq17,mnx+3,mny+3);
		deallocate(heq18,inx+3,iny+3);	deallocate(hmeq18,mnx+3,mny+3);

		deallocate(peq0,inx+3,iny+3);	deallocate(pmeq0,mnx+3,mny+3);      
		deallocate(peq1,inx+3,iny+3);	deallocate(pmeq1,mnx+3,mny+3);
		deallocate(peq2,inx+3,iny+3);	deallocate(pmeq2,mnx+3,mny+3);
		deallocate(peq3,inx+3,iny+3);	deallocate(pmeq3,mnx+3,mny+3);
		deallocate(peq4,inx+3,iny+3);	deallocate(pmeq4,mnx+3,mny+3);
		deallocate(peq5,inx+3,iny+3);	deallocate(pmeq5,mnx+3,mny+3);
		deallocate(peq6,inx+3,iny+3);	deallocate(pmeq6,mnx+3,mny+3);
		deallocate(peq7,inx+3,iny+3);	deallocate(pmeq7,mnx+3,mny+3);
		deallocate(peq8,inx+3,iny+3);	deallocate(pmeq8,mnx+3,mny+3);
		deallocate(peq9,inx+3,iny+3);	deallocate(pmeq9,mnx+3,mny+3);
		deallocate(peq10,inx+3,iny+3);	deallocate(pmeq10,mnx+3,mny+3);
		deallocate(peq11,inx+3,iny+3);	deallocate(pmeq11,mnx+3,mny+3);
		deallocate(peq12,inx+3,iny+3);	deallocate(pmeq12,mnx+3,mny+3);
		deallocate(peq13,inx+3,iny+3);	deallocate(pmeq13,mnx+3,mny+3);
		deallocate(peq14,inx+3,iny+3);	deallocate(pmeq14,mnx+3,mny+3);
		deallocate(peq15,inx+3,iny+3);	deallocate(pmeq15,mnx+3,mny+3);
		deallocate(peq16,inx+3,iny+3);	deallocate(pmeq16,mnx+3,mny+3);
		deallocate(peq17,inx+3,iny+3);	deallocate(pmeq17,mnx+3,mny+3);
		deallocate(peq18,inx+3,iny+3);	deallocate(pmeq18,mnx+3,mny+3);
    }
    #pragma omp section
    {
		deallocate(fn0,inx+3,iny+3);	deallocate(fmn0,mnx+3,mny+3);      
		deallocate(fn1,inx+3,iny+3);	deallocate(fmn1,mnx+3,mny+3);
		deallocate(fn2,inx+3,iny+3);	deallocate(fmn2,mnx+3,mny+3);
		deallocate(fn3,inx+3,iny+3);	deallocate(fmn3,mnx+3,mny+3);
		deallocate(fn4,inx+3,iny+3);	deallocate(fmn4,mnx+3,mny+3);
		deallocate(fn5,inx+3,iny+3);	deallocate(fmn5,mnx+3,mny+3);
		deallocate(fn6,inx+3,iny+3);	deallocate(fmn6,mnx+3,mny+3);
		deallocate(fn7,inx+3,iny+3);	deallocate(fmn7,mnx+3,mny+3);
		deallocate(fn8,inx+3,iny+3);	deallocate(fmn8,mnx+3,mny+3);
		deallocate(fn9,inx+3,iny+3);	deallocate(fmn9,mnx+3,mny+3);
		deallocate(fn10,inx+3,iny+3);	deallocate(fmn10,mnx+3,mny+3);
		deallocate(fn11,inx+3,iny+3);	deallocate(fmn11,mnx+3,mny+3);
		deallocate(fn12,inx+3,iny+3);	deallocate(fmn12,mnx+3,mny+3);
		deallocate(fn13,inx+3,iny+3);	deallocate(fmn13,mnx+3,mny+3);
		deallocate(fn14,inx+3,iny+3);	deallocate(fmn14,mnx+3,mny+3);
		deallocate(fn15,inx+3,iny+3);	deallocate(fmn15,mnx+3,mny+3);
		deallocate(fn16,inx+3,iny+3);	deallocate(fmn16,mnx+3,mny+3);
		deallocate(fn17,inx+3,iny+3);	deallocate(fmn17,mnx+3,mny+3);
		deallocate(fn18,inx+3,iny+3);	deallocate(fmn18,mnx+3,mny+3);

		deallocate(gn0,inx+3,iny+3);	deallocate(gmn0,mnx+3,mny+3);      
		deallocate(gn1,inx+3,iny+3);	deallocate(gmn1,mnx+3,mny+3);
		deallocate(gn2,inx+3,iny+3);	deallocate(gmn2,mnx+3,mny+3);
		deallocate(gn3,inx+3,iny+3);	deallocate(gmn3,mnx+3,mny+3);
		deallocate(gn4,inx+3,iny+3);	deallocate(gmn4,mnx+3,mny+3);
		deallocate(gn5,inx+3,iny+3);	deallocate(gmn5,mnx+3,mny+3);
		deallocate(gn6,inx+3,iny+3);	deallocate(gmn6,mnx+3,mny+3);
		deallocate(gn7,inx+3,iny+3);	deallocate(gmn7,mnx+3,mny+3);
		deallocate(gn8,inx+3,iny+3);	deallocate(gmn8,mnx+3,mny+3);
		deallocate(gn9,inx+3,iny+3);	deallocate(gmn9,mnx+3,mny+3);
		deallocate(gn10,inx+3,iny+3);	deallocate(gmn10,mnx+3,mny+3);
		deallocate(gn11,inx+3,iny+3);	deallocate(gmn11,mnx+3,mny+3);
		deallocate(gn12,inx+3,iny+3);	deallocate(gmn12,mnx+3,mny+3);
		deallocate(gn13,inx+3,iny+3);	deallocate(gmn13,mnx+3,mny+3);
		deallocate(gn14,inx+3,iny+3);	deallocate(gmn14,mnx+3,mny+3);
		deallocate(gn15,inx+3,iny+3);	deallocate(gmn15,mnx+3,mny+3);
		deallocate(gn16,inx+3,iny+3);	deallocate(gmn16,mnx+3,mny+3);
		deallocate(gn17,inx+3,iny+3);	deallocate(gmn17,mnx+3,mny+3);
		deallocate(gn18,inx+3,iny+3);	deallocate(gmn18,mnx+3,mny+3);
	}
	#pragma omp section
    {
		deallocate(hn0,inx+3,iny+3);	deallocate(hmn0,mnx+3,mny+3);      
		deallocate(hn1,inx+3,iny+3);	deallocate(hmn1,mnx+3,mny+3);
		deallocate(hn2,inx+3,iny+3);	deallocate(hmn2,mnx+3,mny+3);
		deallocate(hn3,inx+3,iny+3);	deallocate(hmn3,mnx+3,mny+3);
		deallocate(hn4,inx+3,iny+3);	deallocate(hmn4,mnx+3,mny+3);
		deallocate(hn5,inx+3,iny+3);	deallocate(hmn5,mnx+3,mny+3);
		deallocate(hn6,inx+3,iny+3);	deallocate(hmn6,mnx+3,mny+3);
		deallocate(hn7,inx+3,iny+3);	deallocate(hmn7,mnx+3,mny+3);
		deallocate(hn8,inx+3,iny+3);	deallocate(hmn8,mnx+3,mny+3);
		deallocate(hn9,inx+3,iny+3);	deallocate(hmn9,mnx+3,mny+3);
		deallocate(hn10,inx+3,iny+3);	deallocate(hmn10,mnx+3,mny+3);
		deallocate(hn11,inx+3,iny+3);	deallocate(hmn11,mnx+3,mny+3);
		deallocate(hn12,inx+3,iny+3);	deallocate(hmn12,mnx+3,mny+3);
		deallocate(hn13,inx+3,iny+3);	deallocate(hmn13,mnx+3,mny+3);
		deallocate(hn14,inx+3,iny+3);	deallocate(hmn14,mnx+3,mny+3);
		deallocate(hn15,inx+3,iny+3);	deallocate(hmn15,mnx+3,mny+3);
		deallocate(hn16,inx+3,iny+3);	deallocate(hmn16,mnx+3,mny+3);
		deallocate(hn17,inx+3,iny+3);	deallocate(hmn17,mnx+3,mny+3);
		deallocate(hn18,inx+3,iny+3);	deallocate(hmn18,mnx+3,mny+3);

		deallocate(pn0,inx+3,iny+3);	deallocate(pmn0,mnx+3,mny+3);      
		deallocate(pn1,inx+3,iny+3);	deallocate(pmn1,mnx+3,mny+3);
		deallocate(pn2,inx+3,iny+3);	deallocate(pmn2,mnx+3,mny+3);
		deallocate(pn3,inx+3,iny+3);	deallocate(pmn3,mnx+3,mny+3);
		deallocate(pn4,inx+3,iny+3);	deallocate(pmn4,mnx+3,mny+3);
		deallocate(pn5,inx+3,iny+3);	deallocate(pmn5,mnx+3,mny+3);
		deallocate(pn6,inx+3,iny+3);	deallocate(pmn6,mnx+3,mny+3);
		deallocate(pn7,inx+3,iny+3);	deallocate(pmn7,mnx+3,mny+3);
		deallocate(pn8,inx+3,iny+3);	deallocate(pmn8,mnx+3,mny+3);
		deallocate(pn9,inx+3,iny+3);	deallocate(pmn9,mnx+3,mny+3);
		deallocate(pn10,inx+3,iny+3);	deallocate(pmn10,mnx+3,mny+3);
		deallocate(pn11,inx+3,iny+3);	deallocate(pmn11,mnx+3,mny+3);
		deallocate(pn12,inx+3,iny+3);	deallocate(pmn12,mnx+3,mny+3);
		deallocate(pn13,inx+3,iny+3);	deallocate(pmn13,mnx+3,mny+3);
		deallocate(pn14,inx+3,iny+3);	deallocate(pmn14,mnx+3,mny+3);
		deallocate(pn15,inx+3,iny+3);	deallocate(pmn15,mnx+3,mny+3);
		deallocate(pn16,inx+3,iny+3);	deallocate(pmn16,mnx+3,mny+3);
		deallocate(pn17,inx+3,iny+3);	deallocate(pmn17,mnx+3,mny+3);
		deallocate(pn18,inx+3,iny+3);	deallocate(pmn18,mnx+3,mny+3);
	}
	}
}
//ei's are direction vecors of 19-speed square lattice
int ex[] = { 0, 1, 0, -1, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1, 0, 0, 0, 0 }; 
int ey[] = { 0, 0, 1, 0, -1, 0, 0, 1, 1, -1, -1, 0, 0, 0, 0, 1, -1, -1, 1 };
int ez[] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, 1, -1, -1, 1, 1, -1, -1 };

// Opening co-ordinates in the Horizontal Mixing Channel. 

int ia = ((inx + 1) / 2) - ((mnx - 1) / 2);
int ib = ((inx + 1) / 2) + ((mnx - 1) / 2);

void print_ConversionFactors_LBM_Real()
{
	printf("Length:\t%g\n", L0);
	printf("Mass:\t%e\n", M0);
	printf("Time:\t%e\n", T0);
	printf("Velocity:\t%e\n", V0);
	printf("pressure:\t%e\n", P0);
	printf("Specific Power:\t%e\n", Pw0);
	printf("KinematicViscosity:\t%e\n", Kv0);
}

void writeParameters(double Re)
{
	FILE *fp11;  // File Handler
	char buffer[125]; // Dynamic File name holder

	sprintf(buffer, "Simulation_parameters.txt");
	fp11 = fopen(buffer, "w");
	fprintf(fp11, "***********************************\n");
	fprintf(fp11, "******Simulation Parameters********\n");
	fprintf(fp11, "***********************************\n\n");
	fprintf(fp11, "nx:%d\nny:%d\nnz:%d\nInlet-Channel Width:%d\nOutlet-Channel Width:%d\n", inx, (iny + mny), nz, iny, mnx);
	fprintf(fp11, "Relaxation-Parameter of tracer:%lf\n",taua);
	fprintf(fp11, "Relaxation-Parameter:%lf\nUmean:%lf\n", tau, Umean);
	fprintf(fp11, "Reaction is modelled as A + B ----> C -ra = KCaCb\n");
	fprintf(fp11, "Reaction rate constant : %lf\n",rateconst);
	fprintf(fp11, "Mixing channel opening coordinates(ia,ib):(%d,%d)\n\n", ia, ib);
	fprintf(fp11, "Reynolds Number (Outlet channel): %lf\n\n", Re);
	fprintf(fp11, "Schmidt Number: %lf\n\n", (2.0*tau-1.0)/(2.0*taua-1.0));

	fprintf(fp11, "********************************\n");
	fprintf(fp11, "******Conversion Factors********\n");
	fprintf(fp11, "********************************\n\n");
	fprintf(fp11, "Length:\t%g\n", L0);
	fprintf(fp11, "Mass:\t%e\n", M0);
	fprintf(fp11, "Time:\t%e\n", T0);
	fprintf(fp11, "Velocity:\t%e\n", V0);
	fprintf(fp11, "pressure:\t%e\n", P0);
	fprintf(fp11, "Specific Power:\t%e\n", Pw0);
	fprintf(fp11, "KinematicViscosity:\t%e\n", Kv0);
	fprintf(fp11, "****************************************\n");
	fprintf(fp11, "******Conversion Factors Density********\n");
	fprintf(fp11, "****************************************\n\n");
	fprintf(fp11, "Mass:\t%e\n", M01);
	fprintf(fp11, "pressure:\t%e\n", P01);
	fclose(fp11);
}

/*Input velocity generation ..*/
void GenerateInletVelocity()
{
	/*
	For More Information see the following paper
	"Analytical and Numerical Investigations of the Effects of Microchannel
	Aspect Ratio on Velocity Profile and Friction Factor"
	The 4th International Conference on Computational Methods (ICCM2012), Gold Coast,Australia
	www.ICCM-2012.org
	*/
	//double F = 0;

	double nsum, msum, N, D;

	int k = 0, j = 0, m = 0, n = 0;
	int i = 0;
	char buffer[125];
	FILE *velocity;
	nsum = 0;
	msum = 0;
	
	printf("The value of h,w is %g,%g\n", d, w);
	printf("The value of AR is %lf\n", AR);

	#pragma omp parallel for collapse(2) private(nsum,msum,n,m,N,D)
	for (j = 0; j < iny; j++)
	{
		for (k = 0; k < nz; k++)
		{
			nsum = 0;
			#pragma omp parallel for reduction(+:nsum) private(msum)
			for (n = 1; n <= LIMIT; n++)
			{
				msum = 0;
				#pragma omp parallel for reduction(+:msum) private(N,D)
				for (m = 1; m <= LIMIT; m++)
				{
					N = (1.0 - cos(m*pi))*(1.0 - cos(n*pi))*sin(((m*pi) / (2.0*AR*d))*(j))*sin(((n*pi) / (2.0*d))*(k));
					D = m*n*((m*m) + (AR*AR*n*n));
					msum += (N / D);
				}

				nsum += msum;
			}
			Uin1[j + 1][k+1] = nsum;								
		}
	}
	double sum = 0;
	for (i = 1; i <= iny; i++)
	{
		for (j = 1; j <= nz; j++)
		{
			sum += Uin1[i][j];
		}
	}
	double average = sum/((iny)*(nz));
	for (i = 1; i <= iny; i++)
	{
		for (j = 1; j <= nz; j++)
		{
			Uin1[i][j] = Uin1[i][j]*(Umean/average);
		}
	}
	sprintf(buffer, "InletVelocity.dat");
	velocity = fopen(buffer, "w");

	fprintf(velocity, "ZONE T ='Normal' VARIABLES = 'X','Y' 'U1' 'U2'\r\n");

	printf("Writing the file..\n");

	for (i = 1; i <= iny; i++)
	{
		for (j = 1; j <= nz; j++)
		{
			fprintf(velocity, "%d\t%d\t%lf\t%lf\n", i, j, Uin1[i][j],Uin2[i][j]);
		}
		fprintf(velocity, "\n");
	}
	fclose(velocity);

	printf("Wrote the Inlet velocity to file..\n");

	for (j = 0; j < iny; j++)
	{
		for (k = 0; k < nz; k++)
		{
			Uin1[j + 1][k] = Uin1[j+1][k+1];
			Uin2[j + 1][k + 2] = -Uin1[j+1][k];
		}
	}

	for (i = 1; i <= iny; i++)
	{
		
			Uin1[i][1] = 0;
			Uin2[i][nz] = 0;
		
	}
	
	sprintf(buffer, "InletVelocity_perturbed.dat");
	velocity = fopen(buffer, "w");

	fprintf(velocity, "ZONE T ='perturbed' VARIABLES = 'X','Y' 'U1' 'U2'\r\n");

	for (i = 1; i <= iny; i++)
	{
		for (j = 1; j <= nz; j++)
		{
			fprintf(velocity, "%d\t%d\t%lf\t%lf\n", i, j, Uin1[i][j],Uin2[i][j]);
		}
		fprintf(velocity, "\n");
	}
	fclose(velocity);
	
}

/*----------------------------------------------------------------------------------
Initializing macroscopic properties
------------------------------------------------------------------------------------*/
void initialization()
{
	int i, j, p;
	// Initializing in the inlet horizontal channel 
	#pragma omp parallel for collapse(3)
	for (i = 0; i <= inx + 1; i++)
	{
		for (j = 0; j <= iny + 1; j++)
		{
			for (p = 0; p <= nz + 1; p++)
			{
				rho[i][j][p] = rho0;
				grho[i][j][p] = C0;
				hrho[i][j][p] = C0;
				prho[i][j][p] = C0;
				
				ux[i][j][p] = uxo;
				uy[i][j][p] = uyo;
				uz[i][j][p] = uzo;
			}
		}
	}
	// Initializing in the mixing channel 
	#pragma omp parallel for collapse(3)
	for (i = 0; i <= mnx + 1; i++)
	{
		for (j = 0; j <= mny + 1; j++)
		{
			for (p = 0; p <= nz + 1; p++)
			{
				mrho[i][j][p] = rho0;
				gmrho[i][j][p] = C0;
				hmrho[i][j][p] = C0;
				pmrho[i][j][p] = C0;

				mux[i][j][p] = uxo;
				muy[i][j][p] = uyo;
				muz[i][j][p] = uzo;
			}
		}
	}
}
/*----------------------------------------------------------------------------------
Defining Equilibrium Distribution function for 19-speed square lattice (D3Q19 Model)
-----------------------------------------------------------------------------------------------*/
void equilibrium()
{
	int i, j, p;

	// Equilibrium distribution function at i=0
	// Inlet Channel
	#pragma omp parallel for collapse(3)
	for (i = 0; i <= inx + 1; i++)
	{
		for (j = 0; j <= iny + 1; j++)
		{
			for (p = 0; p <= nz + 1; p++)
			{
				feq0[i][j][p] = rho[i][j][p]  * (1.0 - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0) / 3.0;
				geq0[i][j][p] = grho[i][j][p] * (1.0 - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0) / 3.0;
				heq0[i][j][p] = hrho[i][j][p] * (1.0 - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0) / 3.0;
				peq0[i][j][p] = prho[i][j][p] * (1.0 - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0) / 3.0;
			}
		}
	}
	// Mixing Channel
	#pragma omp parallel for collapse(3)
	for (i = 0; i <= mnx + 1; i++)
	{
		for (j = 0; j <= mny + 1; j++)
		{
			for (p = 0; p <= nz + 1; p++)
			{
				fmeq0[i][j][p] = mrho[i][j][p]  * (1.0 - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0) / 3.0;
				gmeq0[i][j][p] = gmrho[i][j][p] * (1.0 - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0) / 3.0;
				hmeq0[i][j][p] = hmrho[i][j][p] * (1.0 - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0) / 3.0;
				pmeq0[i][j][p] = pmrho[i][j][p] * (1.0 - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0) / 3.0;
			}
		}
	}
	// Equilibrium distribution function for i=1,2,3,4,5,6
	//Inlet Channel
	#pragma omp parallel for collapse(3)
	for (i = 0; i <= inx + 1; i++)
	{
		for (j = 0; j <= iny + 1; j++)
		{
			for (p = 0; p <= nz + 1; p++)
			{
				feq1[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[1] + uy[i][j][p] * ey[1] + uz[i][j][p] * ez[1]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[1] * ux[i][j][p] + ey[1] * uy[i][j][p] + ez[1] * uz[i][j][p])*(ex[1] * ux[i][j][p] + ey[1] * uy[i][j][p] + ez[1] * uz[i][j][p]) / 2.0) / 18.0;
				feq2[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[2] + uy[i][j][p] * ey[2] + uz[i][j][p] * ez[2]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[2] * ux[i][j][p] + ey[2] * uy[i][j][p] + ez[2] * uz[i][j][p])*(ex[2] * ux[i][j][p] + ey[2] * uy[i][j][p] + ez[2] * uz[i][j][p]) / 2.0) / 18.0;
				feq3[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[3] + uy[i][j][p] * ey[3] + uz[i][j][p] * ez[3]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[3] * ux[i][j][p] + ey[3] * uy[i][j][p] + ez[3] * uz[i][j][p])*(ex[3] * ux[i][j][p] + ey[3] * uy[i][j][p] + ez[3] * uz[i][j][p]) / 2.0) / 18.0;
				feq4[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[4] + uy[i][j][p] * ey[4] + uz[i][j][p] * ez[4]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[4] * ux[i][j][p] + ey[4] * uy[i][j][p] + ez[4] * uz[i][j][p])*(ex[4] * ux[i][j][p] + ey[4] * uy[i][j][p] + ez[4] * uz[i][j][p]) / 2.0) / 18.0;
				feq5[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[5] + uy[i][j][p] * ey[5] + uz[i][j][p] * ez[5]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[5] * ux[i][j][p] + ey[5] * uy[i][j][p] + ez[5] * uz[i][j][p])*(ex[5] * ux[i][j][p] + ey[5] * uy[i][j][p] + ez[5] * uz[i][j][p]) / 2.0) / 18.0;
				feq6[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[6] + uy[i][j][p] * ey[6] + uz[i][j][p] * ez[6]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[6] * ux[i][j][p] + ey[6] * uy[i][j][p] + ez[6] * uz[i][j][p])*(ex[6] * ux[i][j][p] + ey[6] * uy[i][j][p] + ez[6] * uz[i][j][p]) / 2.0) / 18.0;

				geq1[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[1] + uy[i][j][p] * ey[1] + uz[i][j][p] * ez[1]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[1] * ux[i][j][p] + ey[1] * uy[i][j][p] + ez[1] * uz[i][j][p])*(ex[1] * ux[i][j][p] + ey[1] * uy[i][j][p] + ez[1] * uz[i][j][p]) / 2.0) / 18.0;
				geq2[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[2] + uy[i][j][p] * ey[2] + uz[i][j][p] * ez[2]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[2] * ux[i][j][p] + ey[2] * uy[i][j][p] + ez[2] * uz[i][j][p])*(ex[2] * ux[i][j][p] + ey[2] * uy[i][j][p] + ez[2] * uz[i][j][p]) / 2.0) / 18.0;
				geq3[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[3] + uy[i][j][p] * ey[3] + uz[i][j][p] * ez[3]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[3] * ux[i][j][p] + ey[3] * uy[i][j][p] + ez[3] * uz[i][j][p])*(ex[3] * ux[i][j][p] + ey[3] * uy[i][j][p] + ez[3] * uz[i][j][p]) / 2.0) / 18.0;
				geq4[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[4] + uy[i][j][p] * ey[4] + uz[i][j][p] * ez[4]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[4] * ux[i][j][p] + ey[4] * uy[i][j][p] + ez[4] * uz[i][j][p])*(ex[4] * ux[i][j][p] + ey[4] * uy[i][j][p] + ez[4] * uz[i][j][p]) / 2.0) / 18.0;
				geq5[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[5] + uy[i][j][p] * ey[5] + uz[i][j][p] * ez[5]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[5] * ux[i][j][p] + ey[5] * uy[i][j][p] + ez[5] * uz[i][j][p])*(ex[5] * ux[i][j][p] + ey[5] * uy[i][j][p] + ez[5] * uz[i][j][p]) / 2.0) / 18.0;
				geq6[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[6] + uy[i][j][p] * ey[6] + uz[i][j][p] * ez[6]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[6] * ux[i][j][p] + ey[6] * uy[i][j][p] + ez[6] * uz[i][j][p])*(ex[6] * ux[i][j][p] + ey[6] * uy[i][j][p] + ez[6] * uz[i][j][p]) / 2.0) / 18.0;

				heq1[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[1] + uy[i][j][p] * ey[1] + uz[i][j][p] * ez[1]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[1] * ux[i][j][p] + ey[1] * uy[i][j][p] + ez[1] * uz[i][j][p])*(ex[1] * ux[i][j][p] + ey[1] * uy[i][j][p] + ez[1] * uz[i][j][p]) / 2.0) / 18.0;
				heq2[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[2] + uy[i][j][p] * ey[2] + uz[i][j][p] * ez[2]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[2] * ux[i][j][p] + ey[2] * uy[i][j][p] + ez[2] * uz[i][j][p])*(ex[2] * ux[i][j][p] + ey[2] * uy[i][j][p] + ez[2] * uz[i][j][p]) / 2.0) / 18.0;
				heq3[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[3] + uy[i][j][p] * ey[3] + uz[i][j][p] * ez[3]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[3] * ux[i][j][p] + ey[3] * uy[i][j][p] + ez[3] * uz[i][j][p])*(ex[3] * ux[i][j][p] + ey[3] * uy[i][j][p] + ez[3] * uz[i][j][p]) / 2.0) / 18.0;
				heq4[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[4] + uy[i][j][p] * ey[4] + uz[i][j][p] * ez[4]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[4] * ux[i][j][p] + ey[4] * uy[i][j][p] + ez[4] * uz[i][j][p])*(ex[4] * ux[i][j][p] + ey[4] * uy[i][j][p] + ez[4] * uz[i][j][p]) / 2.0) / 18.0;
				heq5[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[5] + uy[i][j][p] * ey[5] + uz[i][j][p] * ez[5]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[5] * ux[i][j][p] + ey[5] * uy[i][j][p] + ez[5] * uz[i][j][p])*(ex[5] * ux[i][j][p] + ey[5] * uy[i][j][p] + ez[5] * uz[i][j][p]) / 2.0) / 18.0;
				heq6[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[6] + uy[i][j][p] * ey[6] + uz[i][j][p] * ez[6]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[6] * ux[i][j][p] + ey[6] * uy[i][j][p] + ez[6] * uz[i][j][p])*(ex[6] * ux[i][j][p] + ey[6] * uy[i][j][p] + ez[6] * uz[i][j][p]) / 2.0) / 18.0;

				peq1[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[1] + uy[i][j][p] * ey[1] + uz[i][j][p] * ez[1]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[1] * ux[i][j][p] + ey[1] * uy[i][j][p] + ez[1] * uz[i][j][p])*(ex[1] * ux[i][j][p] + ey[1] * uy[i][j][p] + ez[1] * uz[i][j][p]) / 2.0) / 18.0;
				peq2[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[2] + uy[i][j][p] * ey[2] + uz[i][j][p] * ez[2]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[2] * ux[i][j][p] + ey[2] * uy[i][j][p] + ez[2] * uz[i][j][p])*(ex[2] * ux[i][j][p] + ey[2] * uy[i][j][p] + ez[2] * uz[i][j][p]) / 2.0) / 18.0;
				peq3[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[3] + uy[i][j][p] * ey[3] + uz[i][j][p] * ez[3]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[3] * ux[i][j][p] + ey[3] * uy[i][j][p] + ez[3] * uz[i][j][p])*(ex[3] * ux[i][j][p] + ey[3] * uy[i][j][p] + ez[3] * uz[i][j][p]) / 2.0) / 18.0;
				peq4[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[4] + uy[i][j][p] * ey[4] + uz[i][j][p] * ez[4]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[4] * ux[i][j][p] + ey[4] * uy[i][j][p] + ez[4] * uz[i][j][p])*(ex[4] * ux[i][j][p] + ey[4] * uy[i][j][p] + ez[4] * uz[i][j][p]) / 2.0) / 18.0;
				peq5[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[5] + uy[i][j][p] * ey[5] + uz[i][j][p] * ez[5]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[5] * ux[i][j][p] + ey[5] * uy[i][j][p] + ez[5] * uz[i][j][p])*(ex[5] * ux[i][j][p] + ey[5] * uy[i][j][p] + ez[5] * uz[i][j][p]) / 2.0) / 18.0;
				peq6[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[6] + uy[i][j][p] * ey[6] + uz[i][j][p] * ez[6]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[6] * ux[i][j][p] + ey[6] * uy[i][j][p] + ez[6] * uz[i][j][p])*(ex[6] * ux[i][j][p] + ey[6] * uy[i][j][p] + ez[6] * uz[i][j][p]) / 2.0) / 18.0;
			}
		}
	}
	//Mixing Channel
	#pragma omp parallel for collapse(3)
	for (i = 0; i <= mnx + 1; i++)
	{
		for (j = 0; j <= mny + 1; j++)
		{
			for (p = 0; p <= nz + 1; p++)
			{
				fmeq1[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[1] + muy[i][j][p] * ey[1] + muz[i][j][p] * ez[1]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[1] * mux[i][j][p] + ey[1] * muy[i][j][p] + ez[1] * muz[i][j][p])*(ex[1] * mux[i][j][p] + ey[1] * muy[i][j][p] + ez[1] * muz[i][j][p]) / 2.0) / 18.0;
				fmeq2[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[2] + muy[i][j][p] * ey[2] + muz[i][j][p] * ez[2]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[2] * mux[i][j][p] + ey[2] * muy[i][j][p] + ez[2] * muz[i][j][p])*(ex[2] * mux[i][j][p] + ey[2] * muy[i][j][p] + ez[2] * muz[i][j][p]) / 2.0) / 18.0;
				fmeq3[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[3] + muy[i][j][p] * ey[3] + muz[i][j][p] * ez[3]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[3] * mux[i][j][p] + ey[3] * muy[i][j][p] + ez[3] * muz[i][j][p])*(ex[3] * mux[i][j][p] + ey[3] * muy[i][j][p] + ez[3] * muz[i][j][p]) / 2.0) / 18.0;
				fmeq4[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[4] + muy[i][j][p] * ey[4] + muz[i][j][p] * ez[4]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[4] * mux[i][j][p] + ey[4] * muy[i][j][p] + ez[4] * muz[i][j][p])*(ex[4] * mux[i][j][p] + ey[4] * muy[i][j][p] + ez[4] * muz[i][j][p]) / 2.0) / 18.0;
				fmeq5[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[5] + muy[i][j][p] * ey[5] + muz[i][j][p] * ez[5]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[5] * mux[i][j][p] + ey[5] * muy[i][j][p] + ez[5] * muz[i][j][p])*(ex[5] * mux[i][j][p] + ey[5] * muy[i][j][p] + ez[5] * muz[i][j][p]) / 2.0) / 18.0;
				fmeq6[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[6] + muy[i][j][p] * ey[6] + muz[i][j][p] * ez[6]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[6] * mux[i][j][p] + ey[6] * muy[i][j][p] + ez[6] * muz[i][j][p])*(ex[6] * mux[i][j][p] + ey[6] * muy[i][j][p] + ez[6] * muz[i][j][p]) / 2.0) / 18.0;

				gmeq1[i][j][p] = gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[1] + muy[i][j][p] * ey[1] + muz[i][j][p] * ez[1]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[1] * mux[i][j][p] + ey[1] * muy[i][j][p] + ez[1] * muz[i][j][p])*(ex[1] * mux[i][j][p] + ey[1] * muy[i][j][p] + ez[1] * muz[i][j][p]) / 2.0) / 18.0;
				gmeq2[i][j][p] = gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[2] + muy[i][j][p] * ey[2] + muz[i][j][p] * ez[2]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[2] * mux[i][j][p] + ey[2] * muy[i][j][p] + ez[2] * muz[i][j][p])*(ex[2] * mux[i][j][p] + ey[2] * muy[i][j][p] + ez[2] * muz[i][j][p]) / 2.0) / 18.0;
				gmeq3[i][j][p] = gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[3] + muy[i][j][p] * ey[3] + muz[i][j][p] * ez[3]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[3] * mux[i][j][p] + ey[3] * muy[i][j][p] + ez[3] * muz[i][j][p])*(ex[3] * mux[i][j][p] + ey[3] * muy[i][j][p] + ez[3] * muz[i][j][p]) / 2.0) / 18.0;
				gmeq4[i][j][p] = gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[4] + muy[i][j][p] * ey[4] + muz[i][j][p] * ez[4]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[4] * mux[i][j][p] + ey[4] * muy[i][j][p] + ez[4] * muz[i][j][p])*(ex[4] * mux[i][j][p] + ey[4] * muy[i][j][p] + ez[4] * muz[i][j][p]) / 2.0) / 18.0;
				gmeq5[i][j][p] = gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[5] + muy[i][j][p] * ey[5] + muz[i][j][p] * ez[5]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[5] * mux[i][j][p] + ey[5] * muy[i][j][p] + ez[5] * muz[i][j][p])*(ex[5] * mux[i][j][p] + ey[5] * muy[i][j][p] + ez[5] * muz[i][j][p]) / 2.0) / 18.0;
				gmeq6[i][j][p] = gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[6] + muy[i][j][p] * ey[6] + muz[i][j][p] * ez[6]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[6] * mux[i][j][p] + ey[6] * muy[i][j][p] + ez[6] * muz[i][j][p])*(ex[6] * mux[i][j][p] + ey[6] * muy[i][j][p] + ez[6] * muz[i][j][p]) / 2.0) / 18.0;

				hmeq1[i][j][p] = hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[1] + muy[i][j][p] * ey[1] + muz[i][j][p] * ez[1]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[1] * mux[i][j][p] + ey[1] * muy[i][j][p] + ez[1] * muz[i][j][p])*(ex[1] * mux[i][j][p] + ey[1] * muy[i][j][p] + ez[1] * muz[i][j][p]) / 2.0) / 18.0;
				hmeq2[i][j][p] = hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[2] + muy[i][j][p] * ey[2] + muz[i][j][p] * ez[2]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[2] * mux[i][j][p] + ey[2] * muy[i][j][p] + ez[2] * muz[i][j][p])*(ex[2] * mux[i][j][p] + ey[2] * muy[i][j][p] + ez[2] * muz[i][j][p]) / 2.0) / 18.0;
				hmeq3[i][j][p] = hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[3] + muy[i][j][p] * ey[3] + muz[i][j][p] * ez[3]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[3] * mux[i][j][p] + ey[3] * muy[i][j][p] + ez[3] * muz[i][j][p])*(ex[3] * mux[i][j][p] + ey[3] * muy[i][j][p] + ez[3] * muz[i][j][p]) / 2.0) / 18.0;
				hmeq4[i][j][p] = hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[4] + muy[i][j][p] * ey[4] + muz[i][j][p] * ez[4]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[4] * mux[i][j][p] + ey[4] * muy[i][j][p] + ez[4] * muz[i][j][p])*(ex[4] * mux[i][j][p] + ey[4] * muy[i][j][p] + ez[4] * muz[i][j][p]) / 2.0) / 18.0;
				hmeq5[i][j][p] = hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[5] + muy[i][j][p] * ey[5] + muz[i][j][p] * ez[5]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[5] * mux[i][j][p] + ey[5] * muy[i][j][p] + ez[5] * muz[i][j][p])*(ex[5] * mux[i][j][p] + ey[5] * muy[i][j][p] + ez[5] * muz[i][j][p]) / 2.0) / 18.0;
				hmeq6[i][j][p] = hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[6] + muy[i][j][p] * ey[6] + muz[i][j][p] * ez[6]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[6] * mux[i][j][p] + ey[6] * muy[i][j][p] + ez[6] * muz[i][j][p])*(ex[6] * mux[i][j][p] + ey[6] * muy[i][j][p] + ez[6] * muz[i][j][p]) / 2.0) / 18.0;

				pmeq1[i][j][p] = pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[1] + muy[i][j][p] * ey[1] + muz[i][j][p] * ez[1]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[1] * mux[i][j][p] + ey[1] * muy[i][j][p] + ez[1] * muz[i][j][p])*(ex[1] * mux[i][j][p] + ey[1] * muy[i][j][p] + ez[1] * muz[i][j][p]) / 2.0) / 18.0;
				pmeq2[i][j][p] = pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[2] + muy[i][j][p] * ey[2] + muz[i][j][p] * ez[2]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[2] * mux[i][j][p] + ey[2] * muy[i][j][p] + ez[2] * muz[i][j][p])*(ex[2] * mux[i][j][p] + ey[2] * muy[i][j][p] + ez[2] * muz[i][j][p]) / 2.0) / 18.0;
				pmeq3[i][j][p] = pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[3] + muy[i][j][p] * ey[3] + muz[i][j][p] * ez[3]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[3] * mux[i][j][p] + ey[3] * muy[i][j][p] + ez[3] * muz[i][j][p])*(ex[3] * mux[i][j][p] + ey[3] * muy[i][j][p] + ez[3] * muz[i][j][p]) / 2.0) / 18.0;
				pmeq4[i][j][p] = pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[4] + muy[i][j][p] * ey[4] + muz[i][j][p] * ez[4]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[4] * mux[i][j][p] + ey[4] * muy[i][j][p] + ez[4] * muz[i][j][p])*(ex[4] * mux[i][j][p] + ey[4] * muy[i][j][p] + ez[4] * muz[i][j][p]) / 2.0) / 18.0;
				pmeq5[i][j][p] = pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[5] + muy[i][j][p] * ey[5] + muz[i][j][p] * ez[5]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[5] * mux[i][j][p] + ey[5] * muy[i][j][p] + ez[5] * muz[i][j][p])*(ex[5] * mux[i][j][p] + ey[5] * muy[i][j][p] + ez[5] * muz[i][j][p]) / 2.0) / 18.0;
				pmeq6[i][j][p] = pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[6] + muy[i][j][p] * ey[6] + muz[i][j][p] * ez[6]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[6] * mux[i][j][p] + ey[6] * muy[i][j][p] + ez[6] * muz[i][j][p])*(ex[6] * mux[i][j][p] + ey[6] * muy[i][j][p] + ez[6] * muz[i][j][p]) / 2.0) / 18.0;
			}
		}
	}

	// Equilibrium distribution function for i=7,8,------18
	//Inlet Channel
	#pragma omp parallel for collapse(3)
	for (i = 0; i <= inx + 1; i++)
	{
		for (j = 0; j <= iny + 1; j++)
		{
			for (p = 0; p <= nz + 1; p++)
			{
				feq7[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[7] + uy[i][j][p] * ey[7] + uz[i][j][p] * ez[7]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[7] * ux[i][j][p] + ey[7] * uy[i][j][p] + ez[7] * uz[i][j][p])*(ex[7] * ux[i][j][p] + ey[7] * uy[i][j][p] + ez[7] * uz[i][j][p]) / 2.0) / 36.0;
				feq8[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[8] + uy[i][j][p] * ey[8] + uz[i][j][p] * ez[8]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[8] * ux[i][j][p] + ey[8] * uy[i][j][p] + ez[8] * uz[i][j][p])*(ex[8] * ux[i][j][p] + ey[8] * uy[i][j][p] + ez[8] * uz[i][j][p]) / 2.0) / 36.0;
				feq9[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[9] + uy[i][j][p] * ey[9] + uz[i][j][p] * ez[9]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[9] * ux[i][j][p] + ey[9] * uy[i][j][p] + ez[9] * uz[i][j][p])*(ex[9] * ux[i][j][p] + ey[9] * uy[i][j][p] + ez[9] * uz[i][j][p]) / 2.0) / 36.0;
				feq10[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[10] + uy[i][j][p] * ey[10] + uz[i][j][p] * ez[10]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[10] * ux[i][j][p] + ey[10] * uy[i][j][p] + ez[10] * uz[i][j][p])*(ex[10] * ux[i][j][p] + ey[10] * uy[i][j][p] + ez[10] * uz[i][j][p]) / 2.0) / 36.0;
				feq11[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[11] + uy[i][j][p] * ey[11] + uz[i][j][p] * ez[11]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[11] * ux[i][j][p] + ey[11] * uy[i][j][p] + ez[11] * uz[i][j][p])*(ex[11] * ux[i][j][p] + ey[11] * uy[i][j][p] + ez[11] * uz[i][j][p]) / 2.0) / 36.0;
				feq12[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[12] + uy[i][j][p] * ey[12] + uz[i][j][p] * ez[12]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[12] * ux[i][j][p] + ey[12] * uy[i][j][p] + ez[12] * uz[i][j][p])*(ex[12] * ux[i][j][p] + ey[12] * uy[i][j][p] + ez[12] * uz[i][j][p]) / 2.0) / 36.0;
				feq13[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[13] + uy[i][j][p] * ey[13] + uz[i][j][p] * ez[13]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[13] * ux[i][j][p] + ey[13] * uy[i][j][p] + ez[13] * uz[i][j][p])*(ex[13] * ux[i][j][p] + ey[13] * uy[i][j][p] + ez[13] * uz[i][j][p]) / 2.0) / 36.0;
				feq14[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[14] + uy[i][j][p] * ey[14] + uz[i][j][p] * ez[14]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[14] * ux[i][j][p] + ey[14] * uy[i][j][p] + ez[14] * uz[i][j][p])*(ex[14] * ux[i][j][p] + ey[14] * uy[i][j][p] + ez[14] * uz[i][j][p]) / 2.0) / 36.0;
				feq15[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[15] + uy[i][j][p] * ey[15] + uz[i][j][p] * ez[15]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[15] * ux[i][j][p] + ey[15] * uy[i][j][p] + ez[15] * uz[i][j][p])*(ex[15] * ux[i][j][p] + ey[15] * uy[i][j][p] + ez[15] * uz[i][j][p]) / 2.0) / 36.0;
				feq16[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[16] + uy[i][j][p] * ey[16] + uz[i][j][p] * ez[16]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[16] * ux[i][j][p] + ey[16] * uy[i][j][p] + ez[16] * uz[i][j][p])*(ex[16] * ux[i][j][p] + ey[16] * uy[i][j][p] + ez[16] * uz[i][j][p]) / 2.0) / 36.0;
				feq17[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[17] + uy[i][j][p] * ey[17] + uz[i][j][p] * ez[17]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[17] * ux[i][j][p] + ey[17] * uy[i][j][p] + ez[17] * uz[i][j][p])*(ex[17] * ux[i][j][p] + ey[17] * uy[i][j][p] + ez[17] * uz[i][j][p]) / 2.0) / 36.0;
				feq18[i][j][p] = rho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[18] + uy[i][j][p] * ey[18] + uz[i][j][p] * ez[18]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[18] * ux[i][j][p] + ey[18] * uy[i][j][p] + ez[18] * uz[i][j][p])*(ex[18] * ux[i][j][p] + ey[18] * uy[i][j][p] + ez[18] * uz[i][j][p]) / 2.0) / 36.0;

				geq7[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[7] + uy[i][j][p] * ey[7] + uz[i][j][p] * ez[7]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[7] * ux[i][j][p] + ey[7] * uy[i][j][p] + ez[7] * uz[i][j][p])*(ex[7] * ux[i][j][p] + ey[7] * uy[i][j][p] + ez[7] * uz[i][j][p]) / 2.0) / 36.0;
				geq8[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[8] + uy[i][j][p] * ey[8] + uz[i][j][p] * ez[8]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[8] * ux[i][j][p] + ey[8] * uy[i][j][p] + ez[8] * uz[i][j][p])*(ex[8] * ux[i][j][p] + ey[8] * uy[i][j][p] + ez[8] * uz[i][j][p]) / 2.0) / 36.0;
				geq9[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[9] + uy[i][j][p] * ey[9] + uz[i][j][p] * ez[9]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[9] * ux[i][j][p] + ey[9] * uy[i][j][p] + ez[9] * uz[i][j][p])*(ex[9] * ux[i][j][p] + ey[9] * uy[i][j][p] + ez[9] * uz[i][j][p]) / 2.0) / 36.0;
				geq10[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[10] + uy[i][j][p] * ey[10] + uz[i][j][p] * ez[10]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[10] * ux[i][j][p] + ey[10] * uy[i][j][p] + ez[10] * uz[i][j][p])*(ex[10] * ux[i][j][p] + ey[10] * uy[i][j][p] + ez[10] * uz[i][j][p]) / 2.0) / 36.0;
				geq11[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[11] + uy[i][j][p] * ey[11] + uz[i][j][p] * ez[11]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[11] * ux[i][j][p] + ey[11] * uy[i][j][p] + ez[11] * uz[i][j][p])*(ex[11] * ux[i][j][p] + ey[11] * uy[i][j][p] + ez[11] * uz[i][j][p]) / 2.0) / 36.0;
				geq12[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[12] + uy[i][j][p] * ey[12] + uz[i][j][p] * ez[12]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[12] * ux[i][j][p] + ey[12] * uy[i][j][p] + ez[12] * uz[i][j][p])*(ex[12] * ux[i][j][p] + ey[12] * uy[i][j][p] + ez[12] * uz[i][j][p]) / 2.0) / 36.0;
				geq13[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[13] + uy[i][j][p] * ey[13] + uz[i][j][p] * ez[13]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[13] * ux[i][j][p] + ey[13] * uy[i][j][p] + ez[13] * uz[i][j][p])*(ex[13] * ux[i][j][p] + ey[13] * uy[i][j][p] + ez[13] * uz[i][j][p]) / 2.0) / 36.0;
				geq14[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[14] + uy[i][j][p] * ey[14] + uz[i][j][p] * ez[14]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[14] * ux[i][j][p] + ey[14] * uy[i][j][p] + ez[14] * uz[i][j][p])*(ex[14] * ux[i][j][p] + ey[14] * uy[i][j][p] + ez[14] * uz[i][j][p]) / 2.0) / 36.0;
				geq15[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[15] + uy[i][j][p] * ey[15] + uz[i][j][p] * ez[15]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[15] * ux[i][j][p] + ey[15] * uy[i][j][p] + ez[15] * uz[i][j][p])*(ex[15] * ux[i][j][p] + ey[15] * uy[i][j][p] + ez[15] * uz[i][j][p]) / 2.0) / 36.0;
				geq16[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[16] + uy[i][j][p] * ey[16] + uz[i][j][p] * ez[16]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[16] * ux[i][j][p] + ey[16] * uy[i][j][p] + ez[16] * uz[i][j][p])*(ex[16] * ux[i][j][p] + ey[16] * uy[i][j][p] + ez[16] * uz[i][j][p]) / 2.0) / 36.0;
				geq17[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[17] + uy[i][j][p] * ey[17] + uz[i][j][p] * ez[17]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[17] * ux[i][j][p] + ey[17] * uy[i][j][p] + ez[17] * uz[i][j][p])*(ex[17] * ux[i][j][p] + ey[17] * uy[i][j][p] + ez[17] * uz[i][j][p]) / 2.0) / 36.0;
				geq18[i][j][p] = grho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[18] + uy[i][j][p] * ey[18] + uz[i][j][p] * ez[18]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[18] * ux[i][j][p] + ey[18] * uy[i][j][p] + ez[18] * uz[i][j][p])*(ex[18] * ux[i][j][p] + ey[18] * uy[i][j][p] + ez[18] * uz[i][j][p]) / 2.0) / 36.0;

				heq7[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[7] + uy[i][j][p] * ey[7] + uz[i][j][p] * ez[7]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[7] * ux[i][j][p] + ey[7] * uy[i][j][p] + ez[7] * uz[i][j][p])*(ex[7] * ux[i][j][p] + ey[7] * uy[i][j][p] + ez[7] * uz[i][j][p]) / 2.0) / 36.0;
				heq8[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[8] + uy[i][j][p] * ey[8] + uz[i][j][p] * ez[8]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[8] * ux[i][j][p] + ey[8] * uy[i][j][p] + ez[8] * uz[i][j][p])*(ex[8] * ux[i][j][p] + ey[8] * uy[i][j][p] + ez[8] * uz[i][j][p]) / 2.0) / 36.0;
				heq9[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[9] + uy[i][j][p] * ey[9] + uz[i][j][p] * ez[9]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[9] * ux[i][j][p] + ey[9] * uy[i][j][p] + ez[9] * uz[i][j][p])*(ex[9] * ux[i][j][p] + ey[9] * uy[i][j][p] + ez[9] * uz[i][j][p]) / 2.0) / 36.0;
				heq10[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[10] + uy[i][j][p] * ey[10] + uz[i][j][p] * ez[10]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[10] * ux[i][j][p] + ey[10] * uy[i][j][p] + ez[10] * uz[i][j][p])*(ex[10] * ux[i][j][p] + ey[10] * uy[i][j][p] + ez[10] * uz[i][j][p]) / 2.0) / 36.0;
				heq11[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[11] + uy[i][j][p] * ey[11] + uz[i][j][p] * ez[11]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[11] * ux[i][j][p] + ey[11] * uy[i][j][p] + ez[11] * uz[i][j][p])*(ex[11] * ux[i][j][p] + ey[11] * uy[i][j][p] + ez[11] * uz[i][j][p]) / 2.0) / 36.0;
				heq12[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[12] + uy[i][j][p] * ey[12] + uz[i][j][p] * ez[12]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[12] * ux[i][j][p] + ey[12] * uy[i][j][p] + ez[12] * uz[i][j][p])*(ex[12] * ux[i][j][p] + ey[12] * uy[i][j][p] + ez[12] * uz[i][j][p]) / 2.0) / 36.0;
				heq13[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[13] + uy[i][j][p] * ey[13] + uz[i][j][p] * ez[13]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[13] * ux[i][j][p] + ey[13] * uy[i][j][p] + ez[13] * uz[i][j][p])*(ex[13] * ux[i][j][p] + ey[13] * uy[i][j][p] + ez[13] * uz[i][j][p]) / 2.0) / 36.0;
				heq14[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[14] + uy[i][j][p] * ey[14] + uz[i][j][p] * ez[14]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[14] * ux[i][j][p] + ey[14] * uy[i][j][p] + ez[14] * uz[i][j][p])*(ex[14] * ux[i][j][p] + ey[14] * uy[i][j][p] + ez[14] * uz[i][j][p]) / 2.0) / 36.0;
				heq15[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[15] + uy[i][j][p] * ey[15] + uz[i][j][p] * ez[15]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[15] * ux[i][j][p] + ey[15] * uy[i][j][p] + ez[15] * uz[i][j][p])*(ex[15] * ux[i][j][p] + ey[15] * uy[i][j][p] + ez[15] * uz[i][j][p]) / 2.0) / 36.0;
				heq16[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[16] + uy[i][j][p] * ey[16] + uz[i][j][p] * ez[16]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[16] * ux[i][j][p] + ey[16] * uy[i][j][p] + ez[16] * uz[i][j][p])*(ex[16] * ux[i][j][p] + ey[16] * uy[i][j][p] + ez[16] * uz[i][j][p]) / 2.0) / 36.0;
				heq17[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[17] + uy[i][j][p] * ey[17] + uz[i][j][p] * ez[17]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[17] * ux[i][j][p] + ey[17] * uy[i][j][p] + ez[17] * uz[i][j][p])*(ex[17] * ux[i][j][p] + ey[17] * uy[i][j][p] + ez[17] * uz[i][j][p]) / 2.0) / 36.0;
				heq18[i][j][p] = hrho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[18] + uy[i][j][p] * ey[18] + uz[i][j][p] * ez[18]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[18] * ux[i][j][p] + ey[18] * uy[i][j][p] + ez[18] * uz[i][j][p])*(ex[18] * ux[i][j][p] + ey[18] * uy[i][j][p] + ez[18] * uz[i][j][p]) / 2.0) / 36.0;

				peq7[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[7] + uy[i][j][p] * ey[7] + uz[i][j][p] * ez[7]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[7] * ux[i][j][p] + ey[7] * uy[i][j][p] + ez[7] * uz[i][j][p])*(ex[7] * ux[i][j][p] + ey[7] * uy[i][j][p] + ez[7] * uz[i][j][p]) / 2.0) / 36.0;
				peq8[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[8] + uy[i][j][p] * ey[8] + uz[i][j][p] * ez[8]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[8] * ux[i][j][p] + ey[8] * uy[i][j][p] + ez[8] * uz[i][j][p])*(ex[8] * ux[i][j][p] + ey[8] * uy[i][j][p] + ez[8] * uz[i][j][p]) / 2.0) / 36.0;
				peq9[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[9] + uy[i][j][p] * ey[9] + uz[i][j][p] * ez[9]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[9] * ux[i][j][p] + ey[9] * uy[i][j][p] + ez[9] * uz[i][j][p])*(ex[9] * ux[i][j][p] + ey[9] * uy[i][j][p] + ez[9] * uz[i][j][p]) / 2.0) / 36.0;
				peq10[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[10] + uy[i][j][p] * ey[10] + uz[i][j][p] * ez[10]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[10] * ux[i][j][p] + ey[10] * uy[i][j][p] + ez[10] * uz[i][j][p])*(ex[10] * ux[i][j][p] + ey[10] * uy[i][j][p] + ez[10] * uz[i][j][p]) / 2.0) / 36.0;
				peq11[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[11] + uy[i][j][p] * ey[11] + uz[i][j][p] * ez[11]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[11] * ux[i][j][p] + ey[11] * uy[i][j][p] + ez[11] * uz[i][j][p])*(ex[11] * ux[i][j][p] + ey[11] * uy[i][j][p] + ez[11] * uz[i][j][p]) / 2.0) / 36.0;
				peq12[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[12] + uy[i][j][p] * ey[12] + uz[i][j][p] * ez[12]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[12] * ux[i][j][p] + ey[12] * uy[i][j][p] + ez[12] * uz[i][j][p])*(ex[12] * ux[i][j][p] + ey[12] * uy[i][j][p] + ez[12] * uz[i][j][p]) / 2.0) / 36.0;
				peq13[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[13] + uy[i][j][p] * ey[13] + uz[i][j][p] * ez[13]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[13] * ux[i][j][p] + ey[13] * uy[i][j][p] + ez[13] * uz[i][j][p])*(ex[13] * ux[i][j][p] + ey[13] * uy[i][j][p] + ez[13] * uz[i][j][p]) / 2.0) / 36.0;
				peq14[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[14] + uy[i][j][p] * ey[14] + uz[i][j][p] * ez[14]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[14] * ux[i][j][p] + ey[14] * uy[i][j][p] + ez[14] * uz[i][j][p])*(ex[14] * ux[i][j][p] + ey[14] * uy[i][j][p] + ez[14] * uz[i][j][p]) / 2.0) / 36.0;
				peq15[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[15] + uy[i][j][p] * ey[15] + uz[i][j][p] * ez[15]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[15] * ux[i][j][p] + ey[15] * uy[i][j][p] + ez[15] * uz[i][j][p])*(ex[15] * ux[i][j][p] + ey[15] * uy[i][j][p] + ez[15] * uz[i][j][p]) / 2.0) / 36.0;
				peq16[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[16] + uy[i][j][p] * ey[16] + uz[i][j][p] * ez[16]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[16] * ux[i][j][p] + ey[16] * uy[i][j][p] + ez[16] * uz[i][j][p])*(ex[16] * ux[i][j][p] + ey[16] * uy[i][j][p] + ez[16] * uz[i][j][p]) / 2.0) / 36.0;
				peq17[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[17] + uy[i][j][p] * ey[17] + uz[i][j][p] * ez[17]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[17] * ux[i][j][p] + ey[17] * uy[i][j][p] + ez[17] * uz[i][j][p])*(ex[17] * ux[i][j][p] + ey[17] * uy[i][j][p] + ez[17] * uz[i][j][p]) / 2.0) / 36.0;
				peq18[i][j][p] = prho[i][j][p] * (1.0 + 3.0*(ux[i][j][p] * ex[18] + uy[i][j][p] * ey[18] + uz[i][j][p] * ez[18]) - 3.0*(ux[i][j][p] * ux[i][j][p] + uy[i][j][p] * uy[i][j][p] + uz[i][j][p] * uz[i][j][p]) / 2.0 + 9.0*(ex[18] * ux[i][j][p] + ey[18] * uy[i][j][p] + ez[18] * uz[i][j][p])*(ex[18] * ux[i][j][p] + ey[18] * uy[i][j][p] + ez[18] * uz[i][j][p]) / 2.0) / 36.0;
			}
		}
	}
	//Mixing Channel
	#pragma omp parallel for collapse(3)
	for (i = 0; i <= mnx + 1; i++)
	{
		for (j = 0; j <= mny + 1; j++)
		{
			for (p = 0; p <= nz + 1; p++)
			{
				fmeq7[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[7] + muy[i][j][p] * ey[7] + muz[i][j][p] * ez[7]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[7] * mux[i][j][p] + ey[7] * muy[i][j][p] + ez[7] * muz[i][j][p])*(ex[7] * mux[i][j][p] + ey[7] * muy[i][j][p] + ez[7] * muz[i][j][p]) / 2.0) / 36.0;
				fmeq8[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[8] + muy[i][j][p] * ey[8] + muz[i][j][p] * ez[8]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[8] * mux[i][j][p] + ey[8] * muy[i][j][p] + ez[8] * muz[i][j][p])*(ex[8] * mux[i][j][p] + ey[8] * muy[i][j][p] + ez[8] * muz[i][j][p]) / 2.0) / 36.0;
				fmeq9[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[9] + muy[i][j][p] * ey[9] + muz[i][j][p] * ez[9]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[9] * mux[i][j][p] + ey[9] * muy[i][j][p] + ez[9] * muz[i][j][p])*(ex[9] * mux[i][j][p] + ey[9] * muy[i][j][p] + ez[9] * muz[i][j][p]) / 2.0) / 36.0;
				fmeq10[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[10] + muy[i][j][p] * ey[10] + muz[i][j][p] * ez[10]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[10] * mux[i][j][p] + ey[10] * muy[i][j][p] + ez[10] * muz[i][j][p])*(ex[10] * mux[i][j][p] + ey[10] * muy[i][j][p] + ez[10] * muz[i][j][p]) / 2.0) / 36.0;
				fmeq11[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[11] + muy[i][j][p] * ey[11] + muz[i][j][p] * ez[11]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[11] * mux[i][j][p] + ey[11] * muy[i][j][p] + ez[11] * muz[i][j][p])*(ex[11] * mux[i][j][p] + ey[11] * muy[i][j][p] + ez[11] * muz[i][j][p]) / 2.0) / 36.0;
				fmeq12[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[12] + muy[i][j][p] * ey[12] + muz[i][j][p] * ez[12]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[12] * mux[i][j][p] + ey[12] * muy[i][j][p] + ez[12] * muz[i][j][p])*(ex[12] * mux[i][j][p] + ey[12] * muy[i][j][p] + ez[12] * muz[i][j][p]) / 2.0) / 36.0;
				fmeq13[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[13] + muy[i][j][p] * ey[13] + muz[i][j][p] * ez[13]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[13] * mux[i][j][p] + ey[13] * muy[i][j][p] + ez[13] * muz[i][j][p])*(ex[13] * mux[i][j][p] + ey[13] * muy[i][j][p] + ez[13] * muz[i][j][p]) / 2.0) / 36.0;
				fmeq14[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[14] + muy[i][j][p] * ey[14] + muz[i][j][p] * ez[14]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[14] * mux[i][j][p] + ey[14] * muy[i][j][p] + ez[14] * muz[i][j][p])*(ex[14] * mux[i][j][p] + ey[14] * muy[i][j][p] + ez[14] * muz[i][j][p]) / 2.0) / 36.0;
				fmeq15[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[15] + muy[i][j][p] * ey[15] + muz[i][j][p] * ez[15]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[15] * mux[i][j][p] + ey[15] * muy[i][j][p] + ez[15] * muz[i][j][p])*(ex[15] * mux[i][j][p] + ey[15] * muy[i][j][p] + ez[15] * muz[i][j][p]) / 2.0) / 36.0;
				fmeq16[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[16] + muy[i][j][p] * ey[16] + muz[i][j][p] * ez[16]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[16] * mux[i][j][p] + ey[16] * muy[i][j][p] + ez[16] * muz[i][j][p])*(ex[16] * mux[i][j][p] + ey[16] * muy[i][j][p] + ez[16] * muz[i][j][p]) / 2.0) / 36.0;
				fmeq17[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[17] + muy[i][j][p] * ey[17] + muz[i][j][p] * ez[17]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[17] * mux[i][j][p] + ey[17] * muy[i][j][p] + ez[17] * muz[i][j][p])*(ex[17] * mux[i][j][p] + ey[17] * muy[i][j][p] + ez[17] * muz[i][j][p]) / 2.0) / 36.0;
				fmeq18[i][j][p] = mrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[18] + muy[i][j][p] * ey[18] + muz[i][j][p] * ez[18]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[18] * mux[i][j][p] + ey[18] * muy[i][j][p] + ez[18] * muz[i][j][p])*(ex[18] * mux[i][j][p] + ey[18] * muy[i][j][p] + ez[18] * muz[i][j][p]) / 2.0) / 36.0;

				gmeq7[i][j][p] = gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[7] + muy[i][j][p] * ey[7] + muz[i][j][p] * ez[7]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[7] * mux[i][j][p] + ey[7] * muy[i][j][p] + ez[7] * muz[i][j][p])*(ex[7] * mux[i][j][p] + ey[7] * muy[i][j][p] + ez[7] * muz[i][j][p]) / 2.0) / 36.0;
				gmeq8[i][j][p] = gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[8] + muy[i][j][p] * ey[8] + muz[i][j][p] * ez[8]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[8] * mux[i][j][p] + ey[8] * muy[i][j][p] + ez[8] * muz[i][j][p])*(ex[8] * mux[i][j][p] + ey[8] * muy[i][j][p] + ez[8] * muz[i][j][p]) / 2.0) / 36.0;
				gmeq9[i][j][p] = gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[9] + muy[i][j][p] * ey[9] + muz[i][j][p] * ez[9]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[9] * mux[i][j][p] + ey[9] * muy[i][j][p] + ez[9] * muz[i][j][p])*(ex[9] * mux[i][j][p] + ey[9] * muy[i][j][p] + ez[9] * muz[i][j][p]) / 2.0) / 36.0;
				gmeq10[i][j][p] =gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[10] + muy[i][j][p] * ey[10] + muz[i][j][p] * ez[10]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[10] * mux[i][j][p] + ey[10] * muy[i][j][p] + ez[10] * muz[i][j][p])*(ex[10] * mux[i][j][p] + ey[10] * muy[i][j][p] + ez[10] * muz[i][j][p]) / 2.0) / 36.0;
				gmeq11[i][j][p] =gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[11] + muy[i][j][p] * ey[11] + muz[i][j][p] * ez[11]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[11] * mux[i][j][p] + ey[11] * muy[i][j][p] + ez[11] * muz[i][j][p])*(ex[11] * mux[i][j][p] + ey[11] * muy[i][j][p] + ez[11] * muz[i][j][p]) / 2.0) / 36.0;
				gmeq12[i][j][p] =gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[12] + muy[i][j][p] * ey[12] + muz[i][j][p] * ez[12]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[12] * mux[i][j][p] + ey[12] * muy[i][j][p] + ez[12] * muz[i][j][p])*(ex[12] * mux[i][j][p] + ey[12] * muy[i][j][p] + ez[12] * muz[i][j][p]) / 2.0) / 36.0;
				gmeq13[i][j][p] =gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[13] + muy[i][j][p] * ey[13] + muz[i][j][p] * ez[13]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[13] * mux[i][j][p] + ey[13] * muy[i][j][p] + ez[13] * muz[i][j][p])*(ex[13] * mux[i][j][p] + ey[13] * muy[i][j][p] + ez[13] * muz[i][j][p]) / 2.0) / 36.0;
				gmeq14[i][j][p] =gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[14] + muy[i][j][p] * ey[14] + muz[i][j][p] * ez[14]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[14] * mux[i][j][p] + ey[14] * muy[i][j][p] + ez[14] * muz[i][j][p])*(ex[14] * mux[i][j][p] + ey[14] * muy[i][j][p] + ez[14] * muz[i][j][p]) / 2.0) / 36.0;
				gmeq15[i][j][p] =gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[15] + muy[i][j][p] * ey[15] + muz[i][j][p] * ez[15]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[15] * mux[i][j][p] + ey[15] * muy[i][j][p] + ez[15] * muz[i][j][p])*(ex[15] * mux[i][j][p] + ey[15] * muy[i][j][p] + ez[15] * muz[i][j][p]) / 2.0) / 36.0;
				gmeq16[i][j][p] =gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[16] + muy[i][j][p] * ey[16] + muz[i][j][p] * ez[16]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[16] * mux[i][j][p] + ey[16] * muy[i][j][p] + ez[16] * muz[i][j][p])*(ex[16] * mux[i][j][p] + ey[16] * muy[i][j][p] + ez[16] * muz[i][j][p]) / 2.0) / 36.0;
				gmeq17[i][j][p] =gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[17] + muy[i][j][p] * ey[17] + muz[i][j][p] * ez[17]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[17] * mux[i][j][p] + ey[17] * muy[i][j][p] + ez[17] * muz[i][j][p])*(ex[17] * mux[i][j][p] + ey[17] * muy[i][j][p] + ez[17] * muz[i][j][p]) / 2.0) / 36.0;
				gmeq18[i][j][p] =gmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[18] + muy[i][j][p] * ey[18] + muz[i][j][p] * ez[18]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[18] * mux[i][j][p] + ey[18] * muy[i][j][p] + ez[18] * muz[i][j][p])*(ex[18] * mux[i][j][p] + ey[18] * muy[i][j][p] + ez[18] * muz[i][j][p]) / 2.0) / 36.0;

				hmeq7[i][j][p] = hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[7] + muy[i][j][p] * ey[7] + muz[i][j][p] * ez[7]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[7] * mux[i][j][p] + ey[7] * muy[i][j][p] + ez[7] * muz[i][j][p])*(ex[7] * mux[i][j][p] + ey[7] * muy[i][j][p] + ez[7] * muz[i][j][p]) / 2.0) / 36.0;
				hmeq8[i][j][p] = hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[8] + muy[i][j][p] * ey[8] + muz[i][j][p] * ez[8]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[8] * mux[i][j][p] + ey[8] * muy[i][j][p] + ez[8] * muz[i][j][p])*(ex[8] * mux[i][j][p] + ey[8] * muy[i][j][p] + ez[8] * muz[i][j][p]) / 2.0) / 36.0;
				hmeq9[i][j][p] = hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[9] + muy[i][j][p] * ey[9] + muz[i][j][p] * ez[9]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[9] * mux[i][j][p] + ey[9] * muy[i][j][p] + ez[9] * muz[i][j][p])*(ex[9] * mux[i][j][p] + ey[9] * muy[i][j][p] + ez[9] * muz[i][j][p]) / 2.0) / 36.0;
				hmeq10[i][j][p] =hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[10] + muy[i][j][p] * ey[10] + muz[i][j][p] * ez[10]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[10] * mux[i][j][p] + ey[10] * muy[i][j][p] + ez[10] * muz[i][j][p])*(ex[10] * mux[i][j][p] + ey[10] * muy[i][j][p] + ez[10] * muz[i][j][p]) / 2.0) / 36.0;
				hmeq11[i][j][p] =hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[11] + muy[i][j][p] * ey[11] + muz[i][j][p] * ez[11]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[11] * mux[i][j][p] + ey[11] * muy[i][j][p] + ez[11] * muz[i][j][p])*(ex[11] * mux[i][j][p] + ey[11] * muy[i][j][p] + ez[11] * muz[i][j][p]) / 2.0) / 36.0;
				hmeq12[i][j][p] =hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[12] + muy[i][j][p] * ey[12] + muz[i][j][p] * ez[12]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[12] * mux[i][j][p] + ey[12] * muy[i][j][p] + ez[12] * muz[i][j][p])*(ex[12] * mux[i][j][p] + ey[12] * muy[i][j][p] + ez[12] * muz[i][j][p]) / 2.0) / 36.0;
				hmeq13[i][j][p] =hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[13] + muy[i][j][p] * ey[13] + muz[i][j][p] * ez[13]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[13] * mux[i][j][p] + ey[13] * muy[i][j][p] + ez[13] * muz[i][j][p])*(ex[13] * mux[i][j][p] + ey[13] * muy[i][j][p] + ez[13] * muz[i][j][p]) / 2.0) / 36.0;
				hmeq14[i][j][p] =hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[14] + muy[i][j][p] * ey[14] + muz[i][j][p] * ez[14]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[14] * mux[i][j][p] + ey[14] * muy[i][j][p] + ez[14] * muz[i][j][p])*(ex[14] * mux[i][j][p] + ey[14] * muy[i][j][p] + ez[14] * muz[i][j][p]) / 2.0) / 36.0;
				hmeq15[i][j][p] =hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[15] + muy[i][j][p] * ey[15] + muz[i][j][p] * ez[15]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[15] * mux[i][j][p] + ey[15] * muy[i][j][p] + ez[15] * muz[i][j][p])*(ex[15] * mux[i][j][p] + ey[15] * muy[i][j][p] + ez[15] * muz[i][j][p]) / 2.0) / 36.0;
				hmeq16[i][j][p] =hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[16] + muy[i][j][p] * ey[16] + muz[i][j][p] * ez[16]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[16] * mux[i][j][p] + ey[16] * muy[i][j][p] + ez[16] * muz[i][j][p])*(ex[16] * mux[i][j][p] + ey[16] * muy[i][j][p] + ez[16] * muz[i][j][p]) / 2.0) / 36.0;
				hmeq17[i][j][p] =hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[17] + muy[i][j][p] * ey[17] + muz[i][j][p] * ez[17]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[17] * mux[i][j][p] + ey[17] * muy[i][j][p] + ez[17] * muz[i][j][p])*(ex[17] * mux[i][j][p] + ey[17] * muy[i][j][p] + ez[17] * muz[i][j][p]) / 2.0) / 36.0;
				hmeq18[i][j][p] =hmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[18] + muy[i][j][p] * ey[18] + muz[i][j][p] * ez[18]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[18] * mux[i][j][p] + ey[18] * muy[i][j][p] + ez[18] * muz[i][j][p])*(ex[18] * mux[i][j][p] + ey[18] * muy[i][j][p] + ez[18] * muz[i][j][p]) / 2.0) / 36.0;

				pmeq7[i][j][p] = pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[7] + muy[i][j][p] * ey[7] + muz[i][j][p] * ez[7]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[7] * mux[i][j][p] + ey[7] * muy[i][j][p] + ez[7] * muz[i][j][p])*(ex[7] * mux[i][j][p] + ey[7] * muy[i][j][p] + ez[7] * muz[i][j][p]) / 2.0) / 36.0;
				pmeq8[i][j][p] = pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[8] + muy[i][j][p] * ey[8] + muz[i][j][p] * ez[8]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[8] * mux[i][j][p] + ey[8] * muy[i][j][p] + ez[8] * muz[i][j][p])*(ex[8] * mux[i][j][p] + ey[8] * muy[i][j][p] + ez[8] * muz[i][j][p]) / 2.0) / 36.0;
				pmeq9[i][j][p] = pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[9] + muy[i][j][p] * ey[9] + muz[i][j][p] * ez[9]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[9] * mux[i][j][p] + ey[9] * muy[i][j][p] + ez[9] * muz[i][j][p])*(ex[9] * mux[i][j][p] + ey[9] * muy[i][j][p] + ez[9] * muz[i][j][p]) / 2.0) / 36.0;
				pmeq10[i][j][p] =pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[10] + muy[i][j][p] * ey[10] + muz[i][j][p] * ez[10]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[10] * mux[i][j][p] + ey[10] * muy[i][j][p] + ez[10] * muz[i][j][p])*(ex[10] * mux[i][j][p] + ey[10] * muy[i][j][p] + ez[10] * muz[i][j][p]) / 2.0) / 36.0;
				pmeq11[i][j][p] =pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[11] + muy[i][j][p] * ey[11] + muz[i][j][p] * ez[11]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[11] * mux[i][j][p] + ey[11] * muy[i][j][p] + ez[11] * muz[i][j][p])*(ex[11] * mux[i][j][p] + ey[11] * muy[i][j][p] + ez[11] * muz[i][j][p]) / 2.0) / 36.0;
				pmeq12[i][j][p] =pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[12] + muy[i][j][p] * ey[12] + muz[i][j][p] * ez[12]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[12] * mux[i][j][p] + ey[12] * muy[i][j][p] + ez[12] * muz[i][j][p])*(ex[12] * mux[i][j][p] + ey[12] * muy[i][j][p] + ez[12] * muz[i][j][p]) / 2.0) / 36.0;
				pmeq13[i][j][p] =pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[13] + muy[i][j][p] * ey[13] + muz[i][j][p] * ez[13]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[13] * mux[i][j][p] + ey[13] * muy[i][j][p] + ez[13] * muz[i][j][p])*(ex[13] * mux[i][j][p] + ey[13] * muy[i][j][p] + ez[13] * muz[i][j][p]) / 2.0) / 36.0;
				pmeq14[i][j][p] =pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[14] + muy[i][j][p] * ey[14] + muz[i][j][p] * ez[14]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[14] * mux[i][j][p] + ey[14] * muy[i][j][p] + ez[14] * muz[i][j][p])*(ex[14] * mux[i][j][p] + ey[14] * muy[i][j][p] + ez[14] * muz[i][j][p]) / 2.0) / 36.0;
				pmeq15[i][j][p] =pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[15] + muy[i][j][p] * ey[15] + muz[i][j][p] * ez[15]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[15] * mux[i][j][p] + ey[15] * muy[i][j][p] + ez[15] * muz[i][j][p])*(ex[15] * mux[i][j][p] + ey[15] * muy[i][j][p] + ez[15] * muz[i][j][p]) / 2.0) / 36.0;
				pmeq16[i][j][p] =pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[16] + muy[i][j][p] * ey[16] + muz[i][j][p] * ez[16]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[16] * mux[i][j][p] + ey[16] * muy[i][j][p] + ez[16] * muz[i][j][p])*(ex[16] * mux[i][j][p] + ey[16] * muy[i][j][p] + ez[16] * muz[i][j][p]) / 2.0) / 36.0;
				pmeq17[i][j][p] =pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[17] + muy[i][j][p] * ey[17] + muz[i][j][p] * ez[17]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[17] * mux[i][j][p] + ey[17] * muy[i][j][p] + ez[17] * muz[i][j][p])*(ex[17] * mux[i][j][p] + ey[17] * muy[i][j][p] + ez[17] * muz[i][j][p]) / 2.0) / 36.0;
				pmeq18[i][j][p] =pmrho[i][j][p] * (1.0 + 3.0*(mux[i][j][p] * ex[18] + muy[i][j][p] * ey[18] + muz[i][j][p] * ez[18]) - 3.0*(mux[i][j][p] * mux[i][j][p] + muy[i][j][p] * muy[i][j][p] + muz[i][j][p] * muz[i][j][p]) / 2.0 + 9.0*(ex[18] * mux[i][j][p] + ey[18] * muy[i][j][p] + ez[18] * muz[i][j][p])*(ex[18] * mux[i][j][p] + ey[18] * muy[i][j][p] + ez[18] * muz[i][j][p]) / 2.0) / 36.0;

			}

		}
	}


}
/*----------------------------------------------------------------------------------
Initializing distribution function
----------------------------------------------------------------------------------*/
void initialize_distribution_function()
{
	int i, j, p;
	//Inlet Channel
	#pragma omp parallel for collapse(3)
	for (i = 0; i <= inx + 1; i++)
	{
		for (j = 0; j <= iny + 1; j++)
		{
			for (p = 0; p <= nz + 1; p++)
			{
				f0[i][j][p] = feq0[i][j][p];
				f1[i][j][p] = feq1[i][j][p];
				f2[i][j][p] = feq2[i][j][p];
				f3[i][j][p] = feq3[i][j][p];
				f4[i][j][p] = feq4[i][j][p];
				f5[i][j][p] = feq5[i][j][p];
				f6[i][j][p] = feq6[i][j][p];
				f7[i][j][p] = feq7[i][j][p];
				f8[i][j][p] = feq8[i][j][p];
				f9[i][j][p] = feq9[i][j][p];
				f10[i][j][p] = feq10[i][j][p];
				f11[i][j][p] = feq11[i][j][p];
				f12[i][j][p] = feq12[i][j][p];
				f13[i][j][p] = feq13[i][j][p];
				f14[i][j][p] = feq14[i][j][p];
				f15[i][j][p] = feq15[i][j][p];
				f16[i][j][p] = feq16[i][j][p];
				f17[i][j][p] = feq17[i][j][p];
				f18[i][j][p] = feq18[i][j][p];

				g0[i][j][p] = geq0[i][j][p];
				g1[i][j][p] = geq1[i][j][p];
				g2[i][j][p] = geq2[i][j][p];
				g3[i][j][p] = geq3[i][j][p];
				g4[i][j][p] = geq4[i][j][p];
				g5[i][j][p] = geq5[i][j][p];
				g6[i][j][p] = geq6[i][j][p];
				g7[i][j][p] = geq7[i][j][p];
				g8[i][j][p] = geq8[i][j][p];
				g9[i][j][p] = geq9[i][j][p];
				g10[i][j][p] = geq10[i][j][p];
				g11[i][j][p] = geq11[i][j][p];
				g12[i][j][p] = geq12[i][j][p];
				g13[i][j][p] = geq13[i][j][p];
				g14[i][j][p] = geq14[i][j][p];
				g15[i][j][p] = geq15[i][j][p];
				g16[i][j][p] = geq16[i][j][p];
				g17[i][j][p] = geq17[i][j][p];
				g18[i][j][p] = geq18[i][j][p];

				h0[i][j][p] = heq0[i][j][p];
				h1[i][j][p] = heq1[i][j][p];
				h2[i][j][p] = heq2[i][j][p];
				h3[i][j][p] = heq3[i][j][p];
				h4[i][j][p] = heq4[i][j][p];
				h5[i][j][p] = heq5[i][j][p];
				h6[i][j][p] = heq6[i][j][p];
				h7[i][j][p] = heq7[i][j][p];
				h8[i][j][p] = heq8[i][j][p];
				h9[i][j][p] = heq9[i][j][p];
				h10[i][j][p] = heq10[i][j][p];
				h11[i][j][p] = heq11[i][j][p];
				h12[i][j][p] = heq12[i][j][p];
				h13[i][j][p] = heq13[i][j][p];
				h14[i][j][p] = heq14[i][j][p];
				h15[i][j][p] = heq15[i][j][p];
				h16[i][j][p] = heq16[i][j][p];
				h17[i][j][p] = heq17[i][j][p];
				h18[i][j][p] = heq18[i][j][p];

				p0[i][j][p] = peq0[i][j][p];
				p1[i][j][p] = peq1[i][j][p];
				p2[i][j][p] = peq2[i][j][p];
				p3[i][j][p] = peq3[i][j][p];
				p4[i][j][p] = peq4[i][j][p];
				p5[i][j][p] = peq5[i][j][p];
				p6[i][j][p] = peq6[i][j][p];
				p7[i][j][p] = peq7[i][j][p];
				p8[i][j][p] = peq8[i][j][p];
				p9[i][j][p] = peq9[i][j][p];
				p10[i][j][p] = peq10[i][j][p];
				p11[i][j][p] = peq11[i][j][p];
				p12[i][j][p] = peq12[i][j][p];
				p13[i][j][p] = peq13[i][j][p];
				p14[i][j][p] = peq14[i][j][p];
				p15[i][j][p] = peq15[i][j][p];
				p16[i][j][p] = peq16[i][j][p];
				p17[i][j][p] = peq17[i][j][p];
				p18[i][j][p] = peq18[i][j][p];
			}
		}
	}
	//Mixing Channel
	#pragma omp parallel for collapse(3)
	for (i = 0; i <= mnx + 1; i++)
	{
		for (j = 0; j <= mny + 1; j++)
		{
			for (p = 0; p <= nz + 1; p++)
			{
				fm0[i][j][p] = fmeq0[i][j][p];
				fm1[i][j][p] = fmeq1[i][j][p];
				fm2[i][j][p] = fmeq2[i][j][p];
				fm3[i][j][p] = fmeq3[i][j][p];
				fm4[i][j][p] = fmeq4[i][j][p];
				fm5[i][j][p] = fmeq5[i][j][p];
				fm6[i][j][p] = fmeq6[i][j][p];
				fm7[i][j][p] = fmeq7[i][j][p];
				fm8[i][j][p] = fmeq8[i][j][p];
				fm9[i][j][p] = fmeq9[i][j][p];
				fm10[i][j][p] = fmeq10[i][j][p];
				fm11[i][j][p] = fmeq11[i][j][p];
				fm12[i][j][p] = fmeq12[i][j][p];
				fm13[i][j][p] = fmeq13[i][j][p];
				fm14[i][j][p] = fmeq14[i][j][p];
				fm15[i][j][p] = fmeq15[i][j][p];
				fm16[i][j][p] = fmeq16[i][j][p];
				fm17[i][j][p] = fmeq17[i][j][p];
				fm18[i][j][p] = fmeq18[i][j][p];

				gm0[i][j][p] = gmeq0[i][j][p];
				gm1[i][j][p] = gmeq1[i][j][p];
				gm2[i][j][p] = gmeq2[i][j][p];
				gm3[i][j][p] = gmeq3[i][j][p];
				gm4[i][j][p] = gmeq4[i][j][p];
				gm5[i][j][p] = gmeq5[i][j][p];
				gm6[i][j][p] = gmeq6[i][j][p];
				gm7[i][j][p] = gmeq7[i][j][p];
				gm8[i][j][p] = gmeq8[i][j][p];
				gm9[i][j][p] = gmeq9[i][j][p];
				gm10[i][j][p] = gmeq10[i][j][p];
				gm11[i][j][p] = gmeq11[i][j][p];
				gm12[i][j][p] = gmeq12[i][j][p];
				gm13[i][j][p] = gmeq13[i][j][p];
				gm14[i][j][p] = gmeq14[i][j][p];
				gm15[i][j][p] = gmeq15[i][j][p];
				gm16[i][j][p] = gmeq16[i][j][p];
				gm17[i][j][p] = gmeq17[i][j][p];
				gm18[i][j][p] = gmeq18[i][j][p];

				hm0[i][j][p] = hmeq0[i][j][p];
				hm1[i][j][p] = hmeq1[i][j][p];
				hm2[i][j][p] = hmeq2[i][j][p];
				hm3[i][j][p] = hmeq3[i][j][p];
				hm4[i][j][p] = hmeq4[i][j][p];
				hm5[i][j][p] = hmeq5[i][j][p];
				hm6[i][j][p] = hmeq6[i][j][p];
				hm7[i][j][p] = hmeq7[i][j][p];
				hm8[i][j][p] = hmeq8[i][j][p];
				hm9[i][j][p] = hmeq9[i][j][p];
				hm10[i][j][p] = hmeq10[i][j][p];
				hm11[i][j][p] = hmeq11[i][j][p];
				hm12[i][j][p] = hmeq12[i][j][p];
				hm13[i][j][p] = hmeq13[i][j][p];
				hm14[i][j][p] = hmeq14[i][j][p];
				hm15[i][j][p] = hmeq15[i][j][p];
				hm16[i][j][p] = hmeq16[i][j][p];
				hm17[i][j][p] = hmeq17[i][j][p];
				hm18[i][j][p] = hmeq18[i][j][p];

				pm0[i][j][p] = pmeq0[i][j][p];
				pm1[i][j][p] = pmeq1[i][j][p];
				pm2[i][j][p] = pmeq2[i][j][p];
				pm3[i][j][p] = pmeq3[i][j][p];
				pm4[i][j][p] = pmeq4[i][j][p];
				pm5[i][j][p] = pmeq5[i][j][p];
				pm6[i][j][p] = pmeq6[i][j][p];
				pm7[i][j][p] = pmeq7[i][j][p];
				pm8[i][j][p] = pmeq8[i][j][p];
				pm9[i][j][p] = pmeq9[i][j][p];
				pm10[i][j][p] = pmeq10[i][j][p];
				pm11[i][j][p] = pmeq11[i][j][p];
				pm12[i][j][p] = pmeq12[i][j][p];
				pm13[i][j][p] = pmeq13[i][j][p];
				pm14[i][j][p] = pmeq14[i][j][p];
				pm15[i][j][p] = pmeq15[i][j][p];
				pm16[i][j][p] = pmeq16[i][j][p];
				pm17[i][j][p] = pmeq17[i][j][p];
				pm18[i][j][p] = pmeq18[i][j][p];
			}
		}
	}
}
/*----------------------------------------------------------------------------------
Boundary conditions
----------------------------------------------------------------------------------*/

void boundary_conditions()
{
	int i = 0, j = 0, p = 0;
	double A = 0, B = 0, C = 0;
	int xc, yc;


	//Case 2: H'D' edge in the schematic excluding its nodes..
	#pragma omp parallel for collapse(1) private (A,B,C,yc)
	for (i = 2; i <= ia - 1; i++)
	{
		//		known: f0,f1,f3,f4,f5,f9,f10,f11,f12,f16
		//		unknown: f2,f6,f7,f8,f13,f14,(f15,f17),f18,rho

		B = 0;
		C = 0;
		yc = 1;

		f2[i][yc][nz] = f4[i][yc][nz];
		f6[i][yc][nz] = f5[i][yc][nz];
		f7[i][yc][nz] = f9[i][yc][nz] - (A * 0 + B);
		f8[i][yc][nz] = f10[i][yc][nz] - (-A * 0 + B);
		f13[i][yc][nz] = f11[i][yc][nz] - (-A * 0 - C);
		f14[i][yc][nz] = f12[i][yc][nz] - (A * 0 - C);
		f18[i][yc][nz] = f16[i][yc][nz] - (B - C);

		f15[i][yc][nz] = fn15[i][yc][nz];//undo streaming
		f17[i][yc][nz] = fn17[i][yc][nz];

		f15[i][yc][nz] = (1.0 / 2.0)*(f15[i][yc][nz] + f17[i][yc][nz]);
		f17[i][yc][nz] = f15[i][yc][nz];

		rho[i][yc][nz] = f0[i][yc][nz] + f1[i][yc][nz] + f2[i][yc][nz]
			+ f3[i][yc][nz] + f4[i][yc][nz] + f5[i][yc][nz]
			+ f6[i][yc][nz] + f7[i][yc][nz] + f8[i][yc][nz]
			+ f9[i][yc][nz] + f10[i][yc][nz] + f11[i][yc][nz]
			+ f12[i][yc][nz] + f13[i][yc][nz] + f14[i][yc][nz]
			+ f15[i][yc][nz] + f16[i][yc][nz] + f17[i][yc][nz] + f18[i][yc][nz];

		A = (1.0 / 4.0)*(f1[i][yc][nz] - f3[i][yc][nz]);

		f7[i][yc][nz] = f9[i][yc][nz] - (A + B);
		f8[i][yc][nz] = f10[i][yc][nz] - (-A + B);
		f13[i][yc][nz] = f11[i][yc][nz] - (-A - C);
		f14[i][yc][nz] = f12[i][yc][nz] - (A - C);
	}

	#pragma omp parallel for collapse(1) private (A,B,C,yc)
	for (i = 2; i <= ia - 1; i++)
	{
		//		known: f0,f1,f3,f4,f5,f9,f10,f11,f12,f16
		//		unknown: f2,f6,f7,f8,f13,f14,(f15,f17),f18,rho

		B = 0;
		C = 0;
		yc = 1;

		g2[i][yc][nz] = g4[i][yc][nz];
		g6[i][yc][nz] = g5[i][yc][nz];
		g7[i][yc][nz] = g9[i][yc][nz] - (A * 0 + B);
		g8[i][yc][nz] = g10[i][yc][nz] - (-A * 0 + B);
		g13[i][yc][nz] = g11[i][yc][nz] - (-A * 0 - C);
		g14[i][yc][nz] = g12[i][yc][nz] - (A * 0 - C);
		g18[i][yc][nz] = g16[i][yc][nz] - (B - C);

		g15[i][yc][nz] = gn15[i][yc][nz];//undo streaming
		g17[i][yc][nz] = gn17[i][yc][nz];

		g15[i][yc][nz] = (1.0 / 2.0)*(g15[i][yc][nz] + g17[i][yc][nz]);
		g17[i][yc][nz] = g15[i][yc][nz];

		grho[i][yc][nz] = g0[i][yc][nz] + g1[i][yc][nz] + g2[i][yc][nz]
			+ g3[i][yc][nz] + g4[i][yc][nz] + g5[i][yc][nz]
			+ g6[i][yc][nz] + g7[i][yc][nz] + g8[i][yc][nz]
			+ g9[i][yc][nz] + g10[i][yc][nz] + g11[i][yc][nz]
			+ g12[i][yc][nz] + g13[i][yc][nz] + g14[i][yc][nz]
			+ g15[i][yc][nz] + g16[i][yc][nz] + g17[i][yc][nz] + g18[i][yc][nz];

		A = (1.0 / 4.0)*(g1[i][yc][nz] - g3[i][yc][nz]);

		g7[i][yc][nz] = g9[i][yc][nz] - (A + B);
		g8[i][yc][nz] = g10[i][yc][nz] - (-A + B);
		g13[i][yc][nz] = g11[i][yc][nz] - (-A - C);
		g14[i][yc][nz] = g12[i][yc][nz] - (A - C);
	}

	#pragma omp parallel for collapse(1) private (A,B,C,yc)
	for (i = 2; i <= ia - 1; i++)
	{
		//		known: f0,f1,f3,f4,f5,f9,f10,f11,f12,f16
		//		unknown: f2,f6,f7,f8,f13,f14,(f15,f17),f18,rho

		B = 0;
		C = 0;
		yc = 1;

		h2[i][yc][nz] = h4[i][yc][nz];
		h6[i][yc][nz] = h5[i][yc][nz];
		h7[i][yc][nz] = h9[i][yc][nz] - (A * 0 + B);
		h8[i][yc][nz] = h10[i][yc][nz] - (-A * 0 + B);
		h13[i][yc][nz] = h11[i][yc][nz] - (-A * 0 - C);
		h14[i][yc][nz] = h12[i][yc][nz] - (A * 0 - C);
		h18[i][yc][nz] = h16[i][yc][nz] - (B - C);

		h15[i][yc][nz] = hn15[i][yc][nz];//undo streaminh
		h17[i][yc][nz] = hn17[i][yc][nz];

		h15[i][yc][nz] = (1.0 / 2.0)*(h15[i][yc][nz] + h17[i][yc][nz]);
		h17[i][yc][nz] = h15[i][yc][nz];

		hrho[i][yc][nz] = h0[i][yc][nz] + h1[i][yc][nz] + h2[i][yc][nz]
			+ h3[i][yc][nz] + h4[i][yc][nz] + h5[i][yc][nz]
			+ h6[i][yc][nz] + h7[i][yc][nz] + h8[i][yc][nz]
			+ h9[i][yc][nz] + h10[i][yc][nz] + h11[i][yc][nz]
			+ h12[i][yc][nz] + h13[i][yc][nz] + h14[i][yc][nz]
			+ h15[i][yc][nz] + h16[i][yc][nz] + h17[i][yc][nz] + h18[i][yc][nz];

		A = (1.0 / 4.0)*(h1[i][yc][nz] - h3[i][yc][nz]);

		h7[i][yc][nz] = h9[i][yc][nz] - (A + B);
		h8[i][yc][nz] = h10[i][yc][nz] - (-A + B);
		h13[i][yc][nz] = h11[i][yc][nz] - (-A - C);
		h14[i][yc][nz] = h12[i][yc][nz] - (A - C);
	}

	#pragma omp parallel for collapse(1) private (A,B,C,yc)
	for (i = 2; i <= ia - 1; i++)
	{
		//		known: f0,f1,f3,f4,f5,f9,f10,f11,f12,f16
		//		unknown: f2,f6,f7,f8,f13,f14,(f15,f17),f18,rho

		B = 0;
		C = 0;
		yc = 1;

		p2[i][yc][nz] = p4[i][yc][nz];
		p6[i][yc][nz] = p5[i][yc][nz];
		p7[i][yc][nz] = p9[i][yc][nz] - (A * 0 + B);
		p8[i][yc][nz] = p10[i][yc][nz] - (-A * 0 + B);
		p13[i][yc][nz] = p11[i][yc][nz] - (-A * 0 - C);
		p14[i][yc][nz] = p12[i][yc][nz] - (A * 0 - C);
		p18[i][yc][nz] = p16[i][yc][nz] - (B - C);

		p15[i][yc][nz] = pn15[i][yc][nz];//undo streaminp
		p17[i][yc][nz] = pn17[i][yc][nz];

		p15[i][yc][nz] = (1.0 / 2.0)*(p15[i][yc][nz] + p17[i][yc][nz]);
		p17[i][yc][nz] = p15[i][yc][nz];

		prho[i][yc][nz] = p0[i][yc][nz] + p1[i][yc][nz] + p2[i][yc][nz]
			+ p3[i][yc][nz] + p4[i][yc][nz] + p5[i][yc][nz]
			+ p6[i][yc][nz] + p7[i][yc][nz] + p8[i][yc][nz]
			+ p9[i][yc][nz] + p10[i][yc][nz] + p11[i][yc][nz]
			+ p12[i][yc][nz] + p13[i][yc][nz] + p14[i][yc][nz]
			+ p15[i][yc][nz] + p16[i][yc][nz] + p17[i][yc][nz] + p18[i][yc][nz];

		A = (1.0 / 4.0)*(p1[i][yc][nz] - p3[i][yc][nz]);

		p7[i][yc][nz] = p9[i][yc][nz] - (A + B);
		p8[i][yc][nz] = p10[i][yc][nz] - (-A + B);
		p13[i][yc][nz] = p11[i][yc][nz] - (-A - C);
		p14[i][yc][nz] = p12[i][yc][nz] - (A - C);
	}

	// Case 3: C'E' edge in the schematic excluding its nodes..
	#pragma omp parallel for collapse(1) private (A,B,C,yc)
	for (i = ib + 1; i <= inx - 1; i++)
	{
		//		known: f0,f1,f3,f4,f5,f9,f10,f11,f12,f16
		//		unknown: f2,f6,f7,f8,f13,f14,f15b,f17b,f18,rho

		B = 0;
		C = 0;
		yc = 1;

		f2[i][yc][nz] = f4[i][yc][nz];
		f6[i][yc][nz] = f5[i][yc][nz];
		f7[i][yc][nz] = f9[i][yc][nz] - (A * 0 + B);
		f8[i][yc][nz] = f10[i][yc][nz] - (-A * 0 + B);
		f13[i][yc][nz] = f11[i][yc][nz] - (-A * 0 - C);
		f14[i][yc][nz] = f12[i][yc][nz] - (A * 0 - C);
		f18[i][yc][nz] = f16[i][yc][nz] - (B - C);

		f15[i][yc][nz] = fn15[i][yc][nz];//undo streaming
		f17[i][yc][nz] = fn17[i][yc][nz];

		f15[i][yc][nz] = (1.0 / 2.0)*(f15[i][yc][nz] + f17[i][yc][nz]);
		f17[i][yc][nz] = f15[i][yc][nz];

		rho[i][yc][nz] = f0[i][yc][nz] + f1[i][yc][nz] + f2[i][yc][nz]
			+ f3[i][yc][nz] + f4[i][yc][nz] + f5[i][yc][nz]
			+ f6[i][yc][nz] + f7[i][yc][nz] + f8[i][yc][nz]
			+ f9[i][yc][nz] + f10[i][yc][nz] + f11[i][yc][nz]
			+ f12[i][yc][nz] + f13[i][yc][nz] + f14[i][yc][nz]
			+ f15[i][yc][nz] + f16[i][yc][nz] + f17[i][yc][nz] + f18[i][yc][nz];

		A = (1.0 / 4.0)*(f1[i][yc][nz] - f3[i][yc][nz]);

		f7[i][yc][nz] = f9[i][yc][nz] - (A + B);
		f8[i][yc][nz] = f10[i][yc][nz] - (-A + B);
		f13[i][yc][nz] = f11[i][yc][nz] - (-A - C);
		f14[i][yc][nz] = f12[i][yc][nz] - (A - C);
	}
	#pragma omp parallel for collapse(1) private (A,B,C,yc)
	for (i = ib + 1; i <= inx - 1; i++)
	{
		//		known: f0,f1,f3,f4,f5,f9,f10,f11,f12,f16
		//		unknown: f2,f6,f7,f8,f13,f14,f15b,f17b,f18,rho

		B = 0;
		C = 0;
		yc = 1;

		g2[i][yc][nz] = g4[i][yc][nz];
		g6[i][yc][nz] = g5[i][yc][nz];
		g7[i][yc][nz] = g9[i][yc][nz] - (A * 0 + B);
		g8[i][yc][nz] = g10[i][yc][nz] - (-A * 0 + B);
		g13[i][yc][nz] = g11[i][yc][nz] - (-A * 0 - C);
		g14[i][yc][nz] = g12[i][yc][nz] - (A * 0 - C);
		g18[i][yc][nz] = g16[i][yc][nz] - (B - C);

		g15[i][yc][nz] = gn15[i][yc][nz];//undo streaming
		g17[i][yc][nz] = gn17[i][yc][nz];

		g15[i][yc][nz] = (1.0 / 2.0)*(g15[i][yc][nz] + g17[i][yc][nz]);
		g17[i][yc][nz] = g15[i][yc][nz];

		grho[i][yc][nz] = g0[i][yc][nz] + g1[i][yc][nz] + g2[i][yc][nz]
			+ g3[i][yc][nz] + g4[i][yc][nz] + g5[i][yc][nz]
			+ g6[i][yc][nz] + g7[i][yc][nz] + g8[i][yc][nz]
			+ g9[i][yc][nz] + g10[i][yc][nz] + g11[i][yc][nz]
			+ g12[i][yc][nz] + g13[i][yc][nz] + g14[i][yc][nz]
			+ g15[i][yc][nz] + g16[i][yc][nz] + g17[i][yc][nz] + g18[i][yc][nz];

		A = (1.0 / 4.0)*(g1[i][yc][nz] - g3[i][yc][nz]);

		g7[i][yc][nz] = g9[i][yc][nz] - (A + B);
		g8[i][yc][nz] = g10[i][yc][nz] - (-A + B);
		g13[i][yc][nz] = g11[i][yc][nz] - (-A - C);
		g14[i][yc][nz] = g12[i][yc][nz] - (A - C);
	}
	#pragma omp parallel for collapse(1) private (A,B,C,yc)
	for (i = ib + 1; i <= inx - 1; i++)
	{
		//		known: f0,f1,f3,f4,f5,f9,f10,f11,f12,f16
		//		unknown: f2,f6,f7,f8,f13,f14,f15b,f17b,f18,rho

		B = 0;
		C = 0;
		yc = 1;

		h2[i][yc][nz] = h4[i][yc][nz];
		h6[i][yc][nz] = h5[i][yc][nz];
		h7[i][yc][nz] = h9[i][yc][nz] - (A * 0 + B);
		h8[i][yc][nz] = h10[i][yc][nz] - (-A * 0 + B);
		h13[i][yc][nz] = h11[i][yc][nz] - (-A * 0 - C);
		h14[i][yc][nz] = h12[i][yc][nz] - (A * 0 - C);
		h18[i][yc][nz] = h16[i][yc][nz] - (B - C);

		h15[i][yc][nz] = hn15[i][yc][nz];//undo streaminh
		h17[i][yc][nz] = hn17[i][yc][nz];

		h15[i][yc][nz] = (1.0 / 2.0)*(h15[i][yc][nz] + h17[i][yc][nz]);
		h17[i][yc][nz] = h15[i][yc][nz];

		hrho[i][yc][nz] = h0[i][yc][nz] + h1[i][yc][nz] + h2[i][yc][nz]
			+ h3[i][yc][nz] + h4[i][yc][nz] + h5[i][yc][nz]
			+ h6[i][yc][nz] + h7[i][yc][nz] + h8[i][yc][nz]
			+ h9[i][yc][nz] + h10[i][yc][nz] + h11[i][yc][nz]
			+ h12[i][yc][nz] + h13[i][yc][nz] + h14[i][yc][nz]
			+ h15[i][yc][nz] + h16[i][yc][nz] + h17[i][yc][nz] + h18[i][yc][nz];

		A = (1.0 / 4.0)*(h1[i][yc][nz] - h3[i][yc][nz]);

		h7[i][yc][nz] = h9[i][yc][nz] - (A + B);
		h8[i][yc][nz] = h10[i][yc][nz] - (-A + B);
		h13[i][yc][nz] = h11[i][yc][nz] - (-A - C);
		h14[i][yc][nz] = h12[i][yc][nz] - (A - C);
	}

	#pragma omp parallel for collapse(1) private (A,B,C,yc)
	for (i = ib + 1; i <= inx - 1; i++)
	{
		//		known: f0,f1,f3,f4,f5,f9,f10,f11,f12,f16
		//		unknown: f2,f6,f7,f8,f13,f14,f15b,f17b,f18,rho

		B = 0;
		C = 0;
		yc = 1;

		p2[i][yc][nz] = p4[i][yc][nz];
		p6[i][yc][nz] = p5[i][yc][nz];
		p7[i][yc][nz] = p9[i][yc][nz] - (A * 0 + B);
		p8[i][yc][nz] = p10[i][yc][nz] - (-A * 0 + B);
		p13[i][yc][nz] = p11[i][yc][nz] - (-A * 0 - C);
		p14[i][yc][nz] = p12[i][yc][nz] - (A * 0 - C);
		p18[i][yc][nz] = p16[i][yc][nz] - (B - C);

		p15[i][yc][nz] = pn15[i][yc][nz];//undo streaminp
		p17[i][yc][nz] = pn17[i][yc][nz];

		p15[i][yc][nz] = (1.0 / 2.0)*(p15[i][yc][nz] + p17[i][yc][nz]);
		p17[i][yc][nz] = p15[i][yc][nz];

		prho[i][yc][nz] = p0[i][yc][nz] + p1[i][yc][nz] + p2[i][yc][nz]
			+ p3[i][yc][nz] + p4[i][yc][nz] + p5[i][yc][nz]
			+ p6[i][yc][nz] + p7[i][yc][nz] + p8[i][yc][nz]
			+ p9[i][yc][nz] + p10[i][yc][nz] + p11[i][yc][nz]
			+ p12[i][yc][nz] + p13[i][yc][nz] + p14[i][yc][nz]
			+ p15[i][yc][nz] + p16[i][yc][nz] + p17[i][yc][nz] + p18[i][yc][nz];

		A = (1.0 / 4.0)*(p1[i][yc][nz] - p3[i][yc][nz]);

		p7[i][yc][nz] = p9[i][yc][nz] - (A + B);
		p8[i][yc][nz] = p10[i][yc][nz] - (-A + B);
		p13[i][yc][nz] = p11[i][yc][nz] - (-A - C);
		p14[i][yc][nz] = p12[i][yc][nz] - (A - C);
	}


	//Case 4: G'H' edge shown in schematic (excluding its nodes)
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		//		known: f0,f2,f3,f4,f5,f8,f9,f12,f15,f16
		//		unknown: f1,f6,f7,f10,(f11,f13),f14,f17,f18,rho

		A = 0;
		B = 1.0 / 4.0*(f2[1][j][nz] - f4[1][j][nz]);
		C = 0;

		f1[1][j][nz] = f3[1][j][nz];
		f6[1][j][nz] = f5[1][j][nz];
		f7[1][j][nz] = f9[1][j][nz] - (A + B);
		f10[1][j][nz] = f8[1][j][nz] - (A - B);
		f14[1][j][nz] = f12[1][j][nz] - (A - C);
		f17[1][j][nz] = f15[1][j][nz] - (-B - C);
		f18[1][j][nz] = f16[1][j][nz] - (B - C);

		//Undo Streaming
		f11[1][j][nz] = fn11[1][j][nz];
		f13[1][j][nz] = fn13[1][j][nz];

		f11[1][j][nz] = (1.0 / 2.0)*(rho[1][j][nz - 1]
			- (f0[1][j][nz] + f1[1][j][nz] + f2[1][j][nz]
			+ f3[1][j][nz] + f4[1][j][nz] + f5[1][j][nz]
			+ f6[1][j][nz] + f7[1][j][nz] + f8[1][j][nz]
			+ f9[1][j][nz] + f10[1][j][nz] + f12[1][j][nz]
			+ f14[1][j][nz] + f15[1][j][nz] + f16[1][j][nz]
			+ f17[1][j][nz] + f18[1][j][nz]));
		f13[1][j][nz] = f11[1][j][nz];

	}


	#pragma omp parallel for collapse(1) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		//		known: f0,f2,f3,f4,f5,f8,f9,f12,f15,f16
		//		unknown: f1,f6,f7,f10,(f11,f13),f14,f17,f18,rho

		A = 0;
		B = 1.0 / 4.0*(g2[1][j][nz] - g4[1][j][nz]);
		C = 0;

		g1[1][j][nz] = g3[1][j][nz];
		g6[1][j][nz] = g5[1][j][nz];
		g7[1][j][nz] = g9[1][j][nz] - (A + B);
		g10[1][j][nz] = g8[1][j][nz] - (A - B);
		g14[1][j][nz] = g12[1][j][nz] - (A - C);
		g17[1][j][nz] = g15[1][j][nz] - (-B - C);
		g18[1][j][nz] = g16[1][j][nz] - (B - C);

		//Undo Streaming
		g11[1][j][nz] = gn11[1][j][nz];
		g13[1][j][nz] = gn13[1][j][nz];

		g11[1][j][nz] = (1.0 / 2.0)*(grho[1][j][nz - 1]
			- (g0[1][j][nz] + g1[1][j][nz] + g2[1][j][nz]
			+ g3[1][j][nz] + g4[1][j][nz] + g5[1][j][nz]
			+ g6[1][j][nz] + g7[1][j][nz] + g8[1][j][nz]
			+ g9[1][j][nz] + g10[1][j][nz] + g12[1][j][nz]
			+ g14[1][j][nz] + g15[1][j][nz] + g16[1][j][nz]
			+ g17[1][j][nz] + g18[1][j][nz]));
		g13[1][j][nz] = g11[1][j][nz];

	}

	#pragma omp parallel for collapse(1) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		//		known: f0,f2,f3,f4,f5,f8,f9,f12,f15,f16
		//		unknown: f1,f6,f7,f10,(f11,f13),f14,f17,f18,rho

		A = 0;
		B = 1.0 / 4.0*(h2[1][j][nz] - h4[1][j][nz]);
		C = 0;

		h1[1][j][nz] = h3[1][j][nz];
		h6[1][j][nz] = h5[1][j][nz];
		h7[1][j][nz] = h9[1][j][nz] - (A + B);
		h10[1][j][nz] = h8[1][j][nz] - (A - B);
		h14[1][j][nz] = h12[1][j][nz] - (A - C);
		h17[1][j][nz] = h15[1][j][nz] - (-B - C);
		h18[1][j][nz] = h16[1][j][nz] - (B - C);

		//Undo Streaminh
		h11[1][j][nz] = hn11[1][j][nz];
		h13[1][j][nz] = hn13[1][j][nz];

		h11[1][j][nz] = (1.0 / 2.0)*(hrho[1][j][nz - 1]
			- (h0[1][j][nz] + h1[1][j][nz] + h2[1][j][nz]
			+ h3[1][j][nz] + h4[1][j][nz] + h5[1][j][nz]
			+ h6[1][j][nz] + h7[1][j][nz] + h8[1][j][nz]
			+ h9[1][j][nz] + h10[1][j][nz] + h12[1][j][nz]
			+ h14[1][j][nz] + h15[1][j][nz] + h16[1][j][nz]
			+ h17[1][j][nz] + h18[1][j][nz]));
		h13[1][j][nz] = h11[1][j][nz];

	}

	#pragma omp parallel for collapse(1) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		//		known: f0,f2,f3,f4,f5,f8,f9,f12,f15,f16
		//		unknown: f1,f6,f7,f10,(f11,f13),f14,f17,f18,rho

		A = 0;
		B = 1.0 / 4.0*(p2[1][j][nz] - p4[1][j][nz]);
		C = 0;

		p1[1][j][nz] = p3[1][j][nz];
		p6[1][j][nz] = p5[1][j][nz];
		p7[1][j][nz] = p9[1][j][nz] - (A + B);
		p10[1][j][nz] = p8[1][j][nz] - (A - B);
		p14[1][j][nz] = p12[1][j][nz] - (A - C);
		p17[1][j][nz] = p15[1][j][nz] - (-B - C);
		p18[1][j][nz] = p16[1][j][nz] - (B - C);

		//Undo Streaminp
		p11[1][j][nz] = pn11[1][j][nz];
		p13[1][j][nz] = pn13[1][j][nz];

		p11[1][j][nz] = (1.0 / 2.0)*(prho[1][j][nz - 1]
			- (p0[1][j][nz] + p1[1][j][nz] + p2[1][j][nz]
			+ p3[1][j][nz] + p4[1][j][nz] + p5[1][j][nz]
			+ p6[1][j][nz] + p7[1][j][nz] + p8[1][j][nz]
			+ p9[1][j][nz] + p10[1][j][nz] + p12[1][j][nz]
			+ p14[1][j][nz] + p15[1][j][nz] + p16[1][j][nz]
			+ p17[1][j][nz] + p18[1][j][nz]));
		g13[1][j][nz] = g11[1][j][nz];

	}

	//Case 6: E'F' edge shown in schematic excluding nodes
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		//		known: f0,f1,f2,f4,f5,f7,f10,f11,f15,f16,rho_in2
		//		unknown: f3,f6,f8,f9,f12,f13,f14,f17,f18

		A = 0;//-1.0/6.0*rho_in2*uxwall;
		B = (1.0 / 4.0)*(f2[inx][j][nz] - f4[inx][j][nz]);
		C = 0;

		f3[inx][j][nz] = f1[inx][j][nz];
		f6[inx][j][nz] = f5[inx][j][nz];
		f9[inx][j][nz] = f7[inx][j][nz] - (-A - B);
		f8[inx][j][nz] = f10[inx][j][nz] - (-A + B);
		f13[inx][j][nz] = f11[inx][j][nz] - (-A - C);
		f17[inx][j][nz] = f15[inx][j][nz] - (-B - C);
		f18[inx][j][nz] = f16[inx][j][nz] - (B - C);

		//Undo Streaming
		f12[inx][j][nz] = fn12[inx][j][nz];
		f14[inx][j][nz] = fn14[inx][j][nz];

		f12[inx][j][nz] = (1.0 / 2.0)*(rho[inx][j][nz - 1]
			- (f0[inx][j][nz] + f1[inx][j][nz] + f2[inx][j][nz]
			+ f3[inx][j][nz] + f4[inx][j][nz] + f5[inx][j][nz]
			+ f6[inx][j][nz] + f7[inx][j][nz] + f8[inx][j][nz]
			+ f9[inx][j][nz] + f10[inx][j][nz] + f11[inx][j][nz]
			+ f13[inx][j][nz] + f15[inx][j][nz] + f16[inx][j][nz]
			+ f17[inx][j][nz] + f18[inx][j][nz]));
		f14[inx][j][nz] = f12[inx][j][nz];
	}

	#pragma omp parallel for collapse(1) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		//		known: f0,f1,f2,f4,f5,f7,f10,f11,f15,f16,rho_in2
		//		unknown: f3,f6,f8,f9,f12,f13,f14,f17,f18

		A = 0;//-1.0/6.0*rho_in2*uxwall;
		B = (1.0 / 4.0)*(g2[inx][j][nz] - g4[inx][j][nz]);
		C = 0;

		g3[inx][j][nz] = g1[inx][j][nz];
		g6[inx][j][nz] = g5[inx][j][nz];
		g9[inx][j][nz] = g7[inx][j][nz] - (-A - B);
		g8[inx][j][nz] = g10[inx][j][nz] - (-A + B);
		g13[inx][j][nz] = g11[inx][j][nz] - (-A - C);
		g17[inx][j][nz] = g15[inx][j][nz] - (-B - C);
		g18[inx][j][nz] = g16[inx][j][nz] - (B - C);

		//Undo Streaming
		g12[inx][j][nz] = gn12[inx][j][nz];
		g14[inx][j][nz] = gn14[inx][j][nz];

		g12[inx][j][nz] = (1.0 / 2.0)*(grho[inx][j][nz - 1]
			- (g0[inx][j][nz] + g1[inx][j][nz] + g2[inx][j][nz]
			+ g3[inx][j][nz] + g4[inx][j][nz] + g5[inx][j][nz]
			+ g6[inx][j][nz] + g7[inx][j][nz] + g8[inx][j][nz]
			+ g9[inx][j][nz] + g10[inx][j][nz] + g11[inx][j][nz]
			+ g13[inx][j][nz] + g15[inx][j][nz] + g16[inx][j][nz]
			+ g17[inx][j][nz] + g18[inx][j][nz]));
		g14[inx][j][nz] = g12[inx][j][nz];
	}

	#pragma omp parallel for collapse(1) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		//		known: f0,f1,f2,f4,f5,f7,f10,f11,f15,f16,rho_in2
		//		unknown: f3,f6,f8,f9,f12,f13,f14,f17,f18

		A = 0;//-1.0/6.0*rho_in2*uxwall;
		B = (1.0 / 4.0)*(h2[inx][j][nz] - h4[inx][j][nz]);
		C = 0;

		h3[inx][j][nz] = h1[inx][j][nz];
		h6[inx][j][nz] = h5[inx][j][nz];
		h9[inx][j][nz] = h7[inx][j][nz] - (-A - B);
		h8[inx][j][nz] = h10[inx][j][nz] - (-A + B);
		h13[inx][j][nz] = h11[inx][j][nz] - (-A - C);
		h17[inx][j][nz] = h15[inx][j][nz] - (-B - C);
		h18[inx][j][nz] = h16[inx][j][nz] - (B - C);

		//Undo Streaminh
		h12[inx][j][nz] = hn12[inx][j][nz];
		h14[inx][j][nz] = hn14[inx][j][nz];

		h12[inx][j][nz] = (1.0 / 2.0)*(hrho[inx][j][nz - 1]
			- (h0[inx][j][nz] + h1[inx][j][nz] + h2[inx][j][nz]
			+ h3[inx][j][nz] + h4[inx][j][nz] + h5[inx][j][nz]
			+ h6[inx][j][nz] + h7[inx][j][nz] + h8[inx][j][nz]
			+ h9[inx][j][nz] + h10[inx][j][nz] + h11[inx][j][nz]
			+ h13[inx][j][nz] + h15[inx][j][nz] + h16[inx][j][nz]
			+ h17[inx][j][nz] + h18[inx][j][nz]));
		h14[inx][j][nz] = h12[inx][j][nz];
	}

	#pragma omp parallel for collapse(1) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		//		known: f0,f1,f2,f4,f5,f7,f10,f11,f15,f16,rho_in2
		//		unknown: f3,f6,f8,f9,f12,f13,f14,f17,f18

		A = 0;//-1.0/6.0*rho_in2*uxwall;
		B = (1.0 / 4.0)*(p2[inx][j][nz] - p4[inx][j][nz]);
		C = 0;

		p3[inx][j][nz] = p1[inx][j][nz];
		p6[inx][j][nz] = p5[inx][j][nz];
		p9[inx][j][nz] = p7[inx][j][nz] - (-A - B);
		p8[inx][j][nz] = p10[inx][j][nz] - (-A + B);
		p13[inx][j][nz] = p11[inx][j][nz] - (-A - C);
		p17[inx][j][nz] = p15[inx][j][nz] - (-B - C);
		p18[inx][j][nz] = p16[inx][j][nz] - (B - C);

		//Undo Streaminp
		p12[inx][j][nz] = pn12[inx][j][nz];
		p14[inx][j][nz] = pn14[inx][j][nz];

		p12[inx][j][nz] = (1.0 / 2.0)*(prho[inx][j][nz - 1]
			- (p0[inx][j][nz] + p1[inx][j][nz] + p2[inx][j][nz]
			+ p3[inx][j][nz] + p4[inx][j][nz] + p5[inx][j][nz]
			+ p6[inx][j][nz] + p7[inx][j][nz] + p8[inx][j][nz]
			+ p9[inx][j][nz] + p10[inx][j][nz] + p11[inx][j][nz]
			+ p13[inx][j][nz] + p15[inx][j][nz] + p16[inx][j][nz]
			+ p17[inx][j][nz] + p18[inx][j][nz]));
		p14[inx][j][nz] = p12[inx][j][nz];
	}

	//Inlet1
	#pragma omp parallel for collapse(2) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			rho[1][j][p] = (1.0 / (1.0 - Uin1[j][p]))*((f0[1][j][p] + f2[1][j][p] + f4[1][j][p]
				+ f5[1][j][p] + f6[1][j][p] + f15[1][j][p]
				+ f16[1][j][p] + f17[1][j][p] + f18[1][j][p])
				+ 2.0*(f3[1][j][p] + f8[1][j][p] + f9[1][j][p] + f12[1][j][p] + f13[1][j][p]));
			A = (-rho[1][j][p] * Uin1[j][p]) / 6.0;
			B = (1.0 / 2.0)*(f2[1][j][p] - f4[1][j][p] + f15[1][j][p] - f16[1][j][p] - f17[1][j][p] + f18[1][j][p]); //-Ny
			C = (1.0 / 2.0)*(f5[1][j][p] - f6[1][j][p] + f15[1][j][p] + f16[1][j][p] - f17[1][j][p] - f18[1][j][p]); // -Nz

			f1[1][j][p] = f3[1][j][p] + (rho[1][j][p] * Uin1[j][p]) / 3.0;
			f7[1][j][p] = f9[1][j][p] - (A + B);
			f10[1][j][p] = f8[1][j][p] - (A - B);
			f11[1][j][p] = f13[1][j][p] - (A + C);
			f14[1][j][p] = f12[1][j][p] - (A - C);
		}
	}
	#pragma omp parallel for collapse(2) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
				g1[1][j][p]  = -g3[1][j][p]  + (2.0/18.0)*Ca_in1;
				g7[1][j][p]  = -g9[1][j][p]  + (2.0/36.0)*Ca_in1;
				g10[1][j][p] = -g8[1][j][p]  + (2.0/36.0)*Ca_in1;
				g11[1][j][p] = -g13[1][j][p] + (2.0/36.0)*Ca_in1;
				g14[1][j][p] = -g12[1][j][p] + (2.0/36.0)*Ca_in1;	
		}
	}

		#pragma omp parallel for collapse(2) private (A,B,C)
		for (j = 2; j <= iny - 1; j++)
		{
			for (p = 2; p <= nz - 1; p++)
			{
					h1[1][j][p]  = -h3[1][j][p]  + (2.0/18.0)*Cb_in1;
					h7[1][j][p]  = -h9[1][j][p]  + (2.0/36.0)*Cb_in1;
					h10[1][j][p] = -h8[1][j][p]  + (2.0/36.0)*Cb_in1;
					h11[1][j][p] = -h13[1][j][p] + (2.0/36.0)*Cb_in1;
					h14[1][j][p] = -h12[1][j][p] + (2.0/36.0)*Cb_in1;	
				
			}
		}

		#pragma omp parallel for collapse(2) private (A,B,C)
		for (j = 2; j <= iny - 1; j++)
		{
			for (p = 2; p <= nz - 1; p++)
			{
					p1[1][j][p]  = -p3[1][j][p]  + (2.0/18.0)*Cp_in1;
					p7[1][j][p]  = -p9[1][j][p]  + (2.0/36.0)*Cp_in1;
					p10[1][j][p] = -p8[1][j][p]  + (2.0/36.0)*Cp_in1;
					p11[1][j][p] = -p13[1][j][p] + (2.0/36.0)*Cp_in1;
					p14[1][j][p] = -p12[1][j][p] + (2.0/36.0)*Cp_in1;	
				
			}
		}

	//Inlet2
	#pragma omp parallel for collapse(2) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		for (p = 2; p <= nz - 1; p++)
		{

			rho[inx][j][p] = (1.0 / (1.0 + (Uin2[j][p]))) * ((f0[inx][j][p] + f2[inx][j][p] + f4[inx][j][p]
				+ f5[inx][j][p] + f6[inx][j][p] + f15[inx][j][p]
				+ f16[inx][j][p] + f17[inx][j][p] + f18[inx][j][p])
				+ 2.0*(f1[inx][j][p] + f7[inx][j][p] + f10[inx][j][p] + f11[inx][j][p] + f14[inx][j][p]));

			A = (-rho[inx][j][p] * (Uin2[j][p])) / 6.0;
			B = (1.0 / 2.0)*(f2[inx][j][p] - f4[inx][j][p] + f15[inx][j][p] - f16[inx][j][p] - f17[inx][j][p] + f18[inx][j][p]);//-Ny
			C = (1.0 / 2.0)*(f5[inx][j][p] - f6[inx][j][p] + f15[inx][j][p] + f16[inx][j][p] - f17[inx][j][p] - f18[inx][j][p]);//-Nz

			f3[inx][j][p] = f1[inx][j][p] - (rho[inx][j][p] * (Uin2[j][p])) / 3.0;
			f9[inx][j][p] = f7[inx][j][p] + (A + B);
			f8[inx][j][p] = f10[inx][j][p] + (A - B);
			f13[inx][j][p] = f11[inx][j][p] + (A + C);
			f12[inx][j][p] = f14[inx][j][p] + (A - C);
		}
	}
	#pragma omp parallel for collapse(2) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			g3[inx][j][p]  = -g1[inx][j][p]  + (2.0/18.0)*Ca_in2;
			g9[inx][j][p]  = -g7[inx][j][p]  + (2.0/36.0)*Ca_in2;
			g8[inx][j][p]  = -g10[inx][j][p] + (2.0/36.0)*Ca_in2;
			g13[inx][j][p] = -g11[inx][j][p] + (2.0/36.0)*Ca_in2;
			g12[inx][j][p] = -g14[inx][j][p] + (2.0/36.0)*Ca_in2;
		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			h3[inx][j][p]  = -h1[inx][j][p]  + (2.0/18.0)*Cb_in2;
			h9[inx][j][p]  = -h7[inx][j][p]  + (2.0/36.0)*Cb_in2;
			h8[inx][j][p]  = -h10[inx][j][p] + (2.0/36.0)*Cb_in2;
			h13[inx][j][p] = -h11[inx][j][p] + (2.0/36.0)*Cb_in2;
			h12[inx][j][p] = -h14[inx][j][p] + (2.0/36.0)*Cb_in2;
		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			p3[inx][j][p]  = -p1[inx][j][p]  + (2.0/18.0)*Cp_in2;
			p9[inx][j][p]  = -p7[inx][j][p]  + (2.0/36.0)*Cp_in2;
			p8[inx][j][p]  = -p10[inx][j][p] + (2.0/36.0)*Cp_in2;
			p13[inx][j][p] = -p11[inx][j][p] + (2.0/36.0)*Cp_in2;
			p12[inx][j][p] = -p14[inx][j][p] + (2.0/36.0)*Cp_in2;
		}
	}


	//Case 10 : GFF'G' plane as shown in schematic excluding the edges
	#pragma omp parallel for collapse(2) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//		known: f0,f1,f2,f3,f5,f6,f7,f8,f11,f12,f13,f14,f15,f18
			//		unknown: f4,f9,f10,f16,f17,rho
			B = 0.0;
			C = (1.0 / 2.0)*(f5[i][iny][p] - f6[i][iny][p] + f11[i][iny][p] + f12[i][iny][p] - f13[i][iny][p] - f14[i][iny][p]); //-Nz

			f4[i][iny][p] = f2[i][iny][p];
			f16[i][iny][p] = f18[i][iny][p] - (C - B);
			f17[i][iny][p] = f15[i][iny][p] - (-B - C);

			f9[i][iny][p] = f7[i][iny][p] - (-A * 0 - B); //exclude A  	
			f10[i][iny][p] = f8[i][iny][p] - (A * 0 - B);

			rho[i][iny][p] = f0[i][iny][p] + f1[i][iny][p] + f2[i][iny][p]
				+ f3[i][iny][p] + f4[i][iny][p] + f5[i][iny][p]
				+ f6[i][iny][p] + f7[i][iny][p] + f8[i][iny][p]
				+ f9[i][iny][p] + f10[i][iny][p] + f11[i][iny][p]
				+ f12[i][iny][p] + f13[i][iny][p] + f14[i][iny][p]
				+ f15[i][iny][p] + f16[i][iny][p] + f17[i][iny][p] + f18[i][iny][p];

			A = (1.0 / 2.0)*(f1[i][iny][p] - f3[i][iny][p] + f11[i][iny][p] - f12[i][iny][p] - f13[i][iny][p] + f14[i][iny][p]);
			//Nx

			f9[i][iny][p] = f7[i][iny][p] - (-A - B); //include A  	
			f10[i][iny][p] = f8[i][iny][p] - (A - B);

		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//		known: f0,f1,f2,f3,f5,f6,f7,f8,f11,f12,f13,f14,f15,f18
			//		unknown: f4,f9,f10,f16,f17,rho
			// B = 0.0;
			// C = (1.0 / 2.0)*(g5[i][iny][p] - g6[i][iny][p] + g11[i][iny][p] + g12[i][iny][p] - g13[i][iny][p] - g14[i][iny][p]); //-Nz

			// g4[i][iny][p] = g2[i][iny][p];
			// g16[i][iny][p] = g18[i][iny][p] - (C - B);
			// g17[i][iny][p] = g15[i][iny][p] - (-B - C);

			// g9[i][iny][p] = g7[i][iny][p] - (-A * 0 - B); //exclude A  	
			// g10[i][iny][p] = g8[i][iny][p] - (A * 0 - B);

			// grho[i][iny][p] = g0[i][iny][p] + g1[i][iny][p] + g2[i][iny][p]
			// 	+ g3[i][iny][p] + g4[i][iny][p] + g5[i][iny][p]
			// 	+ g6[i][iny][p] + g7[i][iny][p] + g8[i][iny][p]
			// 	+ g9[i][iny][p] + g10[i][iny][p] + g11[i][iny][p]
			// 	+ g12[i][iny][p] + g13[i][iny][p] + g14[i][iny][p]
			// 	+ g15[i][iny][p] + g16[i][iny][p] + g17[i][iny][p] + g18[i][iny][p];

			// A = (1.0 / 2.0)*(g1[i][iny][p] - g3[i][iny][p] + g11[i][iny][p] - g12[i][iny][p] - g13[i][iny][p] + g14[i][iny][p]);
			// //Nx

			// g9[i][iny][p] = g7[i][iny][p] - (-A - B); //include A  	
			// g10[i][iny][p] = g8[i][iny][p] - (A - B);
                g0[i][iny][p] =  g0[i][iny-1][p] ;
				g1[i][iny][p] =  g1[i][iny-1][p] ;
				g2[i][iny][p] =  g2[i][iny-1][p] ;
			    g3[i][iny][p] =  g3[i][iny-1][p] ;
				g4[i][iny][p] =  g4[i][iny-1][p] ;
				g5[i][iny][p] =  g5[i][iny-1][p] ;
				g6[i][iny][p] =  g6[i][iny-1][p] ;
				g7[i][iny][p] =  g7[i][iny-1][p] ;
				g8[i][iny][p] =  g8[i][iny-1][p] ;
				g9[i][iny][p] =  g9[i][iny-1][p] ;
				g10[i][iny][p] =  g10[i][iny-1][p] ;
				g11[i][iny][p] =  g11[i][iny-1][p] ;
				g12[i][iny][p] =  g12[i][iny-1][p] ;
				g13[i][iny][p] =  g13[i][iny-1][p] ;
				g14[i][iny][p] =  g14[i][iny-1][p] ;
				g15[i][iny][p] =  g15[i][iny-1][p] ;
				g16[i][iny][p] =  g16[i][iny-1][p] ;
				g17[i][iny][p] =  g17[i][iny-1][p] ;
				g18[i][iny][p] =  g18[i][iny-1][p] ;



		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			    h0[i][iny][p] =  h0[i][iny-1][p] ;
				h1[i][iny][p] =  h1[i][iny-1][p] ;
				h2[i][iny][p] =  h2[i][iny-1][p] ;
			    h3[i][iny][p] =  h3[i][iny-1][p] ;
				h4[i][iny][p] =  h4[i][iny-1][p] ;
				h5[i][iny][p] =  h5[i][iny-1][p] ;
				h6[i][iny][p] =  h6[i][iny-1][p] ;
				h7[i][iny][p] =  h7[i][iny-1][p] ;
				h8[i][iny][p] =  h8[i][iny-1][p] ;
				h9[i][iny][p] =  h9[i][iny-1][p] ;
				h10[i][iny][p] =  h10[i][iny-1][p] ;
				h11[i][iny][p] =  h11[i][iny-1][p] ;
				h12[i][iny][p] =  h12[i][iny-1][p] ;
				h13[i][iny][p] =  h13[i][iny-1][p] ;
				h14[i][iny][p] =  h14[i][iny-1][p] ;
				h15[i][iny][p] =  h15[i][iny-1][p] ;
				h16[i][iny][p] =  h16[i][iny-1][p] ;
				h17[i][iny][p] =  h17[i][iny-1][p] ;
				h18[i][iny][p] =  h18[i][iny-1][p] ;
		}
	}

#pragma omp parallel for collapse(2) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			    p0[i][iny][p] =  p0[i][iny-1][p] ;
				p1[i][iny][p] =  p1[i][iny-1][p] ;
				p2[i][iny][p] =  p2[i][iny-1][p] ;
			    p3[i][iny][p] =  p3[i][iny-1][p] ;
				p4[i][iny][p] =  p4[i][iny-1][p] ;
				p5[i][iny][p] =  p5[i][iny-1][p] ;
				p6[i][iny][p] =  p6[i][iny-1][p] ;
				p7[i][iny][p] =  p7[i][iny-1][p] ;
				p8[i][iny][p] =  p8[i][iny-1][p] ;
				p9[i][iny][p] =  p9[i][iny-1][p] ;
				p10[i][iny][p] =  p10[i][iny-1][p] ;
				p11[i][iny][p] =  p11[i][iny-1][p] ;
				p12[i][iny][p] =  p12[i][iny-1][p] ;
				p13[i][iny][p] =  p13[i][iny-1][p] ;
				p14[i][iny][p] =  p14[i][iny-1][p] ;
				p15[i][iny][p] =  p15[i][iny-1][p] ;
				p16[i][iny][p] =  p16[i][iny-1][p] ;
				p17[i][iny][p] =  p17[i][iny-1][p] ;
				p18[i][iny][p] =  p18[i][iny-1][p] ;
		}
	}


	//Case11: HDD'H' plane as shown in schematic excluding the edges
#pragma omp parallel for collapse(2) private (A,B,C,yc)
	for (i = 2; i <= ia - 1; i++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//		known: f0,f1,f3,f4,f9,f10,f16,f17,f5,f6,f11,f12,f13,f14
			//		unknown: f2,f7,f8,f15,f18,rho 

			B = 0.0;
			yc = 1;
			//-Nz = C from notes
			C = (1.0 / 2.0)*(f5[i][yc][p] - f6[i][yc][p] + f11[i][yc][p] + f12[i][yc][p] - f13[i][yc][p] - f14[i][yc][p]);

			f2[i][yc][p] = f4[i][yc][p];
			f15[i][yc][p] = f17[i][yc][p] - (B + C);
			f18[i][yc][p] = f16[i][yc][p] - (B - C);
			f7[i][yc][p] = f9[i][yc][p] - (A * 0 + B); // without A first as it will be cancelled out in rho  		
			f8[i][yc][p] = f10[i][yc][p] - (-A * 0 + B);

			rho[i][yc][p] = f0[i][yc][p] + f1[i][yc][p] + f2[i][yc][p] + f3[i][yc][p]
				+ f4[i][yc][p] + f5[i][yc][p] + f6[i][yc][p] + f7[i][yc][p]
				+ f8[i][yc][p] + f9[i][yc][p] + f10[i][yc][p] + f11[i][yc][p]
				+ f12[i][yc][p] + f13[i][yc][p] + f14[i][yc][p] + f15[i][yc][p]
				+ f16[i][yc][p] + f17[i][yc][p] + f18[i][yc][p];
			//-Nx
			A = (1.0 / 2.0)*(f1[i][yc][p] - f3[i][yc][p] + f11[i][yc][p] - f12[i][yc][p] - f13[i][yc][p] + f14[i][yc][p]);

			f7[i][yc][p] = f9[i][yc][p] - (A + B);  // now include A 	
			f8[i][yc][p] = f10[i][yc][p] - (-A + B);
		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C,yc)
	for (i = 2; i <= ia - 1; i++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//		known: f0,f1,f3,f4,f9,f10,f16,f17,f5,f6,f11,f12,f13,f14
			//		unknown: f2,f7,f8,f15,f18,rho 

			// B = 0.0;
			yc = 1;
			// //-Nz = C from notes
			// C = (1.0 / 2.0)*(g5[i][yc][p] - g6[i][yc][p] + g11[i][yc][p] + g12[i][yc][p] - g13[i][yc][p] - g14[i][yc][p]);

			// g2[i][yc][p] = g4[i][yc][p];
			// g15[i][yc][p] = g17[i][yc][p] - (B + C);
			// g18[i][yc][p] = g16[i][yc][p] - (B - C);
			// g7[i][yc][p] = g9[i][yc][p] - (A * 0 + B); // without A girst as it will be cancelled out in rho  		
			// g8[i][yc][p] = g10[i][yc][p] - (-A * 0 + B);

			// grho[i][yc][p] = g0[i][yc][p] + g1[i][yc][p] + g2[i][yc][p] + g3[i][yc][p]
			// 	+ g4[i][yc][p] + g5[i][yc][p] + g6[i][yc][p] + g7[i][yc][p]
			// 	+ g8[i][yc][p] + g9[i][yc][p] + g10[i][yc][p] + g11[i][yc][p]
			// 	+ g12[i][yc][p] + g13[i][yc][p] + g14[i][yc][p] + g15[i][yc][p]
			// 	+ g16[i][yc][p] + g17[i][yc][p] + g18[i][yc][p];
			// //-Nx
			// A = (1.0 / 2.0)*(g1[i][yc][p] - g3[i][yc][p] + g11[i][yc][p] - g12[i][yc][p] - g13[i][yc][p] + g14[i][yc][p]);

			// g7[i][yc][p] = g9[i][yc][p] - (A + B);  // now include A 	
			// g8[i][yc][p] = g10[i][yc][p] - (-A + B);

			    g0[i][yc][p] =  g0[i][yc+1][p] ;
				g1[i][yc][p] =  g1[i][yc+1][p] ;
				g2[i][yc][p] =  g2[i][yc+1][p] ;
			    g3[i][yc][p] =  g3[i][yc+1][p] ;
				g4[i][yc][p] =  g4[i][yc+1][p] ;
				g5[i][yc][p] =  g5[i][yc+1][p] ;
				g6[i][yc][p] =  g6[i][yc+1][p] ;
				g7[i][yc][p] =  g7[i][yc+1][p] ;
				g8[i][yc][p] =  g8[i][yc+1][p] ;
				g9[i][yc][p] =  g9[i][yc+1][p] ;
				g10[i][yc][p] =  g10[i][yc+1][p] ;
				g11[i][yc][p] =  g11[i][yc+1][p] ;
				g12[i][yc][p] =  g12[i][yc+1][p] ;
				g13[i][yc][p] =  g13[i][yc+1][p] ;
				g14[i][yc][p] =  g14[i][yc+1][p] ;
				g15[i][yc][p] =  g15[i][yc+1][p] ;
				g16[i][yc][p] =  g16[i][yc+1][p] ;
				g17[i][yc][p] =  g17[i][yc+1][p] ;
				g18[i][yc][p] =  g18[i][yc+1][p] ;
		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C,yc)
	for (i = 2; i <= ia - 1; i++)
	{
		for (p = 2; p <= nz - 1; p++)
		{

			yc = 1;
			    h0[i][yc][p] =  h0[i][yc+1][p] ;
				h1[i][yc][p] =  h1[i][yc+1][p] ;
				h2[i][yc][p] =  h2[i][yc+1][p] ;
			    h3[i][yc][p] =  h3[i][yc+1][p] ;
				h4[i][yc][p] =  h4[i][yc+1][p] ;
				h5[i][yc][p] =  h5[i][yc+1][p] ;
				h6[i][yc][p] =  h6[i][yc+1][p] ;
				h7[i][yc][p] =  h7[i][yc+1][p] ;
				h8[i][yc][p] =  h8[i][yc+1][p] ;
				h9[i][yc][p] =  h9[i][yc+1][p] ;
				h10[i][yc][p] =  h10[i][yc+1][p] ;
				h11[i][yc][p] =  h11[i][yc+1][p] ;
				h12[i][yc][p] =  h12[i][yc+1][p] ;
				h13[i][yc][p] =  h13[i][yc+1][p] ;
				h14[i][yc][p] =  h14[i][yc+1][p] ;
				h15[i][yc][p] =  h15[i][yc+1][p] ;
				h16[i][yc][p] =  h16[i][yc+1][p] ;
				h17[i][yc][p] =  h17[i][yc+1][p] ;
				h18[i][yc][p] =  h18[i][yc+1][p] ;
		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C,yc)
	for (i = 2; i <= ia - 1; i++)
	{
		for (p = 2; p <= nz - 1; p++)
		{

			yc = 1;
			    p0[i][yc][p] =  p0[i][yc+1][p] ;
				p1[i][yc][p] =  p1[i][yc+1][p] ;
				p2[i][yc][p] =  p2[i][yc+1][p] ;
			    p3[i][yc][p] =  p3[i][yc+1][p] ;
				p4[i][yc][p] =  p4[i][yc+1][p] ;
				p5[i][yc][p] =  p5[i][yc+1][p] ;
				p6[i][yc][p] =  p6[i][yc+1][p] ;
				p7[i][yc][p] =  p7[i][yc+1][p] ;
				p8[i][yc][p] =  p8[i][yc+1][p] ;
				p9[i][yc][p] =  p9[i][yc+1][p] ;
				p10[i][yc][p] =  p10[i][yc+1][p] ;
				p11[i][yc][p] =  p11[i][yc+1][p] ;
				p12[i][yc][p] =  p12[i][yc+1][p] ;
				p13[i][yc][p] =  p13[i][yc+1][p] ;
				p14[i][yc][p] =  p14[i][yc+1][p] ;
				p15[i][yc][p] =  p15[i][yc+1][p] ;
				p16[i][yc][p] =  p16[i][yc+1][p] ;
				p17[i][yc][p] =  p17[i][yc+1][p] ;
				p18[i][yc][p] =  p18[i][yc+1][p] ;
		}
	}
	// Case12 : CEE'C' plane as shown in schematic excluding edges
	#pragma omp parallel for collapse(2) private (A,B,C,yc)
	for (i = ib + 1; i <= inx - 1; i++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//		known: f0,f1,f3,f4,f9,f10,f16,f17,f5,f6,f11,f12,f13,f14
			//		unknown: f2,f7,f8,f15,f18,rho 

			B = 0.0;
			yc = 1;
			//-Nz = C from notes
			C = 1.0 / 2.0*(f5[i][yc][p] - f6[i][yc][p] + f11[i][yc][p] + f12[i][yc][p] - f13[i][yc][p] - f14[i][yc][p]);

			f2[i][yc][p] = f4[i][yc][p];
			f15[i][yc][p] = f17[i][yc][p] - (B + C);
			f18[i][yc][p] = f16[i][yc][p] - (B - C);
			f7[i][yc][p] = f9[i][yc][p] - (A * 0 + B); // without A first as it will be cancelled out in rho  		
			f8[i][yc][p] = f10[i][yc][p] - (-A * 0 + B);

			rho[i][yc][p] = f0[i][yc][p] + f1[i][yc][p] + f2[i][yc][p]
				+ f3[i][yc][p] + f4[i][yc][p] + f5[i][yc][p]
				+ f6[i][yc][p] + f7[i][yc][p] + f8[i][yc][p]
				+ f9[i][yc][p] + f10[i][yc][p] + f11[i][yc][p]
				+ f12[i][yc][p] + f13[i][yc][p] + f14[i][yc][p]
				+ f15[i][yc][p] + f16[i][yc][p] + f17[i][yc][p] + f18[i][yc][p];
			//-Nx
			A = (1.0 / 2.0)*(f1[i][yc][p] - f3[i][yc][p] + f11[i][yc][p] - f12[i][yc][p] - f13[i][yc][p] + f14[i][yc][p]);

			f7[i][yc][p] = f9[i][yc][p] - (A + B);  // now include A 	
			f8[i][yc][p] = f10[i][yc][p] - (-A + B);
		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C,yc)
	for (i = ib + 1; i <= inx - 1; i++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//		known: f0,f1,f3,f4,f9,f10,f16,f17,f5,f6,f11,f12,f13,f14
			//		unknown: f2,f7,f8,f15,f18,rho 

			// B = 0.0;
			yc = 1;
			// //-Nz = C from notes
			// C = 1.0 / 2.0*(g5[i][yc][p] - g6[i][yc][p] + g11[i][yc][p] + g12[i][yc][p] - g13[i][yc][p] - g14[i][yc][p]);

			// g2[i][yc][p] = g4[i][yc][p];
			// g15[i][yc][p] = g17[i][yc][p] - (B + C);
			// g18[i][yc][p] = g16[i][yc][p] - (B - C);
			// g7[i][yc][p] = g9[i][yc][p] - (A * 0 + B); // without A girst as it will be cancelled out in rho  		
			// g8[i][yc][p] = g10[i][yc][p] - (-A * 0 + B);

			// grho[i][yc][p] = g0[i][yc][p] + g1[i][yc][p] + g2[i][yc][p]
			// 	+ g3[i][yc][p] + g4[i][yc][p] + g5[i][yc][p]
			// 	+ g6[i][yc][p] + g7[i][yc][p] + g8[i][yc][p]
			// 	+ g9[i][yc][p] + g10[i][yc][p] + g11[i][yc][p]
			// 	+ g12[i][yc][p] + g13[i][yc][p] + g14[i][yc][p]
			// 	+ g15[i][yc][p] + g16[i][yc][p] + g17[i][yc][p] + g18[i][yc][p];
			// //-Nx
			// A = (1.0 / 2.0)*(g1[i][yc][p] - g3[i][yc][p] + g11[i][yc][p] - g12[i][yc][p] - g13[i][yc][p] + g14[i][yc][p]);

			// g7[i][yc][p] = g9[i][yc][p] - (A + B);  // now include A 	
			// g8[i][yc][p] = g10[i][yc][p] - (-A + B);
			    g0[i][yc][p] =  g0[i][yc+1][p] ;
				g1[i][yc][p] =  g1[i][yc+1][p] ;
				g2[i][yc][p] =  g2[i][yc+1][p] ;
			    g3[i][yc][p] =  g3[i][yc+1][p] ;
				g4[i][yc][p] =  g4[i][yc+1][p] ;
				g5[i][yc][p] =  g5[i][yc+1][p] ;
				g6[i][yc][p] =  g6[i][yc+1][p] ;
				g7[i][yc][p] =  g7[i][yc+1][p] ;
				g8[i][yc][p] =  g8[i][yc+1][p] ;
				g9[i][yc][p] =  g9[i][yc+1][p] ;
				g10[i][yc][p] =  g10[i][yc+1][p] ;
				g11[i][yc][p] =  g11[i][yc+1][p] ;
				g12[i][yc][p] =  g12[i][yc+1][p] ;
				g13[i][yc][p] =  g13[i][yc+1][p] ;
				g14[i][yc][p] =  g14[i][yc+1][p] ;
				g15[i][yc][p] =  g15[i][yc+1][p] ;
				g16[i][yc][p] =  g16[i][yc+1][p] ;
				g17[i][yc][p] =  g17[i][yc+1][p] ;
				g18[i][yc][p] =  g18[i][yc+1][p] ;
		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C,yc)
	for (i = ib + 1; i <= inx - 1; i++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//		known: f0,f1,f3,f4,f9,f10,f16,f17,f5,f6,f11,f12,f13,f14
			//		unknown: f2,f7,f8,f15,f18,rho 
			yc = 1;
			    h0[i][yc][p] =  h0[i][yc+1][p] ;
				h1[i][yc][p] =  h1[i][yc+1][p] ;
				h2[i][yc][p] =  h2[i][yc+1][p] ;
			    h3[i][yc][p] =  h3[i][yc+1][p] ;
				h4[i][yc][p] =  h4[i][yc+1][p] ;
				h5[i][yc][p] =  h5[i][yc+1][p] ;
				h6[i][yc][p] =  h6[i][yc+1][p] ;
				h7[i][yc][p] =  h7[i][yc+1][p] ;
				h8[i][yc][p] =  h8[i][yc+1][p] ;
				h9[i][yc][p] =  h9[i][yc+1][p] ;
				h10[i][yc][p] =  h10[i][yc+1][p] ;
				h11[i][yc][p] =  h11[i][yc+1][p] ;
				h12[i][yc][p] =  h12[i][yc+1][p] ;
				h13[i][yc][p] =  h13[i][yc+1][p] ;
				h14[i][yc][p] =  h14[i][yc+1][p] ;
				h15[i][yc][p] =  h15[i][yc+1][p] ;
				h16[i][yc][p] =  h16[i][yc+1][p] ;
				h17[i][yc][p] =  h17[i][yc+1][p] ;
				h18[i][yc][p] =  h18[i][yc+1][p] ;
		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C,yc)
	for (i = ib + 1; i <= inx - 1; i++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//		known: f0,f1,f3,f4,f9,f10,f16,f17,f5,f6,f11,f12,f13,f14
			//		unknown: f2,f7,f8,f15,f18,rho 
			yc = 1;
			    p0[i][yc][p] =  p0[i][yc+1][p] ;
				p1[i][yc][p] =  p1[i][yc+1][p] ;
				p2[i][yc][p] =  p2[i][yc+1][p] ;
			    p3[i][yc][p] =  p3[i][yc+1][p] ;
				p4[i][yc][p] =  p4[i][yc+1][p] ;
				p5[i][yc][p] =  p5[i][yc+1][p] ;
				p6[i][yc][p] =  p6[i][yc+1][p] ;
				p7[i][yc][p] =  p7[i][yc+1][p] ;
				p8[i][yc][p] =  p8[i][yc+1][p] ;
				p9[i][yc][p] =  p9[i][yc+1][p] ;
				p10[i][yc][p] =  p10[i][yc+1][p] ;
				p11[i][yc][p] =  p11[i][yc+1][p] ;
				p12[i][yc][p] =  p12[i][yc+1][p] ;
				p13[i][yc][p] =  p13[i][yc+1][p] ;
				p14[i][yc][p] =  p14[i][yc+1][p] ;
				p15[i][yc][p] =  p15[i][yc+1][p] ;
				p16[i][yc][p] =  p16[i][yc+1][p] ;
				p17[i][yc][p] =  p17[i][yc+1][p] ;
				p18[i][yc][p] =  p18[i][yc+1][p] ;
		}
	}

	//Case 13 : Outlet plane ABB'A' excluding edges as shown in schematic
	#pragma omp parallel for collapse(2) private (A,B,C,yc)
	for (i = 2; i <= mnx - 1; i++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//		known: f0,f1,f3,f4,f5,f6,f8,f10,f11,f14,f9,f16,f17,f12,rho_out
			//		unknown: f2,f7,f8,f15,f18

			yc = 1;

			muy[i][yc][p] = 1.0 - (1.0 / rho_out)*((fm0[i][yc][p] + fm1[i][yc][p] + fm3[i][yc][p] + fm5[i][yc][p]
				+ fm6[i][yc][p] + fm11[i][yc][p] + fm12[i][yc][p] + fm13[i][yc][p] + fm14[i][yc][p])
				+ 2.0*(fm4[i][yc][p] + fm9[i][yc][p] + fm10[i][yc][p] + fm16[i][yc][p] + fm17[i][yc][p]));

			A = (rho_out*muy[i][yc][p]) / 6.0;
			//Nx
			B = (1.0 / 2.0)*((fm3[i][yc][p] + fm12[i][yc][p] + fm13[i][yc][p]) - (fm1[i][yc][p] + fm11[i][yc][p] + fm14[i][yc][p])); //Nx
			//Nz
			C = (1.0 / 2.0)*((fm6[i][yc][p] + fm13[i][yc][p] + fm14[i][yc][p]) - (fm5[i][yc][p] + fm11[i][yc][p] + fm12[i][yc][p])); //Nz

			fm2[i][yc][p] = fm4[i][yc][p] + (rho_out*muy[i][yc][p]) / 3.0;

			fm7[i][yc][p] = fm9[i][yc][p] + (A + B);
			fm8[i][yc][p] = fm10[i][yc][p] + (A - B);
			fm15[i][yc][p] = fm17[i][yc][p] + (A + C);
			fm18[i][yc][p] = fm16[i][yc][p] + (A - C);
		}
	}
	#pragma omp parallel for collapse(2) private (A,B,C,yc)
	for (i = 2; i <= mnx - 1; i++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
				yc = 1;
				gm0[i][yc][p] = gm0[i][yc+1][p];
				gm1[i][yc][p] = gm1[i][yc+1][p];
				gm2[i][yc][p] = gm2[i][yc+1][p];
				gm3[i][yc][p] = gm3[i][yc+1][p];
				gm4[i][yc][p] = gm4[i][yc+1][p];
				gm5[i][yc][p] = gm5[i][yc+1][p];
				gm6[i][yc][p] = gm6[i][yc+1][p];
				gm7[i][yc][p] = gm7[i][yc+1][p];
				gm8[i][yc][p] = gm8[i][yc+1][p];
				gm9[i][yc][p] = gm9[i][yc+1][p];
				gm10[i][yc][p] = gm10[i][yc+1][p];
				gm11[i][yc][p] = gm11[i][yc+1][p];
				gm12[i][yc][p] = gm12[i][yc+1][p];
				gm13[i][yc][p] = gm13[i][yc+1][p];
				gm14[i][yc][p] = gm14[i][yc+1][p];
				gm15[i][yc][p] = gm15[i][yc+1][p];
				gm16[i][yc][p] = gm16[i][yc+1][p];
				gm17[i][yc][p] = gm17[i][yc+1][p];
				gm18[i][yc][p] = gm18[i][yc+1][p];
		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C,yc)
	for (i = 2; i <= mnx - 1; i++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
				yc = 1;
				hm0[i][yc][p] = hm0[i][yc+1][p];
				hm1[i][yc][p] = hm1[i][yc+1][p];
				hm2[i][yc][p] = hm2[i][yc+1][p];
				hm3[i][yc][p] = hm3[i][yc+1][p];
				hm4[i][yc][p] = hm4[i][yc+1][p];
				hm5[i][yc][p] = hm5[i][yc+1][p];
				hm6[i][yc][p] = hm6[i][yc+1][p];
				hm7[i][yc][p] = hm7[i][yc+1][p];
				hm8[i][yc][p] = hm8[i][yc+1][p];
				hm9[i][yc][p] = hm9[i][yc+1][p];
				hm10[i][yc][p] = hm10[i][yc+1][p];
				hm11[i][yc][p] = hm11[i][yc+1][p];
				hm12[i][yc][p] = hm12[i][yc+1][p];
				hm13[i][yc][p] = hm13[i][yc+1][p];
				hm14[i][yc][p] = hm14[i][yc+1][p];
				hm15[i][yc][p] = hm15[i][yc+1][p];
				hm16[i][yc][p] = hm16[i][yc+1][p];
				hm17[i][yc][p] = hm17[i][yc+1][p];
				hm18[i][yc][p] = hm18[i][yc+1][p];
		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C,yc)
	for (i = 2; i <= mnx - 1; i++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
				yc = 1;
				pm0[i][yc][p] = pm0[i][yc+1][p];
				pm1[i][yc][p] = pm1[i][yc+1][p];
				pm2[i][yc][p] = pm2[i][yc+1][p];
				pm3[i][yc][p] = pm3[i][yc+1][p];
				pm4[i][yc][p] = pm4[i][yc+1][p];
				pm5[i][yc][p] = pm5[i][yc+1][p];
				pm6[i][yc][p] = pm6[i][yc+1][p];
				pm7[i][yc][p] = pm7[i][yc+1][p];
				pm8[i][yc][p] = pm8[i][yc+1][p];
				pm9[i][yc][p] = pm9[i][yc+1][p];
				pm10[i][yc][p] = pm10[i][yc+1][p];
				pm11[i][yc][p] = pm11[i][yc+1][p];
				pm12[i][yc][p] = pm12[i][yc+1][p];
				pm13[i][yc][p] = pm13[i][yc+1][p];
				pm14[i][yc][p] = pm14[i][yc+1][p];
				pm15[i][yc][p] = pm15[i][yc+1][p];
				pm16[i][yc][p] = pm16[i][yc+1][p];
				pm17[i][yc][p] = pm17[i][yc+1][p];
				pm18[i][yc][p] = pm18[i][yc+1][p];
		}
	}

	//Case14: DAA'D' plane as shown in the schematic excluding  edges
	#pragma omp parallel for collapse(2) private (A,B,C,xc)
	for (j = 2; j <= mny; j++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//Known : f0,f2,f3,f4,f5,f6,f8,f9,f12,f13,f15,f16,f17,f18
			//Unknowns: f1,f7,f10,f11,f14,rho

			B = 0.0;
			xc = 1;
			//Nz = C from notes
			C = (1.0 / 2.0)*((fm6[xc][j][p] + fm17[xc][j][p] + fm18[xc][j][p]) - (fm5[xc][j][p] + fm15[xc][j][p] + fm16[xc][j][p]));

			fm1[xc][j][p] = fm3[xc][j][p];

			fm11[xc][j][p] = fm13[xc][j][p] + (C);
			fm14[xc][j][p] = fm12[xc][j][p] - (C);

			fm7[xc][j][p] = fm9[xc][j][p] - (A * 0 + B); // without A first as it will be cancelled out in rho  		
			fm10[xc][j][p] = fm8[xc][j][p] - (-A * 0 + B);


			mrho[xc][j][p] = fm0[xc][j][p] + fm1[xc][j][p] + fm2[xc][j][p] + fm3[xc][j][p]
				+ fm4[xc][j][p] + fm5[xc][j][p] + fm6[xc][j][p] + fm7[xc][j][p]
				+ fm8[xc][j][p] + fm9[xc][j][p] + fm10[xc][j][p] + fm11[xc][j][p]
				+ fm12[xc][j][p] + fm13[xc][j][p] + fm14[xc][j][p] + fm15[xc][j][p]
				+ fm16[xc][j][p] + fm17[xc][j][p] + fm18[xc][j][p];
			//Nx
			A = (1.0 / 2.0)*((fm2[xc][j][p] + fm15[xc][j][p] + fm18[xc][j][p]) - (fm4[xc][j][p] + fm16[xc][j][p] + fm17[xc][j][p]));


			fm7[xc][j][p] = fm9[xc][j][p] - (A + B);  // now include A 	
			fm10[xc][j][p] = fm8[xc][j][p] + (A + B);
		}
	}
	#pragma omp parallel for collapse(2) private (A,B,C,xc)
	for (j = 2; j <= mny; j++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//Known : f0,f2,f3,f4,f5,f6,f8,f9,f12,f13,f15,f16,f17,f18
			//Unknowns: f1,f7,f10,f11,f14,rho

			//B = 0.0;
			xc = 1;
			
			    gm0[xc][j][p] =  gm0[xc+1][j][p] ;
				gm1[xc][j][p] =  gm1[xc+1][j][p] ;
				gm2[xc][j][p] =  gm2[xc+1][j][p] ;
			    gm3[xc][j][p] =  gm3[xc+1][j][p] ;
				gm4[xc][j][p] =  gm4[xc+1][j][p] ;
				gm5[xc][j][p] =  gm5[xc+1][j][p] ;
				gm6[xc][j][p] =  gm6[xc+1][j][p] ;
				gm7[xc][j][p] =  gm7[xc+1][j][p] ;
				gm8[xc][j][p] =  gm8[xc+1][j][p] ;
				gm9[xc][j][p] =  gm9[xc+1][j][p] ;
				gm10[xc][j][p] =  gm10[xc+1][j][p] ;
				gm11[xc][j][p] =  gm11[xc+1][j][p] ;
				gm12[xc][j][p] =  gm12[xc+1][j][p] ;
				gm13[xc][j][p] =  gm13[xc+1][j][p] ;
				gm14[xc][j][p] =  gm14[xc+1][j][p] ;
				gm15[xc][j][p] =  gm15[xc+1][j][p] ;
				gm16[xc][j][p] =  gm16[xc+1][j][p] ;
				gm17[xc][j][p] =  gm17[xc+1][j][p] ;
				gm18[xc][j][p] =  gm18[xc+1][j][p] ;
		}
	}
	#pragma omp parallel for collapse(2) private (A,B,C,xc)
	for (j = 2; j <= mny; j++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//Known : f0,f2,f3,f4,f5,f6,f8,f9,f12,f13,f15,f16,f17,f18
			//Unknowns: f1,f7,f10,f11,f14,rho

			xc = 1;
			    hm0[xc][j][p] =  hm0[xc+1][j][p] ;
				hm1[xc][j][p] =  hm1[xc+1][j][p] ;
				hm2[xc][j][p] =  hm2[xc+1][j][p] ;
			    hm3[xc][j][p] =  hm3[xc+1][j][p] ;
				hm4[xc][j][p] =  hm4[xc+1][j][p] ;
				hm5[xc][j][p] =  hm5[xc+1][j][p] ;
				hm6[xc][j][p] =  hm6[xc+1][j][p] ;
				hm7[xc][j][p] =  hm7[xc+1][j][p] ;
				hm8[xc][j][p] =  hm8[xc+1][j][p] ;
				hm9[xc][j][p] =  hm9[xc+1][j][p] ;
				hm10[xc][j][p] =  hm10[xc+1][j][p] ;
				hm11[xc][j][p] =  hm11[xc+1][j][p] ;
				hm12[xc][j][p] =  hm12[xc+1][j][p] ;
				hm13[xc][j][p] =  hm13[xc+1][j][p] ;
				hm14[xc][j][p] =  hm14[xc+1][j][p] ;
				hm15[xc][j][p] =  hm15[xc+1][j][p] ;
				hm16[xc][j][p] =  hm16[xc+1][j][p] ;
				hm17[xc][j][p] =  hm17[xc+1][j][p] ;
				hm18[xc][j][p] =  hm18[xc+1][j][p] ;
		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C,xc)
	for (j = 2; j <= mny; j++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//Known : f0,f2,f3,f4,f5,f6,f8,f9,f12,f13,f15,f16,f17,f18
			//Unknowns: f1,f7,f10,f11,f14,rho

			xc = 1;
			    pm0[xc][j][p] =  pm0[xc+1][j][p] ;
				pm1[xc][j][p] =  pm1[xc+1][j][p] ;
				pm2[xc][j][p] =  pm2[xc+1][j][p] ;
			    pm3[xc][j][p] =  pm3[xc+1][j][p] ;
				pm4[xc][j][p] =  pm4[xc+1][j][p] ;
				pm5[xc][j][p] =  pm5[xc+1][j][p] ;
				pm6[xc][j][p] =  pm6[xc+1][j][p] ;
				pm7[xc][j][p] =  pm7[xc+1][j][p] ;
				pm8[xc][j][p] =  pm8[xc+1][j][p] ;
				pm9[xc][j][p] =  pm9[xc+1][j][p] ;
				pm10[xc][j][p] =  pm10[xc+1][j][p] ;
				pm11[xc][j][p] =  pm11[xc+1][j][p] ;
				pm12[xc][j][p] =  pm12[xc+1][j][p] ;
				pm13[xc][j][p] =  pm13[xc+1][j][p] ;
				pm14[xc][j][p] =  pm14[xc+1][j][p] ;
				pm15[xc][j][p] =  pm15[xc+1][j][p] ;
				pm16[xc][j][p] =  pm16[xc+1][j][p] ;
				pm17[xc][j][p] =  pm17[xc+1][j][p] ;
				pm18[xc][j][p] =  pm18[xc+1][j][p] ;
		}
	}

	//Case 15: BCC'B' plane as shown in schematic excluding edges
	#pragma omp parallel for collapse(2) private (A,B,C,xc)
	for (j = 2; j <= mny; j++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//Known : f0,f2,f1,f4,f5,f6,f11,f9,f10,f12,f7,f14,f15,f16,f17,f18
			//Unknowns: f3,f8,f9,f12,f13,rho

			B = 0.0;
			xc = mnx;
			//Nz = C from notes
			C = (1.0 / 2.0)*((fm6[xc][j][p] + fm17[xc][j][p] + fm18[xc][j][p]) - (fm5[xc][j][p] + fm15[xc][j][p] + fm16[xc][j][p]));

			fm3[xc][j][p] = fm1[xc][j][p];

			fm13[xc][j][p] = fm11[xc][j][p] - (C);
			fm12[xc][j][p] = fm14[xc][j][p] + (C);

			fm9[xc][j][p] = fm7[xc][j][p] - (A * 0 + B); // without A first as it will be cancelled out in rho  		
			fm8[xc][j][p] = fm10[xc][j][p] - (-A * 0 + B);

			mrho[xc][j][p] = fm0[xc][j][p] + fm1[xc][j][p] + fm2[xc][j][p] + fm3[xc][j][p]
				+ fm4[xc][j][p] + fm5[xc][j][p] + fm6[xc][j][p] + fm7[xc][j][p]
				+ fm8[xc][j][p] + fm9[xc][j][p] + fm10[xc][j][p] + fm11[xc][j][p]
				+ fm12[xc][j][p] + fm13[xc][j][p] + fm14[xc][j][p] + fm15[xc][j][p]
				+ fm16[xc][j][p] + fm17[xc][j][p] + fm18[xc][j][p];
			//-Ny
			A = (1.0 / 2.0)*((fm2[xc][j][p] + fm15[xc][j][p] + fm18[xc][j][p]) - (fm4[xc][j][p] + fm16[xc][j][p] + fm17[xc][j][p]));

			fm9[xc][j][p] = fm7[xc][j][p] + (A + B);  // now include A 	
			fm8[xc][j][p] = fm10[xc][j][p] - (A + B);
		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C,xc)
	for (j = 2; j <= mny; j++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//Known : f0,f2,f1,f4,f5,f6,f11,f9,f10,f12,f7,f14,f15,f16,f17,f18
			//Unknowns: f3,f8,f9,f12,f13,rho

			// B = 0.0;
			 xc = mnx;
			// //Nz = C from notes
			
			    gm0[xc][j][p] =  gm0[xc-1][j][p] ;
				gm1[xc][j][p] =  gm1[xc-1][j][p] ;
				gm2[xc][j][p] =  gm2[xc-1][j][p] ;
			    gm3[xc][j][p] =  gm3[xc-1][j][p] ;
				gm4[xc][j][p] =  gm4[xc-1][j][p] ;
				gm5[xc][j][p] =  gm5[xc-1][j][p] ;
				gm6[xc][j][p] =  gm6[xc-1][j][p] ;
				gm7[xc][j][p] =  gm7[xc-1][j][p] ;
				gm8[xc][j][p] =  gm8[xc-1][j][p] ;
				gm9[xc][j][p] =  gm9[xc-1][j][p] ;
				gm10[xc][j][p] =  gm10[xc-1][j][p] ;
				gm11[xc][j][p] =  gm11[xc-1][j][p] ;
				gm12[xc][j][p] =  gm12[xc-1][j][p] ;
				gm13[xc][j][p] =  gm13[xc-1][j][p] ;
				gm14[xc][j][p] =  gm14[xc-1][j][p] ;
				gm15[xc][j][p] =  gm15[xc-1][j][p] ;
				gm16[xc][j][p] =  gm16[xc-1][j][p] ;
				gm17[xc][j][p] =  gm17[xc-1][j][p] ;
				gm18[xc][j][p] =  gm18[xc-1][j][p] ;

		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C,xc)
	for (j = 2; j <= mny; j++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//Known : f0,f2,f1,f4,f5,f6,f11,f9,f10,f12,f7,f14,f15,f16,f17,f18
			//Unknowns: f3,f8,f9,f12,f13,rho

		
			 xc = mnx;
			    hm0[xc][j][p] =  hm0[xc-1][j][p] ;
				hm1[xc][j][p] =  hm1[xc-1][j][p] ;
				hm2[xc][j][p] =  hm2[xc-1][j][p] ;
			    hm3[xc][j][p] =  hm3[xc-1][j][p] ;
				hm4[xc][j][p] =  hm4[xc-1][j][p] ;
				hm5[xc][j][p] =  hm5[xc-1][j][p] ;
				hm6[xc][j][p] =  hm6[xc-1][j][p] ;
				hm7[xc][j][p] =  hm7[xc-1][j][p] ;
				hm8[xc][j][p] =  hm8[xc-1][j][p] ;
				hm9[xc][j][p] =  hm9[xc-1][j][p] ;
				hm10[xc][j][p] =  hm10[xc-1][j][p] ;
				hm11[xc][j][p] =  hm11[xc-1][j][p] ;
				hm12[xc][j][p] =  hm12[xc-1][j][p] ;
				hm13[xc][j][p] =  hm13[xc-1][j][p] ;
				hm14[xc][j][p] =  hm14[xc-1][j][p] ;
				hm15[xc][j][p] =  hm15[xc-1][j][p] ;
				hm16[xc][j][p] =  hm16[xc-1][j][p] ;
				hm17[xc][j][p] =  hm17[xc-1][j][p] ;
				hm18[xc][j][p] =  hm18[xc-1][j][p] ;

		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C,xc)
	for (j = 2; j <= mny; j++)
	{
		for (p = 2; p <= nz - 1; p++)
		{
			//Known : f0,f2,f1,f4,f5,f6,f11,f9,f10,f12,f7,f14,f15,f16,f17,f18
			//Unknowns: f3,f8,f9,f12,f13,rho
			 xc = mnx;
			    pm0[xc][j][p] =  pm0[xc-1][j][p] ;
				pm1[xc][j][p] =  pm1[xc-1][j][p] ;
				pm2[xc][j][p] =  pm2[xc-1][j][p] ;
			    pm3[xc][j][p] =  pm3[xc-1][j][p] ;
				pm4[xc][j][p] =  pm4[xc-1][j][p] ;
				pm5[xc][j][p] =  pm5[xc-1][j][p] ;
				pm6[xc][j][p] =  pm6[xc-1][j][p] ;
				pm7[xc][j][p] =  pm7[xc-1][j][p] ;
				pm8[xc][j][p] =  pm8[xc-1][j][p] ;
				pm9[xc][j][p] =  pm9[xc-1][j][p] ;
				pm10[xc][j][p] =  pm10[xc-1][j][p] ;
				pm11[xc][j][p] =  pm11[xc-1][j][p] ;
				pm12[xc][j][p] =  pm12[xc-1][j][p] ;
				pm13[xc][j][p] =  pm13[xc-1][j][p] ;
				pm14[xc][j][p] =  pm14[xc-1][j][p] ;
				pm15[xc][j][p] =  pm15[xc-1][j][p] ;
				pm16[xc][j][p] =  pm16[xc-1][j][p] ;
				pm17[xc][j][p] =  pm17[xc-1][j][p] ;
				pm18[xc][j][p] =  pm18[xc-1][j][p] ;

		}
	}


	//Case16: G'H'D'C'E'F' plane as shown in schematic excluding edges
	#pragma omp parallel for collapse(2) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		for (j = 2; j <= iny - 1; j++)
		{
			//		known: f0,f1,f2,f3,f4,f5,f7,f8,f9,f10,f11,f12,f15,f16
			//		unknown: f6,f13,f14,f17,f18,rho
			C = 0;
			B = (1.0 / 2.0)*(f2[i][j][nz] - f4[i][j][nz] + f7[i][j][nz] + f8[i][j][nz] - f9[i][j][nz] - f10[i][j][nz]);

			f6[i][j][nz] = f5[i][j][nz];
			f17[i][j][nz] = f15[i][j][nz] - (-B - C);
			f18[i][j][nz] = f16[i][j][nz] - (B - C);
			f13[i][j][nz] = f11[i][j][nz] - (-A * 0 - C);
			f14[i][j][nz] = f12[i][j][nz] - (A * 0 - C);

			rho[i][j][nz] = f0[i][j][nz] + f1[i][j][nz] + f2[i][j][nz] + f3[i][j][nz]
				+ f4[i][j][nz] + f5[i][j][nz] + f6[i][j][nz] + f7[i][j][nz]
				+ f8[i][j][nz] + f9[i][j][nz] + f10[i][j][nz] + f11[i][j][nz]
				+ f12[i][j][nz] + f13[i][j][nz] + f14[i][j][nz] + f15[i][j][nz]
				+ f16[i][j][nz] + f17[i][j][nz] + f18[i][j][nz];

			A = (1.0 / 2.0)*(f1[i][j][nz] - f3[i][j][nz] + f7[i][j][nz] - f8[i][j][nz] - f9[i][j][nz] + f10[i][j][nz]);

			f13[i][j][nz] = f11[i][j][nz] - (-A - C);
			f14[i][j][nz] = f12[i][j][nz] - (A - C);

		}
	}
	#pragma omp parallel for collapse(2) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		for (j = 2; j <= iny - 1; j++)
		{
			//		known: f0,f1,f2,f3,f4,f5,f7,f8,f9,f10,f11,f12,f15,f16
			//		unknown: f6,f13,f14,f17,f18,rho
			// C = 0;
			// B = (1.0 / 2.0)*(g2[i][j][nz] - g4[i][j][nz] + g7[i][j][nz] + g8[i][j][nz] - g9[i][j][nz] - g10[i][j][nz]);

			
			    g0[i][j][nz] =  g0[i][j][nz-1] ;
				g1[i][j][nz] =  g1[i][j][nz-1] ;
				g2[i][j][nz] =  g2[i][j][nz-1] ;
			    g3[i][j][nz] =  g3[i][j][nz-1] ;
				g4[i][j][nz] =  g4[i][j][nz-1] ;
				g5[i][j][nz] =  g5[i][j][nz-1] ;
				g6[i][j][nz] =  g6[i][j][nz-1] ;
				g7[i][j][nz] =  g7[i][j][nz-1] ;
				g8[i][j][nz] =  g8[i][j][nz-1] ;
				g9[i][j][nz] =  g9[i][j][nz-1] ;
				g10[i][j][nz] =  g10[i][j][nz-1] ;
				g11[i][j][nz] =  g11[i][j][nz-1] ;
				g12[i][j][nz] =  g12[i][j][nz-1] ;
				g13[i][j][nz] =  g13[i][j][nz-1] ;
				g14[i][j][nz] =  g14[i][j][nz-1] ;
				g15[i][j][nz] =  g15[i][j][nz-1] ;
				g16[i][j][nz] =  g16[i][j][nz-1] ;
				g17[i][j][nz] =  g17[i][j][nz-1] ;
				g18[i][j][nz] =  g18[i][j][nz-1] ;

		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		for (j = 2; j <= iny - 1; j++)
		{
			//		known: f0,f1,f2,f3,f4,f5,f7,f8,f9,f10,f11,f12,f15,f16
			//		unknown: f6,f13,f14,f17,f18,rho
			
			    h0[i][j][nz] =  h0[i][j][nz-1] ;
				h1[i][j][nz] =  h1[i][j][nz-1] ;
				h2[i][j][nz] =  h2[i][j][nz-1] ;
			    h3[i][j][nz] =  h3[i][j][nz-1] ;
				h4[i][j][nz] =  h4[i][j][nz-1] ;
				h5[i][j][nz] =  h5[i][j][nz-1] ;
				h6[i][j][nz] =  h6[i][j][nz-1] ;
				h7[i][j][nz] =  h7[i][j][nz-1] ;
				h8[i][j][nz] =  h8[i][j][nz-1] ;
				h9[i][j][nz] =  h9[i][j][nz-1] ;
				h10[i][j][nz] =  h10[i][j][nz-1] ;
				h11[i][j][nz] =  h11[i][j][nz-1] ;
				h12[i][j][nz] =  h12[i][j][nz-1] ;
				h13[i][j][nz] =  h13[i][j][nz-1] ;
				h14[i][j][nz] =  h14[i][j][nz-1] ;
				h15[i][j][nz] =  h15[i][j][nz-1] ;
				h16[i][j][nz] =  h16[i][j][nz-1] ;
				h17[i][j][nz] =  h17[i][j][nz-1] ;
				h18[i][j][nz] =  h18[i][j][nz-1] ;

		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		for (j = 2; j <= iny - 1; j++)
		{
			//		known: f0,f1,f2,f3,f4,f5,f7,f8,f9,f10,f11,f12,f15,f16
			//		unknown: f6,f13,f14,f17,f18,rho
			
			    p0[i][j][nz] =  p0[i][j][nz-1] ;
				p1[i][j][nz] =  p1[i][j][nz-1] ;
				p2[i][j][nz] =  p2[i][j][nz-1] ;
			    p3[i][j][nz] =  p3[i][j][nz-1] ;
				p4[i][j][nz] =  p4[i][j][nz-1] ;
				p5[i][j][nz] =  p5[i][j][nz-1] ;
				p6[i][j][nz] =  p6[i][j][nz-1] ;
				p7[i][j][nz] =  p7[i][j][nz-1] ;
				p8[i][j][nz] =  p8[i][j][nz-1] ;
				p9[i][j][nz] =  p9[i][j][nz-1] ;
				p10[i][j][nz] =  p10[i][j][nz-1] ;
				p11[i][j][nz] =  p11[i][j][nz-1] ;
				p12[i][j][nz] =  p12[i][j][nz-1] ;
				p13[i][j][nz] =  p13[i][j][nz-1] ;
				p14[i][j][nz] =  p14[i][j][nz-1] ;
				p15[i][j][nz] =  p15[i][j][nz-1] ;
				p16[i][j][nz] =  p16[i][j][nz-1] ;
				p17[i][j][nz] =  p17[i][j][nz-1] ;
				p18[i][j][nz] =  p18[i][j][nz-1] ;

		}
	}

	//Case 17: A'B'C'D' plane as shown in schematic excluding edges
	#pragma omp parallel for collapse(2) private (A,B,C,xc,yc)
	for (i = 2; i <= mnx - 1; i++)
	{
		for (j = 2; j <= mny + 1; j++)
		{
			//		known: f0,f1,f2,f3,f4,f5,f7,f8,f9,f10,f11,f12,f15,f16
			//		unknown: f6,f13,f14,f17,f18,rho
			C = 0;
			B = 1.0 / 2.0*(fm2[i][j][nz] - fm4[i][j][nz] + fm7[i][j][nz] + fm8[i][j][nz] - fm9[i][j][nz] - fm10[i][j][nz]);

			fm6[i][j][nz] = fm5[i][j][nz];
			fm17[i][j][nz] = fm15[i][j][nz] - (-B - C);
			fm18[i][j][nz] = fm16[i][j][nz] - (B - C);
			fm13[i][j][nz] = fm11[i][j][nz] - (-A * 0 - C);
			fm14[i][j][nz] = fm12[i][j][nz] - (A * 0 - C);

			mrho[i][j][nz] = fm0[i][j][nz] + fm1[i][j][nz] + fm2[i][j][nz] + fm3[i][j][nz]
				+ fm4[i][j][nz] + fm5[i][j][nz] + fm6[i][j][nz] + fm7[i][j][nz]
				+ fm8[i][j][nz] + fm9[i][j][nz] + fm10[i][j][nz] + fm11[i][j][nz]
				+ fm12[i][j][nz] + fm13[i][j][nz] + fm14[i][j][nz] + fm15[i][j][nz]
				+ fm16[i][j][nz] + fm17[i][j][nz] + fm18[i][j][nz];

			A = (1.0 / 2.0)*(fm1[i][j][nz] - fm3[i][j][nz] + fm7[i][j][nz] - fm8[i][j][nz] - fm9[i][j][nz] + fm10[i][j][nz]);


			fm13[i][j][nz] = fm11[i][j][nz] - (-A - C);
			fm14[i][j][nz] = fm12[i][j][nz] - (A - C);

			if (j == mny + 1)
			{
				xc = i + ia - 1;
				yc = 1;
				C = 0;
				B = (1.0 / 2.0)*(f2[xc][yc][nz] - f4[xc][yc][nz] + f7[xc][yc][nz] + f8[xc][yc][nz] - f9[xc][yc][nz] - f10[xc][yc][nz]);

				f6[xc][yc][nz] = f5[xc][yc][nz];
				f17[xc][yc][nz] = f15[xc][yc][nz] - (-B - C);
				f18[xc][yc][nz] = f16[xc][yc][nz] - (B - C);
				f13[xc][yc][nz] = f11[xc][yc][nz] - (-A * 0 - C);
				f14[xc][yc][nz] = f12[xc][yc][nz] - (A * 0 - C);

				rho[xc][yc][nz] = f0[xc][yc][nz] + f1[xc][yc][nz] + f2[xc][yc][nz] + f3[xc][yc][nz]
					+ f4[xc][yc][nz] + f5[xc][yc][nz] + f6[xc][yc][nz] + f7[xc][yc][nz]
					+ f8[xc][yc][nz] + f9[xc][yc][nz] + f10[xc][yc][nz] + f11[xc][yc][nz]
					+ f12[xc][yc][nz] + f13[xc][yc][nz] + f14[xc][yc][nz] + f15[xc][yc][nz]
					+ f16[xc][yc][nz] + f17[xc][yc][nz] + f18[xc][yc][nz];

				A = (1.0 / 2.0)*(f1[xc][yc][nz] - f3[xc][yc][nz] + f7[xc][yc][nz] - f8[xc][yc][nz] - f9[xc][yc][nz] + f10[xc][yc][nz]);

				f13[xc][yc][nz] = f11[xc][yc][nz] - (-A - C);
				f14[xc][yc][nz] = f12[xc][yc][nz] - (A - C);
			}
		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C,xc,yc)
	for (i = 2; i <= mnx - 1; i++)
	{
		for (j = 2; j <= mny + 1; j++)
		{
			//		known: f0,f1,f2,f3,f4,f5,f7,f8,f9,f10,f11,f12,f15,f16
			//		unknown: f6,f13,f14,f17,f18,rho
		
			    gm0[i][j][nz] =  gm0[i][j][nz-1] ;
				gm1[i][j][nz] =  gm1[i][j][nz-1] ;
				gm2[i][j][nz] =  gm2[i][j][nz-1] ;
			    gm3[i][j][nz] =  gm3[i][j][nz-1] ;
				gm4[i][j][nz] =  gm4[i][j][nz-1] ;
				gm5[i][j][nz] =  gm5[i][j][nz-1] ;
				gm6[i][j][nz] =  gm6[i][j][nz-1] ;
				gm7[i][j][nz] =  gm7[i][j][nz-1] ;
				gm8[i][j][nz] =  gm8[i][j][nz-1] ;
				gm9[i][j][nz] =  gm9[i][j][nz-1] ;
				gm10[i][j][nz] =  gm10[i][j][nz-1] ;
				gm11[i][j][nz] =  gm11[i][j][nz-1] ;
				gm12[i][j][nz] =  gm12[i][j][nz-1] ;
				gm13[i][j][nz] =  gm13[i][j][nz-1] ;
				gm14[i][j][nz] =  gm14[i][j][nz-1] ;
				gm15[i][j][nz] =  gm15[i][j][nz-1] ;
				gm16[i][j][nz] =  gm16[i][j][nz-1] ;
				gm17[i][j][nz] =  gm17[i][j][nz-1] ;
				gm18[i][j][nz] =  gm18[i][j][nz-1] ;

			if (j == mny + 1)
			{
				 xc = i + ia - 1;
				 yc = 1;
				// C = 0;
				// B = (1.0 / 2.0)*(g2[xc][yc][nz] - g4[xc][yc][nz] + g7[xc][yc][nz] + g8[xc][yc][nz] - g9[xc][yc][nz] - g10[xc][yc][nz]);

				
				g0[xc][yc][nz] =  g0[xc][yc][nz-1] ;
				g1[xc][yc][nz] =  g1[xc][yc][nz-1] ;
				g2[xc][yc][nz] =  g2[xc][yc][nz-1] ;
			    g3[xc][yc][nz] =  g3[xc][yc][nz-1] ;
				g4[xc][yc][nz] =  g4[xc][yc][nz-1] ;
				g5[xc][yc][nz] =  g5[xc][yc][nz-1] ;
				g6[xc][yc][nz] =  g6[xc][yc][nz-1] ;
				g7[xc][yc][nz] =  g7[xc][yc][nz-1] ;
				g8[xc][yc][nz] =  g8[xc][yc][nz-1] ;
				g9[xc][yc][nz] =  g9[xc][yc][nz-1] ;
				g10[xc][yc][nz] =  g10[xc][yc][nz-1] ;
				g11[xc][yc][nz] =  g11[xc][yc][nz-1] ;
				g12[xc][yc][nz] =  g12[xc][yc][nz-1] ;
				g13[xc][yc][nz] =  g13[xc][yc][nz-1] ;
				g14[xc][yc][nz] =  g14[xc][yc][nz-1] ;
				g15[xc][yc][nz] =  g15[xc][yc][nz-1] ;
				g16[xc][yc][nz] =  g16[xc][yc][nz-1] ;
				g17[xc][yc][nz] =  g17[xc][yc][nz-1] ;
				g18[xc][yc][nz] =  g18[xc][yc][nz-1] ;
			}
		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C,xc,yc)
	for (i = 2; i <= mnx - 1; i++)
	{
		for (j = 2; j <= mny + 1; j++)
		{
			    hm0[i][j][nz] =  hm0[i][j][nz-1] ;
				hm1[i][j][nz] =  hm1[i][j][nz-1] ;
				hm2[i][j][nz] =  hm2[i][j][nz-1] ;
			    hm3[i][j][nz] =  hm3[i][j][nz-1] ;
				hm4[i][j][nz] =  hm4[i][j][nz-1] ;
				hm5[i][j][nz] =  hm5[i][j][nz-1] ;
				hm6[i][j][nz] =  hm6[i][j][nz-1] ;
				hm7[i][j][nz] =  hm7[i][j][nz-1] ;
				hm8[i][j][nz] =  hm8[i][j][nz-1] ;
				hm9[i][j][nz] =  hm9[i][j][nz-1] ;
				hm10[i][j][nz] =  hm10[i][j][nz-1] ;
				hm11[i][j][nz] =  hm11[i][j][nz-1] ;
				hm12[i][j][nz] =  hm12[i][j][nz-1] ;
				hm13[i][j][nz] =  hm13[i][j][nz-1] ;
				hm14[i][j][nz] =  hm14[i][j][nz-1] ;
				hm15[i][j][nz] =  hm15[i][j][nz-1] ;
				hm16[i][j][nz] =  hm16[i][j][nz-1] ;
				hm17[i][j][nz] =  hm17[i][j][nz-1] ;
				hm18[i][j][nz] =  hm18[i][j][nz-1] ;

			if (j == mny + 1)
			{
				 xc = i + ia - 1;
				 yc = 1;
				
				h0[xc][yc][nz] =  h0[xc][yc][nz-1] ;
				h1[xc][yc][nz] =  h1[xc][yc][nz-1] ;
				h2[xc][yc][nz] =  h2[xc][yc][nz-1] ;
			    h3[xc][yc][nz] =  h3[xc][yc][nz-1] ;
				h4[xc][yc][nz] =  h4[xc][yc][nz-1] ;
				h5[xc][yc][nz] =  h5[xc][yc][nz-1] ;
				h6[xc][yc][nz] =  h6[xc][yc][nz-1] ;
				h7[xc][yc][nz] =  h7[xc][yc][nz-1] ;
				h8[xc][yc][nz] =  h8[xc][yc][nz-1] ;
				h9[xc][yc][nz] =  h9[xc][yc][nz-1] ;
				h10[xc][yc][nz] =  h10[xc][yc][nz-1] ;
				h11[xc][yc][nz] =  h11[xc][yc][nz-1] ;
				h12[xc][yc][nz] =  h12[xc][yc][nz-1] ;
				h13[xc][yc][nz] =  h13[xc][yc][nz-1] ;
				h14[xc][yc][nz] =  h14[xc][yc][nz-1] ;
				h15[xc][yc][nz] =  h15[xc][yc][nz-1] ;
				h16[xc][yc][nz] =  h16[xc][yc][nz-1] ;
				h17[xc][yc][nz] =  h17[xc][yc][nz-1] ;
				h18[xc][yc][nz] =  h18[xc][yc][nz-1] ;
			}
		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C,xc,yc)
	for (i = 2; i <= mnx - 1; i++)
	{
		for (j = 2; j <= mny + 1; j++)
		{
			    pm0[i][j][nz] =  pm0[i][j][nz-1] ;
				pm1[i][j][nz] =  pm1[i][j][nz-1] ;
				pm2[i][j][nz] =  pm2[i][j][nz-1] ;
			    pm3[i][j][nz] =  pm3[i][j][nz-1] ;
				pm4[i][j][nz] =  pm4[i][j][nz-1] ;
				pm5[i][j][nz] =  pm5[i][j][nz-1] ;
				pm6[i][j][nz] =  pm6[i][j][nz-1] ;
				pm7[i][j][nz] =  pm7[i][j][nz-1] ;
				pm8[i][j][nz] =  pm8[i][j][nz-1] ;
				pm9[i][j][nz] =  pm9[i][j][nz-1] ;
				pm10[i][j][nz] =  pm10[i][j][nz-1] ;
				pm11[i][j][nz] =  pm11[i][j][nz-1] ;
				pm12[i][j][nz] =  pm12[i][j][nz-1] ;
				pm13[i][j][nz] =  pm13[i][j][nz-1] ;
				pm14[i][j][nz] =  pm14[i][j][nz-1] ;
				pm15[i][j][nz] =  pm15[i][j][nz-1] ;
				pm16[i][j][nz] =  pm16[i][j][nz-1] ;
				pm17[i][j][nz] =  pm17[i][j][nz-1] ;
				pm18[i][j][nz] =  pm18[i][j][nz-1] ;

			if (j == mny + 1)
			{
				 xc = i + ia - 1;
				 yc = 1;
				
				p0[xc][yc][nz] =  p0[xc][yc][nz-1] ;
				p1[xc][yc][nz] =  p1[xc][yc][nz-1] ;
				p2[xc][yc][nz] =  p2[xc][yc][nz-1] ;
			    p3[xc][yc][nz] =  p3[xc][yc][nz-1] ;
				p4[xc][yc][nz] =  p4[xc][yc][nz-1] ;
				p5[xc][yc][nz] =  p5[xc][yc][nz-1] ;
				p6[xc][yc][nz] =  p6[xc][yc][nz-1] ;
				p7[xc][yc][nz] =  p7[xc][yc][nz-1] ;
				p8[xc][yc][nz] =  p8[xc][yc][nz-1] ;
				p9[xc][yc][nz] =  p9[xc][yc][nz-1] ;
				p10[xc][yc][nz] =  p10[xc][yc][nz-1] ;
				p11[xc][yc][nz] =  p11[xc][yc][nz-1] ;
				p12[xc][yc][nz] =  p12[xc][yc][nz-1] ;
				p13[xc][yc][nz] =  p13[xc][yc][nz-1] ;
				p14[xc][yc][nz] =  p14[xc][yc][nz-1] ;
				p15[xc][yc][nz] =  p15[xc][yc][nz-1] ;
				p16[xc][yc][nz] =  p16[xc][yc][nz-1] ;
				p17[xc][yc][nz] =  p17[xc][yc][nz-1] ;
				p18[xc][yc][nz] =  p18[xc][yc][nz-1] ;
			}
		}
	}


	//Case 18: GHEF plane as shown in schematic excluding edges
	#pragma omp parallel for collapse(2) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		for (j = 2; j <= iny - 1; j++)
		{

			//		known: f0,f1,f2,f3,f4,f6,f7,f8,f9,f10,f13,f14,f17,f18
			//		unknown: f5,f11,f12,f15,f16,rho

			C = 0;A=0;
			B = (1.0 / 2.0)*(f2[i][j][1] - f4[i][j][1] + f7[i][j][1] + f8[i][j][1] - f9[i][j][1] - f10[i][j][1]);

			f5[i][j][1] = f6[i][j][1];
			f15[i][j][1] = f17[i][j][1] - (B + C);
			f16[i][j][1] = f18[i][j][1] - (-B + C);
			f11[i][j][1] = f13[i][j][1] - (A * 0 + C);
			f12[i][j][1] = f14[i][j][1] - (-A * 0 + C);

			rho[i][j][1] = f0[i][j][1] + f1[i][j][1] + f2[i][j][1] + f3[i][j][1]
				+ f4[i][j][1] + f5[i][j][1] + f6[i][j][1] + f7[i][j][1]
				+ f8[i][j][1] + f9[i][j][1] + f10[i][j][1] + f11[i][j][1]
				+ f12[i][j][1] + f13[i][j][1] + f14[i][j][1] + f15[i][j][1]
				+ f16[i][j][1] + f17[i][j][1] + f18[i][j][1];

			A = (1.0 / 2.0)*(f1[i][j][1] - f3[i][j][1] + f7[i][j][1] - f8[i][j][1] - f9[i][j][1] + f10[i][j][1]);

			f11[i][j][1] = f13[i][j][1] - (A + C);
			f12[i][j][1] = f14[i][j][1] - (-A + C);

		}
	}
	#pragma omp parallel for collapse(2) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		for (j = 2; j <= iny - 1; j++)
		{

			//		known: f0,f1,f2,f3,f4,f6,f7,f8,f9,f10,f13,f14,f17,f18
			//		unknown: f5,f11,f12,f15,f16,rho

		
			    g0[i][j][1] =  g0[i][j][2] ;
				g1[i][j][1] =  g1[i][j][2] ;
				g2[i][j][1] =  g2[i][j][2] ;
			    g3[i][j][1] =  g3[i][j][2] ;
				g4[i][j][1] =  g4[i][j][2] ;
				g5[i][j][1] =  g5[i][j][2] ;
				g6[i][j][1] =  g6[i][j][2] ;
				g7[i][j][1] =  g7[i][j][2] ;
				g8[i][j][1] =  g8[i][j][2] ;
				g9[i][j][1] =  g9[i][j][2] ;
				g10[i][j][1] =  g10[i][j][2] ;
				g11[i][j][1] =  g11[i][j][2] ;
				g12[i][j][1] =  g12[i][j][2] ;
				g13[i][j][1] =  g13[i][j][2] ;
				g14[i][j][1] =  g14[i][j][2] ;
				g15[i][j][1] =  g15[i][j][2] ;
				g16[i][j][1] =  g16[i][j][2] ;
				g17[i][j][1] =  g17[i][j][2] ;
				g18[i][j][1] =  g18[i][j][2] ;

		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		for (j = 2; j <= iny - 1; j++)
		{

			    h0[i][j][1] =  h0[i][j][2] ;
				h1[i][j][1] =  h1[i][j][2] ;
				h2[i][j][1] =  h2[i][j][2] ;
			    h3[i][j][1] =  h3[i][j][2] ;
				h4[i][j][1] =  h4[i][j][2] ;
				h5[i][j][1] =  h5[i][j][2] ;
				h6[i][j][1] =  h6[i][j][2] ;
				h7[i][j][1] =  h7[i][j][2] ;
				h8[i][j][1] =  h8[i][j][2] ;
				h9[i][j][1] =  h9[i][j][2] ;
				h10[i][j][1] =  h10[i][j][2] ;
				h11[i][j][1] =  h11[i][j][2] ;
				h12[i][j][1] =  h12[i][j][2] ;
				h13[i][j][1] =  h13[i][j][2] ;
				h14[i][j][1] =  h14[i][j][2] ;
				h15[i][j][1] =  h15[i][j][2] ;
				h16[i][j][1] =  h16[i][j][2] ;
				h17[i][j][1] =  h17[i][j][2] ;
				h18[i][j][1] =  h18[i][j][2] ;

		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		for (j = 2; j <= iny - 1; j++)
		{

			    p0[i][j][1] =  p0[i][j][2] ;
				p1[i][j][1] =  p1[i][j][2] ;
				p2[i][j][1] =  p2[i][j][2] ;
			    p3[i][j][1] =  p3[i][j][2] ;
				p4[i][j][1] =  p4[i][j][2] ;
				p5[i][j][1] =  p5[i][j][2] ;
				p6[i][j][1] =  p6[i][j][2] ;
				p7[i][j][1] =  p7[i][j][2] ;
				p8[i][j][1] =  p8[i][j][2] ;
				p9[i][j][1] =  p9[i][j][2] ;
				p10[i][j][1] =  p10[i][j][2] ;
				p11[i][j][1] =  p11[i][j][2] ;
				p12[i][j][1] =  p12[i][j][2] ;
				p13[i][j][1] =  p13[i][j][2] ;
				p14[i][j][1] =  p14[i][j][2] ;
				p15[i][j][1] =  p15[i][j][2] ;
				p16[i][j][1] =  p16[i][j][2] ;
				p17[i][j][1] =  p17[i][j][2] ;
				p18[i][j][1] =  p18[i][j][2] ;

		}
	}
	// Case19: ABCD plane  as shown in schematic excluding edges
	#pragma omp parallel for collapse(2) private (A,B,C,xc,yc)
	for (i = 2; i <= mnx - 1; i++)
	{
		for (j = 2; j <= mny + 1; j++)
		{
			//		known: f0,f1,f2,f3,f4,f6,f7,f8,f9,f10,f13,f14,f17,f18
			//		unknown: f5,f11,f12,f15,f16,rho

			C = 0;A=0;
			B = (1.0 / 2.0)*(fm2[i][j][1] - fm4[i][j][1] + fm7[i][j][1] + fm8[i][j][1] - fm9[i][j][1] - fm10[i][j][1]);

			fm5[i][j][1] = fm6[i][j][1];
			fm15[i][j][1] = fm17[i][j][1] - (B + C);
			fm16[i][j][1] = fm18[i][j][1] - (-B + C);
			fm11[i][j][1] = fm13[i][j][1] - (A * 0 + C);
			fm12[i][j][1] = fm14[i][j][1] - (-A * 0 + C);

			mrho[i][j][1] = fm0[i][j][1] + fm1[i][j][1] + fm2[i][j][1] + fm3[i][j][1]
				+ fm4[i][j][1] + fm5[i][j][1] + fm6[i][j][1] + fm7[i][j][1]
				+ fm8[i][j][1] + fm9[i][j][1] + fm10[i][j][1] + fm11[i][j][1]
				+ fm12[i][j][1] + fm13[i][j][1] + fm14[i][j][1] + fm15[i][j][1]
				+ fm16[i][j][1] + fm17[i][j][1] + fm18[i][j][1];

			A = (1.0 / 2.0)*(fm1[i][j][1] - fm3[i][j][1] + fm7[i][j][1] - fm8[i][j][1] - fm9[i][j][1] + fm10[i][j][1]);

			fm11[i][j][1] = fm13[i][j][1] - (A + C);
			fm12[i][j][1] = fm14[i][j][1] - (-A + C);

			if (j == mny + 1)
			{
				xc = i + ia - 1;
				yc = 1;
				C = 0;
				A= 0;
				B = (1.0 / 2.0)*(f2[xc][yc][1] - f4[xc][yc][1] + f7[xc][yc][1] + f8[xc][yc][1] - f9[xc][yc][1] - f10[xc][yc][1]);

				f5[xc][yc][1] = f6[xc][yc][1];
				f15[xc][yc][1] = f17[xc][yc][1] - (B + C);
				f16[xc][yc][1] = f18[xc][yc][1] - (-B + C);
				f11[xc][yc][1] = f13[xc][yc][1] - (A * 0 + C);
				f12[xc][yc][1] = f14[xc][yc][1] - (-A * 0 + C);

				rho[xc][yc][1] = f0[xc][yc][1] + f1[xc][yc][1] + f2[xc][yc][1] + f3[xc][yc][1]
					+ f4[xc][yc][1] + f5[xc][yc][1] + f6[xc][yc][1] + f7[xc][yc][1]
					+ f8[xc][yc][1] + f9[xc][yc][1] + f10[xc][yc][1] + f11[xc][yc][1]
					+ f12[xc][yc][1] + f13[xc][yc][1] + f14[xc][yc][1] + f15[xc][yc][1]
					+ f16[xc][yc][1] + f17[xc][yc][1] + f18[xc][yc][1];

				A = (1.0 / 2.0)*(f1[xc][yc][1] - f3[xc][yc][1] + f7[xc][yc][1] - f8[xc][yc][1] - f9[xc][yc][1] + f10[xc][yc][1]);
				f11[xc][yc][1] = f13[xc][yc][1] - (A + C);
				f12[xc][yc][1] = f14[xc][yc][1] - (-A + C);
			}

		}
	}
	#pragma omp parallel for collapse(2) private (A,B,C,xc,yc)
	for (i = 2; i <= mnx - 1; i++)
	{
		for (j = 2; j <= mny + 1; j++)
		{
			//		known: f0,f1,f2,f3,f4,f6,f7,f8,f9,f10,f13,f14,f17,f18
			//		unknown: f5,f11,f12,f15,f16,rho

			    gm0[i][j][1] =  gm0[i][j][1+1] ;
				gm1[i][j][1] =  gm1[i][j][1+1] ;
				gm2[i][j][1] =  gm2[i][j][1+1] ;
			    gm3[i][j][1] =  gm3[i][j][1+1] ;
				gm4[i][j][1] =  gm4[i][j][1+1] ;
				gm5[i][j][1] =  gm5[i][j][1+1] ;
				gm6[i][j][1] =  gm6[i][j][1+1] ;
				gm7[i][j][1] =  gm7[i][j][1+1] ;
				gm8[i][j][1] =  gm8[i][j][1+1] ;
				gm9[i][j][1] =  gm9[i][j][1+1] ;
				gm10[i][j][1] =  gm10[i][j][1+1] ;
				gm11[i][j][1] =  gm11[i][j][1+1] ;
				gm12[i][j][1] =  gm12[i][j][1+1] ;
				gm13[i][j][1] =  gm13[i][j][1+1] ;
				gm14[i][j][1] =  gm14[i][j][1+1] ;
				gm15[i][j][1] =  gm15[i][j][1+1] ;
				gm16[i][j][1] =  gm16[i][j][1+1] ;
				gm17[i][j][1] =  gm17[i][j][1+1] ;
				gm18[i][j][1] =  gm18[i][j][1+1] ;

			if (j == mny + 1)
			{
				xc = i + ia - 1;
				yc = 1;
				// C = 0;
				// A= 0;
				// B = (1.0 / 2.0)*(g2[xc][yc][1] - g4[xc][yc][1] + g7[xc][yc][1] + g8[xc][yc][1] - g9[xc][yc][1] - g10[xc][yc][1]);

		
				g0[xc][yc][1] =  g0[xc][yc][1+1] ;
				g1[xc][yc][1] =  g1[xc][yc][1+1] ;
				g2[xc][yc][1] =  g2[xc][yc][1+1] ;
			    g3[xc][yc][1] =  g3[xc][yc][1+1] ;
				g4[xc][yc][1] =  g4[xc][yc][1+1] ;
				g5[xc][yc][1] =  g5[xc][yc][1+1] ;
				g6[xc][yc][1] =  g6[xc][yc][1+1] ;
				g7[xc][yc][1] =  g7[xc][yc][1+1] ;
				g8[xc][yc][1] =  g8[xc][yc][1+1] ;
				g9[xc][yc][1] =  g9[xc][yc][1+1] ;
				g10[xc][yc][1] =  g10[xc][yc][1+1] ;
				g11[xc][yc][1] =  g11[xc][yc][1+1] ;
				g12[xc][yc][1] =  g12[xc][yc][1+1] ;
				g13[xc][yc][1] =  g13[xc][yc][1+1] ;
				g14[xc][yc][1] =  g14[xc][yc][1+1] ;
				g15[xc][yc][1] =  g15[xc][yc][1+1] ;
				g16[xc][yc][1] =  g16[xc][yc][1+1] ;
				g17[xc][yc][1] =  g17[xc][yc][1+1] ;
				g18[xc][yc][1] =  g18[xc][yc][1+1] ;
			}

		}
	}
#pragma omp parallel for collapse(2) private (A,B,C,xc,yc)
	for (i = 2; i <= mnx - 1; i++)
	{
		for (j = 2; j <= mny + 1; j++)
		{
			//		known: f0,f1,f2,f3,f4,f6,f7,f8,f9,f10,f13,f14,f17,f18
			//		unknown: f5,f11,f12,f15,f16,rho
			    hm0[i][j][1] =  hm0[i][j][1+1] ;
				hm1[i][j][1] =  hm1[i][j][1+1] ;
				hm2[i][j][1] =  hm2[i][j][1+1] ;
			    hm3[i][j][1] =  hm3[i][j][1+1] ;
				hm4[i][j][1] =  hm4[i][j][1+1] ;
				hm5[i][j][1] =  hm5[i][j][1+1] ;
				hm6[i][j][1] =  hm6[i][j][1+1] ;
				hm7[i][j][1] =  hm7[i][j][1+1] ;
				hm8[i][j][1] =  hm8[i][j][1+1] ;
				hm9[i][j][1] =  hm9[i][j][1+1] ;
				hm10[i][j][1] =  hm10[i][j][1+1] ;
				hm11[i][j][1] =  hm11[i][j][1+1] ;
				hm12[i][j][1] =  hm12[i][j][1+1] ;
				hm13[i][j][1] =  hm13[i][j][1+1] ;
				hm14[i][j][1] =  hm14[i][j][1+1] ;
				hm15[i][j][1] =  hm15[i][j][1+1] ;
				hm16[i][j][1] =  hm16[i][j][1+1] ;
				hm17[i][j][1] =  hm17[i][j][1+1] ;
				hm18[i][j][1] =  hm18[i][j][1+1] ;

			if (j == mny + 1)
			{
				xc = i + ia - 1;
				yc = 1;
				
				h0[xc][yc][1] =  h0[xc][yc][1+1] ;
				h1[xc][yc][1] =  h1[xc][yc][1+1] ;
				h2[xc][yc][1] =  h2[xc][yc][1+1] ;
			    h3[xc][yc][1] =  h3[xc][yc][1+1] ;
				h4[xc][yc][1] =  h4[xc][yc][1+1] ;
				h5[xc][yc][1] =  h5[xc][yc][1+1] ;
				h6[xc][yc][1] =  h6[xc][yc][1+1] ;
				h7[xc][yc][1] =  h7[xc][yc][1+1] ;
				h8[xc][yc][1] =  h8[xc][yc][1+1] ;
				h9[xc][yc][1] =  h9[xc][yc][1+1] ;
				h10[xc][yc][1] =  h10[xc][yc][1+1] ;
				h11[xc][yc][1] =  h11[xc][yc][1+1] ;
				h12[xc][yc][1] =  h12[xc][yc][1+1] ;
				h13[xc][yc][1] =  h13[xc][yc][1+1] ;
				h14[xc][yc][1] =  h14[xc][yc][1+1] ;
				h15[xc][yc][1] =  h15[xc][yc][1+1] ;
				h16[xc][yc][1] =  h16[xc][yc][1+1] ;
				h17[xc][yc][1] =  h17[xc][yc][1+1] ;
				h18[xc][yc][1] =  h18[xc][yc][1+1] ;
			}

		}
	}

	#pragma omp parallel for collapse(2) private (A,B,C,xc,yc)
	for (i = 2; i <= mnx - 1; i++)
	{
		for (j = 2; j <= mny + 1; j++)
		{
			//		known: f0,f1,f2,f3,f4,f6,f7,f8,f9,f10,f13,f14,f17,f18
			//		unknown: f5,f11,f12,f15,f16,rho
			    pm0[i][j][1] =  pm0[i][j][1+1] ;
				pm1[i][j][1] =  pm1[i][j][1+1] ;
				pm2[i][j][1] =  pm2[i][j][1+1] ;
			    pm3[i][j][1] =  pm3[i][j][1+1] ;
				pm4[i][j][1] =  pm4[i][j][1+1] ;
				pm5[i][j][1] =  pm5[i][j][1+1] ;
				pm6[i][j][1] =  pm6[i][j][1+1] ;
				pm7[i][j][1] =  pm7[i][j][1+1] ;
				pm8[i][j][1] =  pm8[i][j][1+1] ;
				pm9[i][j][1] =  pm9[i][j][1+1] ;
				pm10[i][j][1] =  pm10[i][j][1+1] ;
				pm11[i][j][1] =  pm11[i][j][1+1] ;
				pm12[i][j][1] =  pm12[i][j][1+1] ;
				pm13[i][j][1] =  pm13[i][j][1+1] ;
				pm14[i][j][1] =  pm14[i][j][1+1] ;
				pm15[i][j][1] =  pm15[i][j][1+1] ;
				pm16[i][j][1] =  pm16[i][j][1+1] ;
				pm17[i][j][1] =  pm17[i][j][1+1] ;
				pm18[i][j][1] =  pm18[i][j][1+1] ;

			if (j == mny + 1)
			{
				xc = i + ia - 1;
				yc = 1;
				
				p0[xc][yc][1] =  p0[xc][yc][1+1] ;
				p1[xc][yc][1] =  p1[xc][yc][1+1] ;
				p2[xc][yc][1] =  p2[xc][yc][1+1] ;
			    p3[xc][yc][1] =  p3[xc][yc][1+1] ;
				p4[xc][yc][1] =  p4[xc][yc][1+1] ;
				p5[xc][yc][1] =  p5[xc][yc][1+1] ;
				p6[xc][yc][1] =  p6[xc][yc][1+1] ;
				p7[xc][yc][1] =  p7[xc][yc][1+1] ;
				p8[xc][yc][1] =  p8[xc][yc][1+1] ;
				p9[xc][yc][1] =  p9[xc][yc][1+1] ;
				p10[xc][yc][1] =  p10[xc][yc][1+1] ;
				p11[xc][yc][1] =  p11[xc][yc][1+1] ;
				p12[xc][yc][1] =  p12[xc][yc][1+1] ;
				p13[xc][yc][1] =  p13[xc][yc][1+1] ;
				p14[xc][yc][1] =  p14[xc][yc][1+1] ;
				p15[xc][yc][1] =  p15[xc][yc][1+1] ;
				p16[xc][yc][1] =  p16[xc][yc][1+1] ;
				p17[xc][yc][1] =  p17[xc][yc][1+1] ;
				p18[xc][yc][1] =  p18[xc][yc][1+1] ;
			}

		}
	}

	// Case20: G'F' edge as shown in schematic excluding its nodes
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		//		known: f0,f1,f2,f3,f5,f7,f8,f11,f12,f15
		//		unknown: f4,f6,f9,f10,f13,f14,f16,f17,f18,rho

		B = 0;
		C = 0;
		A = 0;
		f4[i][iny][nz] = f2[i][iny][nz];
		f6[i][iny][nz] = f5[i][iny][nz];
		f9[i][iny][nz] = f7[i][iny][nz] - (-B - A * 0);
		f10[i][iny][nz] = f8[i][iny][nz] - (A * 0 - B);
		f13[i][iny][nz] = f11[i][iny][nz] - (-A * 0 - C);
		f14[i][iny][nz] = f12[i][iny][nz] - (A * 0 - C);
		f17[i][iny][nz] = f15[i][iny][nz] - (-B - C);

		f16[i][iny][nz] = fn16[i][iny][nz];//undo streaming
		f18[i][iny][nz] = fn18[i][iny][nz];

		f16[i][iny][nz] = (1.0 / 2.0)*(f16[i][iny][nz] + f18[i][iny][nz]);
		f18[i][iny][nz] = f16[i][iny][nz];

		rho[i][iny][nz] = f0[i][iny][nz] + f1[i][iny][nz] + f2[i][iny][nz] + f3[i][iny][nz]
			+ f4[i][iny][nz] + f5[i][iny][nz] + f6[i][iny][nz] + f7[i][iny][nz]
			+ f8[i][iny][nz] + f9[i][iny][nz] + f10[i][iny][nz] + f11[i][iny][nz]
			+ f12[i][iny][nz] + f13[i][iny][nz] + f14[i][iny][nz] + f15[i][iny][nz]
			+ f16[i][iny][nz] + f17[i][iny][nz] + f18[i][iny][nz];

		A = (1.0 / 4.0)*(f1[i][iny][nz] - f3[i][iny][nz]);

		f9[i][iny][nz] = f7[i][iny][nz] - (-B - A);
		f10[i][iny][nz] = f8[i][iny][nz] - (A - B);
		f13[i][iny][nz] = f11[i][iny][nz] - (-A - C);
		f14[i][iny][nz] = f12[i][iny][nz] - (A - C);
	}

	#pragma omp parallel for collapse(1) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		//		known: f0,f1,f2,f3,f5,f7,f8,f11,f12,f15
		//		unknown: f4,f6,f9,f10,f13,f14,f16,f17,f18,rho

		B = 0;
		C = 0;
		A = 0;
		g4[i][iny][nz] = g2[i][iny][nz];
		g6[i][iny][nz] = g5[i][iny][nz];
		g9[i][iny][nz] = g7[i][iny][nz] - (-B - A * 0);
		g10[i][iny][nz] = g8[i][iny][nz] - (A * 0 - B);
		g13[i][iny][nz] = g11[i][iny][nz] - (-A * 0 - C);
		g14[i][iny][nz] = g12[i][iny][nz] - (A * 0 - C);
		g17[i][iny][nz] = g15[i][iny][nz] - (-B - C);

		g16[i][iny][nz] = gn16[i][iny][nz];//undo streaming
		g18[i][iny][nz] = gn18[i][iny][nz];

		g16[i][iny][nz] = (1.0 / 2.0)*(g16[i][iny][nz] + g18[i][iny][nz]);
		g18[i][iny][nz] = g16[i][iny][nz];

		grho[i][iny][nz] = g0[i][iny][nz] + g1[i][iny][nz] + g2[i][iny][nz] + g3[i][iny][nz]
			+ g4[i][iny][nz] + g5[i][iny][nz] + g6[i][iny][nz] + g7[i][iny][nz]
			+ g8[i][iny][nz] + g9[i][iny][nz] + g10[i][iny][nz] + g11[i][iny][nz]
			+ g12[i][iny][nz] + g13[i][iny][nz] + g14[i][iny][nz] + g15[i][iny][nz]
			+ g16[i][iny][nz] + g17[i][iny][nz] + g18[i][iny][nz];

		A = (1.0 / 4.0)*(g1[i][iny][nz] - g3[i][iny][nz]);

		g9[i][iny][nz] = g7[i][iny][nz] - (-B - A);
		g10[i][iny][nz] = g8[i][iny][nz] - (A - B);
		g13[i][iny][nz] = g11[i][iny][nz] - (-A - C);
		g14[i][iny][nz] = g12[i][iny][nz] - (A - C);
	}

	#pragma omp parallel for collapse(1) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		//		known: f0,f1,f2,f3,f5,f7,f8,f11,f12,f15
		//		unknown: f4,f6,f9,f10,f13,f14,f16,f17,f18,rho

		B = 0;
		C = 0;
		A = 0;
		h4[i][iny][nz] = h2[i][iny][nz];
		h6[i][iny][nz] = h5[i][iny][nz];
		h9[i][iny][nz] = h7[i][iny][nz] - (-B - A * 0);
		h10[i][iny][nz] = h8[i][iny][nz] - (A * 0 - B);
		h13[i][iny][nz] = h11[i][iny][nz] - (-A * 0 - C);
		h14[i][iny][nz] = h12[i][iny][nz] - (A * 0 - C);
		h17[i][iny][nz] = h15[i][iny][nz] - (-B - C);

		h16[i][iny][nz] = hn16[i][iny][nz];//undo streaminh
		h18[i][iny][nz] = hn18[i][iny][nz];

		h16[i][iny][nz] = (1.0 / 2.0)*(h16[i][iny][nz] + h18[i][iny][nz]);
		h18[i][iny][nz] = h16[i][iny][nz];

		hrho[i][iny][nz] = h0[i][iny][nz] + h1[i][iny][nz] + h2[i][iny][nz] + h3[i][iny][nz]
			+ h4[i][iny][nz] + h5[i][iny][nz] + h6[i][iny][nz] + h7[i][iny][nz]
			+ h8[i][iny][nz] + h9[i][iny][nz] + h10[i][iny][nz] + h11[i][iny][nz]
			+ h12[i][iny][nz] + h13[i][iny][nz] + h14[i][iny][nz] + h15[i][iny][nz]
			+ h16[i][iny][nz] + h17[i][iny][nz] + h18[i][iny][nz];

		A = (1.0 / 4.0)*(h1[i][iny][nz] - h3[i][iny][nz]);

		h9[i][iny][nz] = h7[i][iny][nz] - (-B - A);
		h10[i][iny][nz] = h8[i][iny][nz] - (A - B);
		h13[i][iny][nz] = h11[i][iny][nz] - (-A - C);
		h14[i][iny][nz] = h12[i][iny][nz] - (A - C);
	}
#pragma omp parallel for collapse(1) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		//		known: f0,f1,f2,f3,f5,f7,f8,f11,f12,f15
		//		unknown: f4,f6,f9,f10,f13,f14,f16,f17,f18,rho

		B = 0;
		C = 0;
		A = 0;
		p4[i][iny][nz] = p2[i][iny][nz];
		p6[i][iny][nz] = p5[i][iny][nz];
		p9[i][iny][nz] = p7[i][iny][nz] - (-B - A * 0);
		p10[i][iny][nz] = p8[i][iny][nz] - (A * 0 - B);
		p13[i][iny][nz] = p11[i][iny][nz] - (-A * 0 - C);
		p14[i][iny][nz] = p12[i][iny][nz] - (A * 0 - C);
		p17[i][iny][nz] = p15[i][iny][nz] - (-B - C);

		p16[i][iny][nz] = pn16[i][iny][nz];//undo streaminp
		p18[i][iny][nz] = pn18[i][iny][nz];

		p16[i][iny][nz] = (1.0 / 2.0)*(p16[i][iny][nz] + p18[i][iny][nz]);
		p18[i][iny][nz] = p16[i][iny][nz];

		prho[i][iny][nz] = p0[i][iny][nz] + p1[i][iny][nz] + p2[i][iny][nz] + p3[i][iny][nz]
			+ p4[i][iny][nz] + p5[i][iny][nz] + p6[i][iny][nz] + p7[i][iny][nz]
			+ p8[i][iny][nz] + p9[i][iny][nz] + p10[i][iny][nz] + p11[i][iny][nz]
			+ p12[i][iny][nz] + p13[i][iny][nz] + p14[i][iny][nz] + p15[i][iny][nz]
			+ p16[i][iny][nz] + p17[i][iny][nz] + p18[i][iny][nz];

		A = (1.0 / 4.0)*(p1[i][iny][nz] - p3[i][iny][nz]);

		p9[i][iny][nz] = p7[i][iny][nz] - (-B - A);
		p10[i][iny][nz] = p8[i][iny][nz] - (A - B);
		p13[i][iny][nz] = p11[i][iny][nz] - (-A - C);
		p14[i][iny][nz] = p12[i][iny][nz] - (A - C);
	}

	// Case21: GF edge as shown in the schematic excluding nodes
#pragma omp parallel for collapse(1) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		//              known: f0,f1,f2,f3,f6,f7,f8,f13,f14,f18
		//              unknown: f4,f5,f9,f10,f11,f12,f15b,f16,f17b,rho
		C = 0;
		B = 0;
		A = 0;

		f4[i][iny][1] = f2[i][iny][1];
		f5[i][iny][1] = f6[i][iny][1];
		f9[i][iny][1] = f7[i][iny][1] - (-A * 0 - B);
		f10[i][iny][1] = f8[i][iny][1] - (A * 0 - B);
		f11[i][iny][1] = f13[i][iny][1] - (A * 0 + C);
		f12[i][iny][1] = f14[i][iny][1] - (-A * 0 + C);
		f16[i][iny][1] = f18[i][iny][1] - (C - B);

		f17[i][iny][1] = fn17[i][iny][1];//undo streaming
		f15[i][iny][1] = fn15[i][iny][1];

		f15[i][iny][1] = (1.0 / 2.0)*(f15[i][iny][1] + f17[i][iny][1]);
		f17[i][iny][1] = f15[i][iny][1];

		rho[i][iny][1] = f0[i][iny][1] + f1[i][iny][1] + f2[i][iny][1] + f3[i][iny][1]
			+ f4[i][iny][1] + f5[i][iny][1] + f6[i][iny][1] + f7[i][iny][1]
			+ f8[i][iny][1] + f9[i][iny][1] + f10[i][iny][1] + f11[i][iny][1]
			+ f12[i][iny][1] + f13[i][iny][1] + f14[i][iny][1] + f15[i][iny][1]
			+ f16[i][iny][1] + f17[i][iny][1] + f18[i][iny][1];

		A = (1.0 / 4.0)*(f1[i][iny][1] - f3[i][iny][1]);

		f9[i][iny][1] = f7[i][iny][1] - (-A - B);
		f10[i][iny][1] = f8[i][iny][1] - (A - B);
		f11[i][iny][1] = f13[i][iny][1] - (A + C);
		f12[i][iny][1] = f14[i][iny][1] - (-A + C);
	}
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		//              known: f0,f1,f2,f3,f6,f7,f8,f13,f14,f18
		//              unknown: f4,f5,f9,f10,f11,f12,f15b,f16,f17b,rho
		C = 0;
		B = 0;
		A = 0;

		g4[i][iny][1] = g2[i][iny][1];
		g5[i][iny][1] = g6[i][iny][1];
		g9[i][iny][1] = g7[i][iny][1] - (-A * 0 - B);
		g10[i][iny][1] = g8[i][iny][1] - (A * 0 - B);
		g11[i][iny][1] = g13[i][iny][1] - (A * 0 + C);
		g12[i][iny][1] = g14[i][iny][1] - (-A * 0 + C);
		g16[i][iny][1] = g18[i][iny][1] - (C - B);

		g17[i][iny][1] = gn17[i][iny][1];//undo streaming
		g15[i][iny][1] = gn15[i][iny][1];

		g15[i][iny][1] = (1.0 / 2.0)*(g15[i][iny][1] + g17[i][iny][1]);
		g17[i][iny][1] = g15[i][iny][1];

		grho[i][iny][1] = g0[i][iny][1] + g1[i][iny][1] + g2[i][iny][1] + g3[i][iny][1]
			+ g4[i][iny][1] + g5[i][iny][1] + g6[i][iny][1] + g7[i][iny][1]
			+ g8[i][iny][1] + g9[i][iny][1] + g10[i][iny][1] + g11[i][iny][1]
			+ g12[i][iny][1] + g13[i][iny][1] + g14[i][iny][1] + g15[i][iny][1]
			+ g16[i][iny][1] + g17[i][iny][1] + g18[i][iny][1];

		A = (1.0 / 4.0)*(g1[i][iny][1] - g3[i][iny][1]);

		g9[i][iny][1] = g7[i][iny][1] - (-A - B);
		g10[i][iny][1] = g8[i][iny][1] - (A - B);
		g11[i][iny][1] = g13[i][iny][1] - (A + C);
		g12[i][iny][1] = g14[i][iny][1] - (-A + C);
	}
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		//              known: f0,f1,f2,f3,f6,f7,f8,f13,f14,f18
		//              unknown: f4,f5,f9,f10,f11,f12,f15b,f16,f17b,rho
		C = 0;
		B = 0;
		A = 0;

		h4[i][iny][1] = h2[i][iny][1];
		h5[i][iny][1] = h6[i][iny][1];
		h9[i][iny][1] = h7[i][iny][1] - (-A * 0 - B);
		h10[i][iny][1] = h8[i][iny][1] - (A * 0 - B);
		h11[i][iny][1] = h13[i][iny][1] - (A * 0 + C);
		h12[i][iny][1] = h14[i][iny][1] - (-A * 0 + C);
		h16[i][iny][1] = h18[i][iny][1] - (C - B);

		h17[i][iny][1] = hn17[i][iny][1];//undo streaminh
		h15[i][iny][1] = hn15[i][iny][1];

		h15[i][iny][1] = (1.0 / 2.0)*(h15[i][iny][1] + h17[i][iny][1]);
		h17[i][iny][1] = h15[i][iny][1];

		hrho[i][iny][1] = h0[i][iny][1] + h1[i][iny][1] + h2[i][iny][1] + h3[i][iny][1]
			+ h4[i][iny][1] + h5[i][iny][1] + h6[i][iny][1] + h7[i][iny][1]
			+ h8[i][iny][1] + h9[i][iny][1] + h10[i][iny][1] + h11[i][iny][1]
			+ h12[i][iny][1] + h13[i][iny][1] + h14[i][iny][1] + h15[i][iny][1]
			+ h16[i][iny][1] + h17[i][iny][1] + h18[i][iny][1];

		A = (1.0 / 4.0)*(h1[i][iny][1] - h3[i][iny][1]);

		h9[i][iny][1] = h7[i][iny][1] - (-A - B);
		h10[i][iny][1] = h8[i][iny][1] - (A - B);
		h11[i][iny][1] = h13[i][iny][1] - (A + C);
		h12[i][iny][1] = h14[i][iny][1] - (-A + C);
	}

		#pragma omp parallel for collapse(1) private (A,B,C)
	for (i = 2; i <= inx - 1; i++)
	{
		//              known: f0,f1,f2,f3,f6,f7,f8,f13,f14,f18
		//              unknown: f4,f5,f9,f10,f11,f12,f15b,f16,f17b,rho
		C = 0;
		B = 0;
		A = 0;

		p4[i][iny][1] = p2[i][iny][1];
		p5[i][iny][1] = p6[i][iny][1];
		p9[i][iny][1] = p7[i][iny][1] - (-A * 0 - B);
		p10[i][iny][1] = p8[i][iny][1] - (A * 0 - B);
		p11[i][iny][1] = p13[i][iny][1] - (A * 0 + C);
		p12[i][iny][1] = p14[i][iny][1] - (-A * 0 + C);
		p16[i][iny][1] = p18[i][iny][1] - (C - B);

		p17[i][iny][1] = pn17[i][iny][1];//undo streaminp
		p15[i][iny][1] = pn15[i][iny][1];

		p15[i][iny][1] = (1.0 / 2.0)*(p15[i][iny][1] + p17[i][iny][1]);
		p17[i][iny][1] = p15[i][iny][1];

		prho[i][iny][1] = p0[i][iny][1] + p1[i][iny][1] + p2[i][iny][1] + p3[i][iny][1]
			+ p4[i][iny][1] + p5[i][iny][1] + p6[i][iny][1] + p7[i][iny][1]
			+ p8[i][iny][1] + p9[i][iny][1] + p10[i][iny][1] + p11[i][iny][1]
			+ p12[i][iny][1] + p13[i][iny][1] + p14[i][iny][1] + p15[i][iny][1]
			+ p16[i][iny][1] + p17[i][iny][1] + p18[i][iny][1];

		A = (1.0 / 4.0)*(p1[i][iny][1] - p3[i][iny][1]);

		p9[i][iny][1] = p7[i][iny][1] - (-A - B);
		p10[i][iny][1] = p8[i][iny][1] - (A - B);
		p11[i][iny][1] = p13[i][iny][1] - (A + C);
		p12[i][iny][1] = p14[i][iny][1] - (-A + C);
	}


	//Case22 :GH edge as shown in schematic excluding nodes
#pragma omp parallel for collapse(1) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		//		known: f0,f2,f3,f4,f6,f8,f9,f13,f17,f18,rho_in
		//		unknown: f1,f5,f7,f10,f11,f12,f14,f16,f15
		A = 0;//-;//;//1.0/6.0*rho_in*uxwall;
		C = 0;
		B = 1.0 / 4.0*(f2[1][j][1] - f4[1][j][1]);

		f1[1][j][1] = f3[1][j][1];
		f5[1][j][1] = f6[1][j][1];
		f7[1][j][1] = f9[1][j][1] - (A + B);
		f10[1][j][1] = f8[1][j][1] - (A - B);
		f11[1][j][1] = f13[1][j][1] - (A + C);
		f15[1][j][1] = f17[1][j][1] - (B + C);
		f16[1][j][1] = f18[1][j][1] - (C - B);

		//Undo Streaming

		f12[1][j][1] = fn12[1][j][1];
		f14[1][j][1] = fn14[1][j][1];


		f12[1][j][1] = (1.0 / 2.0)*(rho[1][j][2] -
			(f0[1][j][1] + f1[1][j][1] + f2[1][j][1] + f3[1][j][1]
			+ f4[1][j][1] + f5[1][j][1] + f6[1][j][1] + f7[1][j][1]
			+ f8[1][j][1] + f9[1][j][1] + f10[1][j][1] + f11[1][j][1]
			+ f13[1][j][1] + f15[1][j][1] + f16[1][j][1] + f17[1][j][1] + f18[1][j][1]));
		f14[1][j][1] = f12[1][j][1];
	}
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		//		known: f0,f2,f3,f4,f6,f8,f9,f13,f17,f18,rho_in
		//		unknown: f1,f5,f7,f10,f11,f12,f14,f16,f15
		A = 0;//-;//;//1.0/6.0*rho_in*uxwall;
		C = 0;
		B = 1.0 / 4.0*(g2[1][j][1] - g4[1][j][1]);

		g1[1][j][1] = g3[1][j][1];
		g5[1][j][1] = g6[1][j][1];
		g7[1][j][1] = g9[1][j][1] - (A + B);
		g10[1][j][1] = g8[1][j][1] - (A - B);
		g11[1][j][1] = g13[1][j][1] - (A + C);
		g15[1][j][1] = g17[1][j][1] - (B + C);
		g16[1][j][1] = g18[1][j][1] - (C - B);

		//Undo Streaming

		g12[1][j][1] = gn12[1][j][1];
		g14[1][j][1] = gn14[1][j][1];


		g12[1][j][1] = (1.0 / 2.0)*(grho[1][j][2] -
			(g0[1][j][1] + g1[1][j][1] + g2[1][j][1] + g3[1][j][1]
			+ g4[1][j][1] + g5[1][j][1] + g6[1][j][1] + g7[1][j][1]
			+ g8[1][j][1] + g9[1][j][1] + g10[1][j][1] + g11[1][j][1]
			+ g13[1][j][1] + g15[1][j][1] + g16[1][j][1] + g17[1][j][1] + g18[1][j][1]));
		g14[1][j][1] = g12[1][j][1];
	}
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		//		known: f0,f2,f3,f4,f6,f8,f9,f13,f17,f18,rho_in
		//		unknown: f1,f5,f7,f10,f11,f12,f14,f16,f15
		A = 0;//-;//;//1.0/6.0*rho_in*uxwall;
		C = 0;
		B = 1.0 / 4.0*(h2[1][j][1] - h4[1][j][1]);

		h1[1][j][1] = h3[1][j][1];
		h5[1][j][1] = h6[1][j][1];
		h7[1][j][1] = h9[1][j][1] - (A + B);
		h10[1][j][1] = h8[1][j][1] - (A - B);
		h11[1][j][1] = h13[1][j][1] - (A + C);
		h15[1][j][1] = h17[1][j][1] - (B + C);
		h16[1][j][1] = h18[1][j][1] - (C - B);

		//Undo Streaminh

		h12[1][j][1] = hn12[1][j][1];
		h14[1][j][1] = hn14[1][j][1];


		h12[1][j][1] = (1.0 / 2.0)*(hrho[1][j][2] -
			(h0[1][j][1] + h1[1][j][1] + h2[1][j][1] + h3[1][j][1]
			+ h4[1][j][1] + h5[1][j][1] + h6[1][j][1] + h7[1][j][1]
			+ h8[1][j][1] + h9[1][j][1] + h10[1][j][1] + h11[1][j][1]
			+ h13[1][j][1] + h15[1][j][1] + h16[1][j][1] + h17[1][j][1] + h18[1][j][1]));
		h14[1][j][1] = h12[1][j][1];
	}

	#pragma omp parallel for collapse(1) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		//		known: f0,f2,f3,f4,f6,f8,f9,f13,f17,f18,rho_in
		//		unknown: f1,f5,f7,f10,f11,f12,f14,f16,f15
		A = 0;//-;//;//1.0/6.0*rho_in*uxwall;
		C = 0;
		B = 1.0 / 4.0*(p2[1][j][1] - p4[1][j][1]);

		p1[1][j][1] = p3[1][j][1];
		p5[1][j][1] = p6[1][j][1];
		p7[1][j][1] = p9[1][j][1] - (A + B);
		p10[1][j][1] = p8[1][j][1] - (A - B);
		p11[1][j][1] = p13[1][j][1] - (A + C);
		p15[1][j][1] = p17[1][j][1] - (B + C);
		p16[1][j][1] = p18[1][j][1] - (C - B);

		//Undo Streaminp

		p12[1][j][1] = pn12[1][j][1];
		p14[1][j][1] = pn14[1][j][1];


		p12[1][j][1] = (1.0 / 2.0)*(prho[1][j][2] -
			(p0[1][j][1] + p1[1][j][1] + p2[1][j][1] + p3[1][j][1]
			+ p4[1][j][1] + p5[1][j][1] + p6[1][j][1] + p7[1][j][1]
			+ p8[1][j][1] + p9[1][j][1] + p10[1][j][1] + p11[1][j][1]
			+ p13[1][j][1] + p15[1][j][1] + p16[1][j][1] + p17[1][j][1] + p18[1][j][1]));
		p14[1][j][1] = p12[1][j][1];
	}

	//Case 23: FE edge as shown in schematic excluding nodes
#pragma omp parallel for collapse(1) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		//		known: f0,f1,f2,f4,f6,f7,f10,f14,f17,f18,rho_out
		//		unknown: f3,f5,f8,f9,f11,f12,f13,f15,f16
		A = 0;
		C = 0;
		B = 1.0 / 4.0*(f2[inx][j][1] - f4[inx][j][1]);

		f3[inx][j][1] = f1[inx][j][1];
		f5[inx][j][1] = f6[inx][j][1];
		f8[inx][j][1] = f10[inx][j][1] - (-A + B);
		f9[inx][j][1] = f7[inx][j][1] - (-A - B);
		f12[inx][j][1] = f14[inx][j][1] - (-A + C);
		f15[inx][j][1] = f17[inx][j][1] - (B + C);
		f16[inx][j][1] = f18[inx][j][1] - (-B + C);

		//Undo Streaming
		f11[inx][j][1] = fn11[inx][j][1];
		f13[inx][j][1] = fn13[inx][j][1];


		f11[inx][j][1] = (1.0 / 2.0)*(rho[inx][j][2] -
			(f0[inx][j][1] + f1[inx][j][1] + f2[inx][j][1] + f3[inx][j][1]
			+ f4[inx][j][1] + f5[inx][j][1] + f6[inx][j][1] + f7[inx][j][1]
			+ f8[inx][j][1] + f9[inx][j][1] + f10[inx][j][1] + f12[inx][j][1]
			+ f14[inx][j][1] + f15[inx][j][1] + f16[inx][j][1] + f17[inx][j][1]
			+ f18[inx][j][1]));
		f13[inx][j][1] = f11[inx][j][1];
	}
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		//		known: f0,f1,f2,f4,f6,f7,f10,f14,f17,f18,rho_out
		//		unknown: f3,f5,f8,f9,f11,f12,f13,f15,f16
		A = 0;
		C = 0;
		B = 1.0 / 4.0*(g2[inx][j][1] - g4[inx][j][1]);

		g3[inx][j][1] = g1[inx][j][1];
		g5[inx][j][1] = g6[inx][j][1];
		g8[inx][j][1] = g10[inx][j][1] - (-A + B);
		g9[inx][j][1] = g7[inx][j][1] - (-A - B);
		g12[inx][j][1] = g14[inx][j][1] - (-A + C);
		g15[inx][j][1] = g17[inx][j][1] - (B + C);
		g16[inx][j][1] = g18[inx][j][1] - (-B + C);

		//Undo Streaming
		g11[inx][j][1] = gn11[inx][j][1];
		g13[inx][j][1] = gn13[inx][j][1];


		g11[inx][j][1] = (1.0 / 2.0)*(grho[inx][j][2] -
			(g0[inx][j][1] + g1[inx][j][1] + g2[inx][j][1] + g3[inx][j][1]
			+ g4[inx][j][1] + g5[inx][j][1] + g6[inx][j][1] + g7[inx][j][1]
			+ g8[inx][j][1] + g9[inx][j][1] + g10[inx][j][1] + g12[inx][j][1]
			+ g14[inx][j][1] + g15[inx][j][1] + g16[inx][j][1] + g17[inx][j][1]
			+ g18[inx][j][1]));
		g13[inx][j][1] = g11[inx][j][1];
	}

	#pragma omp parallel for collapse(1) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		//		known: f0,f1,f2,f4,f6,f7,f10,f14,f17,f18,rho_out
		//		unknown: f3,f5,f8,f9,f11,f12,f13,f15,f16
		A = 0;
		C = 0;
		B = 1.0 / 4.0*(h2[inx][j][1] - h4[inx][j][1]);

		h3[inx][j][1] = h1[inx][j][1];
		h5[inx][j][1] = h6[inx][j][1];
		h8[inx][j][1] = h10[inx][j][1] - (-A + B);
		h9[inx][j][1] = h7[inx][j][1] - (-A - B);
		h12[inx][j][1] = h14[inx][j][1] - (-A + C);
		h15[inx][j][1] = h17[inx][j][1] - (B + C);
		h16[inx][j][1] = h18[inx][j][1] - (-B + C);

		//Undo Streaminh
		h11[inx][j][1] = hn11[inx][j][1];
		h13[inx][j][1] = hn13[inx][j][1];


		h11[inx][j][1] = (1.0 / 2.0)*(hrho[inx][j][2] -
			(h0[inx][j][1] + h1[inx][j][1] + h2[inx][j][1] + h3[inx][j][1]
			+ h4[inx][j][1] + h5[inx][j][1] + h6[inx][j][1] + h7[inx][j][1]
			+ h8[inx][j][1] + h9[inx][j][1] + h10[inx][j][1] + h12[inx][j][1]
			+ h14[inx][j][1] + h15[inx][j][1] + h16[inx][j][1] + h17[inx][j][1]
			+ h18[inx][j][1]));
		h13[inx][j][1] = h11[inx][j][1];
	}

	#pragma omp parallel for collapse(1) private (A,B,C)
	for (j = 2; j <= iny - 1; j++)
	{
		//		known: f0,f1,f2,f4,f6,f7,f10,f14,f17,f18,rho_out
		//		unknown: f3,f5,f8,f9,f11,f12,f13,f15,f16
		A = 0;
		C = 0;
		B = 1.0 / 4.0*(p2[inx][j][1] - p4[inx][j][1]);

		p3[inx][j][1] = p1[inx][j][1];
		p5[inx][j][1] = p6[inx][j][1];
		p8[inx][j][1] = p10[inx][j][1] - (-A + B);
		p9[inx][j][1] = p7[inx][j][1] - (-A - B);
		p12[inx][j][1] = p14[inx][j][1] - (-A + C);
		p15[inx][j][1] = p17[inx][j][1] - (B + C);
		p16[inx][j][1] = p18[inx][j][1] - (-B + C);

		//Undo Streaminp
		p11[inx][j][1] = pn11[inx][j][1];
		p13[inx][j][1] = pn13[inx][j][1];


		p11[inx][j][1] = (1.0 / 2.0)*(prho[inx][j][2] -
			(p0[inx][j][1] + p1[inx][j][1] + p2[inx][j][1] + p3[inx][j][1]
			+ p4[inx][j][1] + p5[inx][j][1] + p6[inx][j][1] + p7[inx][j][1]
			+ p8[inx][j][1] + p9[inx][j][1] + p10[inx][j][1] + p12[inx][j][1]
			+ p14[inx][j][1] + p15[inx][j][1] + p16[inx][j][1] + p17[inx][j][1]
			+ p18[inx][j][1]));
		p13[inx][j][1] = p11[inx][j][1];
	}
	//Case24:  Edge HH' as shown in schematic excluding nodes
	#pragma omp parallel for collapse(1) private (yc,A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		//		known: f0,f3,f4,f5,f6,f9,f12,f13,f16,f17,rho_in
		//		unknown: f1,f2,f7,f8,f10,f11,f14,f15,f18
		yc = 1;
		A = 0;
		B = 0;
		C = 1.0 / 4.0*(f5[1][yc][p] - f6[1][yc][p]);

		f1[1][yc][p] = f3[1][yc][p];
		f2[1][yc][p] = f4[1][yc][p];
		f7[1][yc][p] = f9[1][yc][p] - (A + B);
		f11[1][yc][p] = f13[1][yc][p] - (A + C);
		f14[1][yc][p] = f12[1][yc][p] - (A - C);
		f15[1][yc][p] = f17[1][yc][p] - (B + C);
		f18[1][yc][p] = f16[1][yc][p] - (B - C);

		//Undo Streaming
		f10[1][yc][p] = fn10[1][yc][p];
		f8[1][yc][p] = fn8[1][yc][p];


		f8[1][yc][p] = (1.0 / 2.0)*(rho[1][yc + 1][p] -
			(f0[1][yc][p] + f1[1][yc][p] + f2[1][yc][p] + f3[1][yc][p]
			+ f4[1][yc][p] + f5[1][yc][p] + f6[1][yc][p] + f7[1][yc][p]
			+ f9[1][yc][p] + f11[1][yc][p] + f12[1][yc][p] + f13[1][yc][p]
			+ f14[1][yc][p] + f15[1][yc][p] + f16[1][yc][p] + f17[1][yc][p]
			+ f18[1][yc][p]));
		f10[1][yc][p] = f8[1][yc][p];
	}

	#pragma omp parallel for collapse(1) private (yc,A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		//		known: f0,f3,f4,f5,f6,f9,f12,f13,f16,f17,rho_in
		//		unknown: f1,f2,f7,f8,f10,f11,f14,f15,f18
		yc = 1;
		A = 0;
		B = 0;
		C = 1.0 / 4.0*(g5[1][yc][p] - g6[1][yc][p]);

		g1[1][yc][p] = g3[1][yc][p];
		g2[1][yc][p] = g4[1][yc][p];
		g7[1][yc][p] = g9[1][yc][p] - (A + B);
		g11[1][yc][p] = g13[1][yc][p] - (A + C);
		g14[1][yc][p] = g12[1][yc][p] - (A - C);
		g15[1][yc][p] = g17[1][yc][p] - (B + C);
		g18[1][yc][p] = g16[1][yc][p] - (B - C);

		//Undo Streaming
		g10[1][yc][p] = gn10[1][yc][p];
		g8[1][yc][p] = gn8[1][yc][p];


		g8[1][yc][p] = (1.0 / 2.0)*(grho[1][yc + 1][p] -
			(g0[1][yc][p] + g1[1][yc][p] + g2[1][yc][p] + g3[1][yc][p]
			+ g4[1][yc][p] + g5[1][yc][p] + g6[1][yc][p] + g7[1][yc][p]
			+ g9[1][yc][p] + g11[1][yc][p] + g12[1][yc][p] + g13[1][yc][p]
			+ g14[1][yc][p] + g15[1][yc][p] + g16[1][yc][p] + g17[1][yc][p]
			+ g18[1][yc][p]));
		g10[1][yc][p] = g8[1][yc][p];
	}


	#pragma omp parallel for collapse(1) private (yc,A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		//		known: f0,f3,f4,f5,f6,f9,f12,f13,f16,f17,rho_in
		//		unknown: f1,f2,f7,f8,f10,f11,f14,f15,f18
		yc = 1;
		A = 0;
		B = 0;
		C = 1.0 / 4.0*(h5[1][yc][p] - h6[1][yc][p]);

		h1[1][yc][p] = h3[1][yc][p];
		h2[1][yc][p] = h4[1][yc][p];
		h7[1][yc][p] = h9[1][yc][p] - (A + B);
		h11[1][yc][p] = h13[1][yc][p] - (A + C);
		h14[1][yc][p] = h12[1][yc][p] - (A - C);
		h15[1][yc][p] = h17[1][yc][p] - (B + C);
		h18[1][yc][p] = h16[1][yc][p] - (B - C);

		//Undo Streaminh
		h10[1][yc][p] = hn10[1][yc][p];
		h8[1][yc][p] = hn8[1][yc][p];


		h8[1][yc][p] = (1.0 / 2.0)*(hrho[1][yc + 1][p] -
			(h0[1][yc][p] + h1[1][yc][p] + h2[1][yc][p] + h3[1][yc][p]
			+ h4[1][yc][p] + h5[1][yc][p] + h6[1][yc][p] + h7[1][yc][p]
			+ h9[1][yc][p] + h11[1][yc][p] + h12[1][yc][p] + h13[1][yc][p]
			+ h14[1][yc][p] + h15[1][yc][p] + h16[1][yc][p] + h17[1][yc][p]
			+ h18[1][yc][p]));
		h10[1][yc][p] = h8[1][yc][p];
	}

	#pragma omp parallel for collapse(1) private (yc,A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		//		known: f0,f3,f4,f5,f6,f9,f12,f13,f16,f17,rho_in
		//		unknown: f1,f2,f7,f8,f10,f11,f14,f15,f18
		yc = 1;
		A = 0;
		B = 0;
		C = 1.0 / 4.0*(p5[1][yc][p] - p6[1][yc][p]);

		p1[1][yc][p] = p3[1][yc][p];
		p2[1][yc][p] = p4[1][yc][p];
		p7[1][yc][p] = p9[1][yc][p] - (A + B);
		p11[1][yc][p] = p13[1][yc][p] - (A + C);
		p14[1][yc][p] = p12[1][yc][p] - (A - C);
		p15[1][yc][p] = p17[1][yc][p] - (B + C);
		p18[1][yc][p] = p16[1][yc][p] - (B - C);

		//Undo Streaminp
		p10[1][yc][p] = pn10[1][yc][p];
		p8[1][yc][p] = pn8[1][yc][p];


		p8[1][yc][p] = (1.0 / 2.0)*(prho[1][yc + 1][p] -
			(p0[1][yc][p] + p1[1][yc][p] + p2[1][yc][p] + p3[1][yc][p]
			+ p4[1][yc][p] + p5[1][yc][p] + p6[1][yc][p] + p7[1][yc][p]
			+ p9[1][yc][p] + p11[1][yc][p] + p12[1][yc][p] + p13[1][yc][p]
			+ p14[1][yc][p] + p15[1][yc][p] + p16[1][yc][p] + p17[1][yc][p]
			+ p18[1][yc][p]));
		p10[1][yc][p] = p8[1][yc][p];
	}

	//Case25 Edge GG' as shown in schematic excluding nodes
#pragma omp parallel for collapse(1) private (A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		//		known: f0,f2,f3,f5,f6,f8,f12,f13,f15,f18,rho_in
		//		unknown: f1,f4,f7,f9,f10,f11,f14,f16,f17
		A = 0;
		C = 1.0 / 4.0*(f5[1][iny][p] - f6[1][iny][p]);
		B = 0;

		f1[1][iny][p] = f3[1][iny][p];
		f4[1][iny][p] = f2[1][iny][p];
		f10[1][iny][p] = f8[1][iny][p] - (A - B);
		f11[1][iny][p] = f13[1][iny][p] - (A + C);
		f14[1][iny][p] = f12[1][iny][p] - (A - C);
		f16[1][iny][p] = f18[1][iny][p] - (-B + C);
		f17[1][iny][p] = f15[1][iny][p] - (-B - C);

		//Undo Streaming
		f7[1][iny][p] = fn7[1][iny][p];
		f9[1][iny][p] = fn9[1][iny][p];


		f7[1][iny][p] = (1.0 / 2.0)*(rho[1][iny - 1][p] -
			(f0[1][iny][p] + f1[1][iny][p] + f2[1][iny][p] + f3[1][iny][p]
			+ f4[1][iny][p] + f5[1][iny][p] + f6[1][iny][p] + f8[1][iny][p]
			+ f10[1][iny][p] + f11[1][iny][p] + f12[1][iny][p] + f13[1][iny][p]
			+ f14[1][iny][p] + f15[1][iny][p] + f16[1][iny][p] + f17[1][iny][p]
			+ f18[1][iny][p]));
		f9[1][iny][p] = f7[1][iny][p];
	}
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		//		known: f0,f2,f3,f5,f6,f8,f12,f13,f15,f18,rho_in
		//		unknown: f1,f4,f7,f9,f10,f11,f14,f16,f17
		A = 0;
		C = 1.0 / 4.0*(g5[1][iny][p] - g6[1][iny][p]);
		B = 0;

		g1[1][iny][p] = g3[1][iny][p];
		g4[1][iny][p] = g2[1][iny][p];
		g10[1][iny][p] = g8[1][iny][p] - (A - B);
		g11[1][iny][p] = g13[1][iny][p] - (A + C);
		g14[1][iny][p] = g12[1][iny][p] - (A - C);
		g16[1][iny][p] = g18[1][iny][p] - (-B + C);
		g17[1][iny][p] = g15[1][iny][p] - (-B - C);

		//Undo Streaming
		g7[1][iny][p] = gn7[1][iny][p];
		g9[1][iny][p] = gn9[1][iny][p];


		g7[1][iny][p] = (1.0 / 2.0)*(grho[1][iny - 1][p] -
			(g0[1][iny][p] + g1[1][iny][p] + g2[1][iny][p] + g3[1][iny][p]
			+ g4[1][iny][p] + g5[1][iny][p] + g6[1][iny][p] + g8[1][iny][p]
			+ g10[1][iny][p] + g11[1][iny][p] + g12[1][iny][p] + g13[1][iny][p]
			+ g14[1][iny][p] + g15[1][iny][p] + g16[1][iny][p] + g17[1][iny][p]
			+ g18[1][iny][p]));
		g9[1][iny][p] = g7[1][iny][p];
	}

	#pragma omp parallel for collapse(1) private (A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		//		known: f0,f2,f3,f5,f6,f8,f12,f13,f15,f18,rho_in
		//		unknown: f1,f4,f7,f9,f10,f11,f14,f16,f17
		A = 0;
		C = 1.0 / 4.0*(h5[1][iny][p] - h6[1][iny][p]);
		B = 0;

		h1[1][iny][p] = h3[1][iny][p];
		h4[1][iny][p] = h2[1][iny][p];
		h10[1][iny][p] = h8[1][iny][p] - (A - B);
		h11[1][iny][p] = h13[1][iny][p] - (A + C);
		h14[1][iny][p] = h12[1][iny][p] - (A - C);
		h16[1][iny][p] = h18[1][iny][p] - (-B + C);
		h17[1][iny][p] = h15[1][iny][p] - (-B - C);

		//Undo Streaminh
		h7[1][iny][p] = hn7[1][iny][p];
		h9[1][iny][p] = hn9[1][iny][p];


		h7[1][iny][p] = (1.0 / 2.0)*(hrho[1][iny - 1][p] -
			(h0[1][iny][p] + h1[1][iny][p] + h2[1][iny][p] + h3[1][iny][p]
			+ h4[1][iny][p] + h5[1][iny][p] + h6[1][iny][p] + h8[1][iny][p]
			+ h10[1][iny][p] + h11[1][iny][p] + h12[1][iny][p] + h13[1][iny][p]
			+ h14[1][iny][p] + h15[1][iny][p] + h16[1][iny][p] + h17[1][iny][p]
			+ h18[1][iny][p]));
		h9[1][iny][p] = h7[1][iny][p];
	}
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		//		known: f0,f2,f3,f5,f6,f8,f12,f13,f15,f18,rho_in
		//		unknown: f1,f4,f7,f9,f10,f11,f14,f16,f17
		A = 0;
		C = 1.0 / 4.0*(p5[1][iny][p] - p6[1][iny][p]);
		B = 0;

		p1[1][iny][p] = p3[1][iny][p];
		p4[1][iny][p] = p2[1][iny][p];
		p10[1][iny][p] = p8[1][iny][p] - (A - B);
		p11[1][iny][p] = p13[1][iny][p] - (A + C);
		p14[1][iny][p] = p12[1][iny][p] - (A - C);
		p16[1][iny][p] = p18[1][iny][p] - (-B + C);
		p17[1][iny][p] = p15[1][iny][p] - (-B - C);

		//Undo Streaminp
		p7[1][iny][p] = pn7[1][iny][p];
		p9[1][iny][p] = pn9[1][iny][p];


		p7[1][iny][p] = (1.0 / 2.0)*(prho[1][iny - 1][p] -
			(p0[1][iny][p] + p1[1][iny][p] + p2[1][iny][p] + p3[1][iny][p]
			+ p4[1][iny][p] + p5[1][iny][p] + p6[1][iny][p] + p8[1][iny][p]
			+ p10[1][iny][p] + p11[1][iny][p] + p12[1][iny][p] + p13[1][iny][p]
			+ p14[1][iny][p] + p15[1][iny][p] + p16[1][iny][p] + p17[1][iny][p]
			+ p18[1][iny][p]));
		p9[1][iny][p] = p7[1][iny][p];
	}

	//Case26 : Edge EE' as shown in schematic  excluding nodes
#pragma omp parallel for collapse(1) private (yc,A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		//		known: f0,f1,f4,f5,f6,f10,f11,f14,f17,f16,rho_in2
		//		unknown: f2,f3,f7,f8,f9,f12,f13,f15,f18
		yc = 1;
		A = 0;//-1;//;//.0/6.0*rho_in2*uxwall;
		B = 0;
		C = 1.0 / 4.0*(f5[inx][yc][p] - f6[inx][yc][p]);

		f3[inx][yc][p] = f1[inx][yc][p];
		f2[inx][yc][p] = f4[inx][yc][p];
		f8[inx][yc][p] = f10[inx][yc][p] - (-A + B);
		f12[inx][yc][p] = f14[inx][yc][p] - (-A + C);
		f13[inx][yc][p] = f11[inx][yc][p] - (-C - A);
		f15[inx][yc][p] = f17[inx][yc][p] - (B + C);
		f18[inx][yc][p] = f16[inx][yc][p] - (B - C);

		//Undo Streaming
		f7[inx][yc][p] = fn7[inx][yc][p];
		f9[inx][yc][p] = fn9[inx][yc][p];


		f7[inx][yc][p] = (1.0 / 2.0)*(rho[inx][yc + 1][p] -
			(f0[inx][yc][p] + f1[inx][yc][p] + f2[inx][yc][p] + f3[inx][yc][p]
			+ f4[inx][yc][p] + f5[inx][yc][p] + f6[inx][yc][p] + f8[inx][yc][p]
			+ f10[inx][yc][p] + f11[inx][yc][p] + f12[inx][yc][p] + f13[inx][yc][p]
			+ f14[inx][yc][p] + f15[inx][yc][p] + f16[inx][yc][p] + f17[inx][yc][p]
			+ f18[inx][yc][p]));
		f9[inx][yc][p] = f7[inx][yc][p];
	}

	#pragma omp parallel for collapse(1) private (yc,A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		//		known: f0,f1,f4,f5,f6,f10,f11,f14,f17,f16,rho_in2
		//		unknown: f2,f3,f7,f8,f9,f12,f13,f15,f18
		yc = 1;
		A = 0;//-1;//;//.0/6.0*rho_in2*uxwall;
		B = 0;
		C = 1.0 / 4.0*(g5[inx][yc][p] - g6[inx][yc][p]);

		g3[inx][yc][p] = g1[inx][yc][p];
		g2[inx][yc][p] = g4[inx][yc][p];
		g8[inx][yc][p] = g10[inx][yc][p] - (-A + B);
		g12[inx][yc][p] = g14[inx][yc][p] - (-A + C);
		g13[inx][yc][p] = g11[inx][yc][p] - (-C - A);
		g15[inx][yc][p] = g17[inx][yc][p] - (B + C);
		g18[inx][yc][p] = g16[inx][yc][p] - (B - C);

		//Undo Streaming
		g7[inx][yc][p] = gn7[inx][yc][p];
		g9[inx][yc][p] = gn9[inx][yc][p];


		g7[inx][yc][p] = (1.0 / 2.0)*(grho[inx][yc + 1][p] -
			(g0[inx][yc][p] + g1[inx][yc][p] + g2[inx][yc][p] + g3[inx][yc][p]
			+ g4[inx][yc][p] + g5[inx][yc][p] + g6[inx][yc][p] + g8[inx][yc][p]
			+ g10[inx][yc][p] + g11[inx][yc][p] + g12[inx][yc][p] + g13[inx][yc][p]
			+ g14[inx][yc][p] + g15[inx][yc][p] + g16[inx][yc][p] + g17[inx][yc][p]
			+ g18[inx][yc][p]));
		g9[inx][yc][p] = g7[inx][yc][p];
	}

	#pragma omp parallel for collapse(1) private (yc,A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		//		known: f0,f1,f4,f5,f6,f10,f11,f14,f17,f16,rho_in2
		//		unknown: f2,f3,f7,f8,f9,f12,f13,f15,f18
		yc = 1;
		A = 0;//-1;//;//.0/6.0*rho_in2*uxwall;
		B = 0;
		C = 1.0 / 4.0*(h5[inx][yc][p] - h6[inx][yc][p]);

		h3[inx][yc][p] = h1[inx][yc][p];
		h2[inx][yc][p] = h4[inx][yc][p];
		h8[inx][yc][p] = h10[inx][yc][p] - (-A + B);
		h12[inx][yc][p] = h14[inx][yc][p] - (-A + C);
		h13[inx][yc][p] = h11[inx][yc][p] - (-C - A);
		h15[inx][yc][p] = h17[inx][yc][p] - (B + C);
		h18[inx][yc][p] = h16[inx][yc][p] - (B - C);

		//Undo Streaminh
		h7[inx][yc][p] = hn7[inx][yc][p];
		h9[inx][yc][p] = hn9[inx][yc][p];


		h7[inx][yc][p] = (1.0 / 2.0)*(hrho[inx][yc + 1][p] -
			(h0[inx][yc][p] + h1[inx][yc][p] + h2[inx][yc][p] + h3[inx][yc][p]
			+ h4[inx][yc][p] + h5[inx][yc][p] + h6[inx][yc][p] + h8[inx][yc][p]
			+ h10[inx][yc][p] + h11[inx][yc][p] + h12[inx][yc][p] + h13[inx][yc][p]
			+ h14[inx][yc][p] + h15[inx][yc][p] + h16[inx][yc][p] + h17[inx][yc][p]
			+ h18[inx][yc][p]));
		h9[inx][yc][p] = h7[inx][yc][p];
	}
	#pragma omp parallel for collapse(1) private (yc,A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		//		known: f0,f1,f4,f5,f6,f10,f11,f14,f17,f16,rho_in2
		//		unknown: f2,f3,f7,f8,f9,f12,f13,f15,f18
		yc = 1;
		A = 0;//-1;//;//.0/6.0*rho_in2*uxwall;
		B = 0;
		C = 1.0 / 4.0*(p5[inx][yc][p] - p6[inx][yc][p]);

		p3[inx][yc][p] = p1[inx][yc][p];
		p2[inx][yc][p] = p4[inx][yc][p];
		p8[inx][yc][p] = p10[inx][yc][p] - (-A + B);
		p12[inx][yc][p] = p14[inx][yc][p] - (-A + C);
		p13[inx][yc][p] = p11[inx][yc][p] - (-C - A);
		p15[inx][yc][p] = p17[inx][yc][p] - (B + C);
		p18[inx][yc][p] = p16[inx][yc][p] - (B - C);

		//Undo Streaminp
		p7[inx][yc][p] = pn7[inx][yc][p];
		p9[inx][yc][p] = pn9[inx][yc][p];


		p7[inx][yc][p] = (1.0 / 2.0)*(prho[inx][yc + 1][p] -
			(p0[inx][yc][p] + p1[inx][yc][p] + p2[inx][yc][p] + p3[inx][yc][p]
			+ p4[inx][yc][p] + p5[inx][yc][p] + p6[inx][yc][p] + p8[inx][yc][p]
			+ p10[inx][yc][p] + p11[inx][yc][p] + p12[inx][yc][p] + p13[inx][yc][p]
			+ p14[inx][yc][p] + p15[inx][yc][p] + p16[inx][yc][p] + p17[inx][yc][p]
			+ p18[inx][yc][p]));
		p9[inx][yc][p] = p7[inx][yc][p];
	}
	//Case27 : Edge FF' as shown in schematic excluding nodes
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		//		known: f0,f1,f2,f5,f6,f7,f11,f14,f15,f18,rho_in2  
		//		unknown: f3,f4,f8,f9,f10,f12,f13,f17,f16
		A = 0;//-1;//;//.0/6.0*rho_in2*uxwall;
		B = 0;
		C = 1.0 / 4.0*(f5[inx][iny][p] - f6[inx][iny][p]);

		f3[inx][iny][p] = f1[inx][iny][p];//-1;//;//.0/3.0*rho_in2*uxwall;
		f4[inx][iny][p] = f2[inx][iny][p];
		f9[inx][iny][p] = f7[inx][iny][p] - (-A - B);
		f12[inx][iny][p] = f14[inx][iny][p] - (-A + C);
		f13[inx][iny][p] = f11[inx][iny][p] - (-A - C);
		f16[inx][iny][p] = f18[inx][iny][p] - (-B + C);
		f17[inx][iny][p] = f15[inx][iny][p] - (-B - C);

		//Undo Streaming
		f8[inx][iny][p] = fn8[inx][iny][p];
		f10[inx][iny][p] = fn10[inx][iny][p];


		f8[inx][iny][p] = (1.0 / 2.0)*(rho[inx][iny - 1][p] -
			(f0[inx][iny][p] + f1[inx][iny][p] + f2[inx][iny][p] + f3[inx][iny][p]
			+ f4[inx][iny][p] + f5[inx][iny][p] + f6[inx][iny][p] + f7[inx][iny][p]
			+ f9[inx][iny][p] + f11[inx][iny][p] + f12[inx][iny][p] + f13[inx][iny][p]
			+ f14[inx][iny][p] + f15[inx][iny][p] + f16[inx][iny][p] + f17[inx][iny][p]
			+ f18[inx][iny][p]));
		f10[inx][iny][p] = f8[inx][iny][p];
	}
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		//		known: f0,f1,f2,f5,f6,f7,f11,f14,f15,f18,rho_in2  
		//		unknown: f3,f4,f8,f9,f10,f12,f13,f17,f16
		A = 0;//-1;//;//.0/6.0*rho_in2*uxwall;
		B = 0;
		C = 1.0 / 4.0*(g5[inx][iny][p] - g6[inx][iny][p]);

		g3[inx][iny][p] = g1[inx][iny][p];//-1;//;//.0/3.0*rho_in2*uxwall;
		g4[inx][iny][p] = g2[inx][iny][p];
		g9[inx][iny][p] = g7[inx][iny][p] - (-A - B);
		g12[inx][iny][p] = g14[inx][iny][p] - (-A + C);
		g13[inx][iny][p] = g11[inx][iny][p] - (-A - C);
		g16[inx][iny][p] = g18[inx][iny][p] - (-B + C);
		g17[inx][iny][p] = g15[inx][iny][p] - (-B - C);

		//Undo Streaming
		g8[inx][iny][p] = gn8[inx][iny][p];
		g10[inx][iny][p] = gn10[inx][iny][p];


		g8[inx][iny][p] = (1.0 / 2.0)*(grho[inx][iny - 1][p] -
			(g0[inx][iny][p] + g1[inx][iny][p] + g2[inx][iny][p] + g3[inx][iny][p]
			+ g4[inx][iny][p] + g5[inx][iny][p] + g6[inx][iny][p] + g7[inx][iny][p]
			+ g9[inx][iny][p] + g11[inx][iny][p] + g12[inx][iny][p] + g13[inx][iny][p]
			+ g14[inx][iny][p] + g15[inx][iny][p] + g16[inx][iny][p] + g17[inx][iny][p]
			+ g18[inx][iny][p]));
		g10[inx][iny][p] = g8[inx][iny][p];
	}

	#pragma omp parallel for collapse(1) private (A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		//		known: f0,f1,f2,f5,f6,f7,f11,f14,f15,f18,rho_in2  
		//		unknown: f3,f4,f8,f9,f10,f12,f13,f17,f16
		A = 0;//-1;//;//.0/6.0*rho_in2*uxwall;
		B = 0;
		C = 1.0 / 4.0*(h5[inx][iny][p] - h6[inx][iny][p]);

		h3[inx][iny][p] = h1[inx][iny][p];//-1;//;//.0/3.0*rho_in2*uxwall;
		h4[inx][iny][p] = h2[inx][iny][p];
		h9[inx][iny][p] = h7[inx][iny][p] - (-A - B);
		h12[inx][iny][p] = h14[inx][iny][p] - (-A + C);
		h13[inx][iny][p] = h11[inx][iny][p] - (-A - C);
		h16[inx][iny][p] = h18[inx][iny][p] - (-B + C);
		h17[inx][iny][p] = h15[inx][iny][p] - (-B - C);

		//Undo Streaminh
		h8[inx][iny][p] = hn8[inx][iny][p];
		h10[inx][iny][p] = hn10[inx][iny][p];


		h8[inx][iny][p] = (1.0 / 2.0)*(hrho[inx][iny - 1][p] -
			(h0[inx][iny][p] + h1[inx][iny][p] + h2[inx][iny][p] + h3[inx][iny][p]
			+ h4[inx][iny][p] + h5[inx][iny][p] + h6[inx][iny][p] + h7[inx][iny][p]
			+ h9[inx][iny][p] + h11[inx][iny][p] + h12[inx][iny][p] + h13[inx][iny][p]
			+ h14[inx][iny][p] + h15[inx][iny][p] + h16[inx][iny][p] + h17[inx][iny][p]
			+ h18[inx][iny][p]));
		h10[inx][iny][p] = h8[inx][iny][p];
	}
#pragma omp parallel for collapse(1) private (A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		//		known: f0,f1,f2,f5,f6,f7,f11,f14,f15,f18,rho_in2  
		//		unknown: f3,f4,f8,f9,f10,f12,f13,f17,f16
		A = 0;//-1;//;//.0/6.0*rho_in2*uxwall;
		B = 0;
		C = 1.0 / 4.0*(p5[inx][iny][p] - p6[inx][iny][p]);

		p3[inx][iny][p] = p1[inx][iny][p];//-1;//;//.0/3.0*rho_in2*uxwall;
		p4[inx][iny][p] = p2[inx][iny][p];
		p9[inx][iny][p] = p7[inx][iny][p] - (-A - B);
		p12[inx][iny][p] = p14[inx][iny][p] - (-A + C);
		p13[inx][iny][p] = p11[inx][iny][p] - (-A - C);
		p16[inx][iny][p] = p18[inx][iny][p] - (-B + C);
		p17[inx][iny][p] = p15[inx][iny][p] - (-B - C);

		//Undo Streaminp
		p8[inx][iny][p] = pn8[inx][iny][p];
		p10[inx][iny][p] = pn10[inx][iny][p];


		p8[inx][iny][p] = (1.0 / 2.0)*(prho[inx][iny - 1][p] -
			(p0[inx][iny][p] + p1[inx][iny][p] + p2[inx][iny][p] + p3[inx][iny][p]
			+ p4[inx][iny][p] + p5[inx][iny][p] + p6[inx][iny][p] + p7[inx][iny][p]
			+ p9[inx][iny][p] + p11[inx][iny][p] + p12[inx][iny][p] + p13[inx][iny][p]
			+ p14[inx][iny][p] + p15[inx][iny][p] + p16[inx][iny][p] + p17[inx][iny][p]
			+ p18[inx][iny][p]));
		p10[inx][iny][p] = p8[inx][iny][p];
	}
	//Case28: edge HD as shown in schematic excluding nodes
#pragma omp parallel for collapse(1) private (yc,A,B,C)
	for (i = 2; i <= ia - 1; i++)
	{
		//		known: f0,f1,f3,f4,f6,f9,f10,f13,f14,f17
		//		unknown: f2,f5,f7,f8,f11,f12,f15,f16b,f18b,rho
		yc = 1;
		B = 0;
		C = 0;
		A = 0;
		f2[i][yc][1] = f4[i][yc][1];
		f5[i][yc][1] = f6[i][yc][1];
		f7[i][yc][1] = f9[i][yc][1] - (A * 0 + B);
		f8[i][yc][1] = f10[i][yc][1] - (-A * 0 + B);
		f11[i][yc][1] = f13[i][yc][1] - (A * 0 + C);
		f12[i][yc][1] = f14[i][yc][1] - (-A * 0 + C);
		f15[i][yc][1] = f17[i][yc][1] - (B + C);

		f18[i][yc][1] = fn18[i][yc][1];//undo streaming
		f16[i][yc][1] = fn16[i][yc][1];
		f16[i][yc][1] = 1.0 / 2.0*(f16[i][yc][1] + f18[i][yc][1]);
		f18[i][yc][1] = f16[i][yc][1];

		rho[i][yc][1] = f0[i][yc][1] + f1[i][yc][1] + f2[i][yc][1] + f3[i][yc][1]
			+ f4[i][yc][1] + f5[i][yc][1] + f6[i][yc][1] + f7[i][yc][1]
			+ f8[i][yc][1] + f9[i][yc][1] + f10[i][yc][1] + f11[i][yc][1]
			+ f12[i][yc][1] + f13[i][yc][1] + f14[i][yc][1] + f15[i][yc][1]
			+ f16[i][yc][1] + f17[i][yc][1] + f18[i][yc][1];

		A = (1.0 / 4.0)*(f1[i][yc][1] - f3[i][yc][1]);
		f7[i][yc][1] = f9[i][yc][1] - (A + B);
		f8[i][yc][1] = f10[i][yc][1] - (-A + B);
		f11[i][yc][1] = f13[i][yc][1] - (A + C);
		f12[i][yc][1] = f14[i][yc][1] - (-A + C);

	}
	#pragma omp parallel for collapse(1) private (yc,A,B,C)
	for (i = 2; i <= ia - 1; i++)
	{
		//		known: f0,f1,f3,f4,f6,f9,f10,f13,f14,f17
		//		unknown: f2,f5,f7,f8,f11,f12,f15,f16b,f18b,rho
		yc = 1;
		B = 0;
		C = 0;
		A = 0;
		g2[i][yc][1] = g4[i][yc][1];
		g5[i][yc][1] = g6[i][yc][1];
		g7[i][yc][1] = g9[i][yc][1] - (A * 0 + B);
		g8[i][yc][1] = g10[i][yc][1] - (-A * 0 + B);
		g11[i][yc][1] = g13[i][yc][1] - (A * 0 + C);
		g12[i][yc][1] = g14[i][yc][1] - (-A * 0 + C);
		g15[i][yc][1] = g17[i][yc][1] - (B + C);

		g18[i][yc][1] = gn18[i][yc][1];//undo streaming
		g16[i][yc][1] = gn16[i][yc][1];
		g16[i][yc][1] = 1.0 / 2.0*(g16[i][yc][1] + g18[i][yc][1]);
		g18[i][yc][1] = g16[i][yc][1];

		grho[i][yc][1] = g0[i][yc][1] + g1[i][yc][1] + g2[i][yc][1] + g3[i][yc][1]
			+ g4[i][yc][1] + g5[i][yc][1] + g6[i][yc][1] + g7[i][yc][1]
			+ g8[i][yc][1] + g9[i][yc][1] + g10[i][yc][1] + g11[i][yc][1]
			+ g12[i][yc][1] + g13[i][yc][1] + g14[i][yc][1] + g15[i][yc][1]
			+ g16[i][yc][1] + g17[i][yc][1] + g18[i][yc][1];

		A = (1.0 / 4.0)*(g1[i][yc][1] - g3[i][yc][1]);
		g7[i][yc][1] = g9[i][yc][1] - (A + B);
		g8[i][yc][1] = g10[i][yc][1] - (-A + B);
		g11[i][yc][1] = g13[i][yc][1] - (A + C);
		g12[i][yc][1] = g14[i][yc][1] - (-A + C);

	}

	#pragma omp parallel for collapse(1) private (yc,A,B,C)
	for (i = 2; i <= ia - 1; i++)
	{
		//		known: f0,f1,f3,f4,f6,f9,f10,f13,f14,f17
		//		unknown: f2,f5,f7,f8,f11,f12,f15,f16b,f18b,rho
		yc = 1;
		B = 0;
		C = 0;
		A = 0;
		h2[i][yc][1] = h4[i][yc][1];
		h5[i][yc][1] = h6[i][yc][1];
		h7[i][yc][1] = h9[i][yc][1] - (A * 0 + B);
		h8[i][yc][1] = h10[i][yc][1] - (-A * 0 + B);
		h11[i][yc][1] = h13[i][yc][1] - (A * 0 + C);
		h12[i][yc][1] = h14[i][yc][1] - (-A * 0 + C);
		h15[i][yc][1] = h17[i][yc][1] - (B + C);

		h18[i][yc][1] = hn18[i][yc][1];//undo streaminh
		h16[i][yc][1] = hn16[i][yc][1];
		h16[i][yc][1] = 1.0 / 2.0*(h16[i][yc][1] + h18[i][yc][1]);
		h18[i][yc][1] = h16[i][yc][1];

		hrho[i][yc][1] = h0[i][yc][1] + h1[i][yc][1] + h2[i][yc][1] + h3[i][yc][1]
			+ h4[i][yc][1] + h5[i][yc][1] + h6[i][yc][1] + h7[i][yc][1]
			+ h8[i][yc][1] + h9[i][yc][1] + h10[i][yc][1] + h11[i][yc][1]
			+ h12[i][yc][1] + h13[i][yc][1] + h14[i][yc][1] + h15[i][yc][1]
			+ h16[i][yc][1] + h17[i][yc][1] + h18[i][yc][1];

		A = (1.0 / 4.0)*(h1[i][yc][1] - h3[i][yc][1]);
		h7[i][yc][1] = h9[i][yc][1] - (A + B);
		h8[i][yc][1] = h10[i][yc][1] - (-A + B);
		h11[i][yc][1] = h13[i][yc][1] - (A + C);
		h12[i][yc][1] = h14[i][yc][1] - (-A + C);

	}
	#pragma omp parallel for collapse(1) private (yc,A,B,C)
	for (i = 2; i <= ia - 1; i++)
	{
		//		known: f0,f1,f3,f4,f6,f9,f10,f13,f14,f17
		//		unknown: f2,f5,f7,f8,f11,f12,f15,f16b,f18b,rho
		yc = 1;
		B = 0;
		C = 0;
		A = 0;
		p2[i][yc][1] = p4[i][yc][1];
		p5[i][yc][1] = p6[i][yc][1];
		p7[i][yc][1] = p9[i][yc][1] - (A * 0 + B);
		p8[i][yc][1] = p10[i][yc][1] - (-A * 0 + B);
		p11[i][yc][1] = p13[i][yc][1] - (A * 0 + C);
		p12[i][yc][1] = p14[i][yc][1] - (-A * 0 + C);
		p15[i][yc][1] = p17[i][yc][1] - (B + C);

		p18[i][yc][1] = pn18[i][yc][1];//undo streaminp
		p16[i][yc][1] = pn16[i][yc][1];
		p16[i][yc][1] = 1.0 / 2.0*(p16[i][yc][1] + p18[i][yc][1]);
		p18[i][yc][1] = p16[i][yc][1];

		prho[i][yc][1] = p0[i][yc][1] + p1[i][yc][1] + p2[i][yc][1] + p3[i][yc][1]
			+ p4[i][yc][1] + p5[i][yc][1] + p6[i][yc][1] + p7[i][yc][1]
			+ p8[i][yc][1] + p9[i][yc][1] + p10[i][yc][1] + p11[i][yc][1]
			+ p12[i][yc][1] + p13[i][yc][1] + p14[i][yc][1] + p15[i][yc][1]
			+ p16[i][yc][1] + p17[i][yc][1] + p18[i][yc][1];

		A = (1.0 / 4.0)*(p1[i][yc][1] - p3[i][yc][1]);
		p7[i][yc][1] = p9[i][yc][1] - (A + B);
		p8[i][yc][1] = p10[i][yc][1] - (-A + B);
		p11[i][yc][1] = p13[i][yc][1] - (A + C);
		p12[i][yc][1] = p14[i][yc][1] - (-A + C);

	}


	//Case29: Edge CE as shown in the schematic excluding edges
#pragma omp parallel for collapse(1) private (yc,A,B,C)
	for (i = ib + 1; i <= inx - 1; i++)
	{
		//		known: f0,f1,f3,f4,f6,f9,f10,f13,f14,f17
		//		unknown: f2,f5,f7,f8,f11,f12,f15,f16b,f18b,rho
		yc = 1;
		B = 0;
		C = 0;
		A = 0;
		f2[i][yc][1] = f4[i][yc][1];
		f5[i][yc][1] = f6[i][yc][1];
		f7[i][yc][1] = f9[i][yc][1] - (A * 0 + B);
		f8[i][yc][1] = f10[i][yc][1] - (-A * 0 + B);
		f11[i][yc][1] = f13[i][yc][1] - (A * 0 + C);
		f12[i][yc][1] = f14[i][yc][1] - (-A * 0 + C);
		f15[i][yc][1] = f17[i][yc][1] - (B + C);

		f18[i][yc][1] = fn18[i][yc][1];//undo streaming
		f16[i][yc][1] = fn16[i][yc][1];
		f16[i][yc][1] = 1.0 / 2.0*(f16[i][yc][1] + f18[i][yc][1]);
		f18[i][yc][1] = f16[i][yc][1];

		rho[i][yc][1] = f0[i][yc][1] + f1[i][yc][1] + f2[i][yc][1] + f3[i][yc][1]
			+ f4[i][yc][1] + f5[i][yc][1] + f6[i][yc][1] + f7[i][yc][1]
			+ f8[i][yc][1] + f9[i][yc][1] + f10[i][yc][1] + f11[i][yc][1]
			+ f12[i][yc][1] + f13[i][yc][1] + f14[i][yc][1] + f15[i][yc][1]
			+ f16[i][yc][1] + f17[i][yc][1] + f18[i][yc][1];

		A = 1.0 / 4.0*(f1[i][yc][1] - f3[i][yc][1]);

		f7[i][yc][1] = f9[i][yc][1] - (A + B);
		f8[i][yc][1] = f10[i][yc][1] - (-A + B);
		f11[i][yc][1] = f13[i][yc][1] - (A + C);
		f12[i][yc][1] = f14[i][yc][1] - (-A + C);

	}
	#pragma omp parallel for collapse(1) private (yc,A,B,C)
	for (i = ib + 1; i <= inx - 1; i++)
	{
		//		known: f0,f1,f3,f4,f6,f9,f10,f13,f14,f17
		//		unknown: f2,f5,f7,f8,f11,f12,f15,f16b,f18b,rho
		yc = 1;
		B = 0;
		C = 0;
		A = 0;
		g2[i][yc][1] = g4[i][yc][1];
		g5[i][yc][1] = g6[i][yc][1];
		g7[i][yc][1] = g9[i][yc][1] - (A * 0 + B);
		g8[i][yc][1] = g10[i][yc][1] - (-A * 0 + B);
		g11[i][yc][1] = g13[i][yc][1] - (A * 0 + C);
		g12[i][yc][1] = g14[i][yc][1] - (-A * 0 + C);
		g15[i][yc][1] = g17[i][yc][1] - (B + C);

		g18[i][yc][1] = gn18[i][yc][1];//undo streaming
		g16[i][yc][1] = gn16[i][yc][1];
		g16[i][yc][1] = 1.0 / 2.0*(g16[i][yc][1] + g18[i][yc][1]);
		g18[i][yc][1] = g16[i][yc][1];

		grho[i][yc][1] = g0[i][yc][1] + g1[i][yc][1] + g2[i][yc][1] + g3[i][yc][1]
			+ g4[i][yc][1] + g5[i][yc][1] + g6[i][yc][1] + g7[i][yc][1]
			+ g8[i][yc][1] + g9[i][yc][1] + g10[i][yc][1] + g11[i][yc][1]
			+ g12[i][yc][1] + g13[i][yc][1] + g14[i][yc][1] + g15[i][yc][1]
			+ g16[i][yc][1] + g17[i][yc][1] + g18[i][yc][1];

		A = 1.0 / 4.0*(g1[i][yc][1] - g3[i][yc][1]);

		g7[i][yc][1] = g9[i][yc][1] - (A + B);
		g8[i][yc][1] = g10[i][yc][1] - (-A + B);
		g11[i][yc][1] = g13[i][yc][1] - (A + C);
		g12[i][yc][1] = g14[i][yc][1] - (-A + C);

	}

	#pragma omp parallel for collapse(1) private (yc,A,B,C)
	for (i = ib + 1; i <= inx - 1; i++)
	{
		//		known: f0,f1,f3,f4,f6,f9,f10,f13,f14,f17
		//		unknown: f2,f5,f7,f8,f11,f12,f15,f16b,f18b,rho
		yc = 1;
		B = 0;
		C = 0;
		A = 0;
		h2[i][yc][1] = h4[i][yc][1];
		h5[i][yc][1] = h6[i][yc][1];
		h7[i][yc][1] = h9[i][yc][1] - (A * 0 + B);
		h8[i][yc][1] = h10[i][yc][1] - (-A * 0 + B);
		h11[i][yc][1] = h13[i][yc][1] - (A * 0 + C);
		h12[i][yc][1] = h14[i][yc][1] - (-A * 0 + C);
		h15[i][yc][1] = h17[i][yc][1] - (B + C);

		h18[i][yc][1] = hn18[i][yc][1];//undo streaminh
		h16[i][yc][1] = hn16[i][yc][1];
		h16[i][yc][1] = 1.0 / 2.0*(h16[i][yc][1] + h18[i][yc][1]);
		h18[i][yc][1] = h16[i][yc][1];

		hrho[i][yc][1] = h0[i][yc][1] + h1[i][yc][1] + h2[i][yc][1] + h3[i][yc][1]
			+ h4[i][yc][1] + h5[i][yc][1] + h6[i][yc][1] + h7[i][yc][1]
			+ h8[i][yc][1] + h9[i][yc][1] + h10[i][yc][1] + h11[i][yc][1]
			+ h12[i][yc][1] + h13[i][yc][1] + h14[i][yc][1] + h15[i][yc][1]
			+ h16[i][yc][1] + h17[i][yc][1] + h18[i][yc][1];

		A = 1.0 / 4.0*(h1[i][yc][1] - h3[i][yc][1]);

		h7[i][yc][1] = h9[i][yc][1] - (A + B);
		h8[i][yc][1] = h10[i][yc][1] - (-A + B);
		h11[i][yc][1] = h13[i][yc][1] - (A + C);
		h12[i][yc][1] = h14[i][yc][1] - (-A + C);

	}
	#pragma omp parallel for collapse(1) private (yc,A,B,C)
	for (i = ib + 1; i <= inx - 1; i++)
	{
		//		known: f0,f1,f3,f4,f6,f9,f10,f13,f14,f17
		//		unknown: f2,f5,f7,f8,f11,f12,f15,f16b,f18b,rho
		yc = 1;
		B = 0;
		C = 0;
		A = 0;
		p2[i][yc][1] = p4[i][yc][1];
		p5[i][yc][1] = p6[i][yc][1];
		p7[i][yc][1] = p9[i][yc][1] - (A * 0 + B);
		p8[i][yc][1] = p10[i][yc][1] - (-A * 0 + B);
		p11[i][yc][1] = p13[i][yc][1] - (A * 0 + C);
		p12[i][yc][1] = p14[i][yc][1] - (-A * 0 + C);
		p15[i][yc][1] = p17[i][yc][1] - (B + C);

		p18[i][yc][1] = pn18[i][yc][1];//undo streaminp
		p16[i][yc][1] = pn16[i][yc][1];
		p16[i][yc][1] = 1.0 / 2.0*(p16[i][yc][1] + p18[i][yc][1]);
		p18[i][yc][1] = p16[i][yc][1];

		prho[i][yc][1] = p0[i][yc][1] + p1[i][yc][1] + p2[i][yc][1] + p3[i][yc][1]
			+ p4[i][yc][1] + p5[i][yc][1] + p6[i][yc][1] + p7[i][yc][1]
			+ p8[i][yc][1] + p9[i][yc][1] + p10[i][yc][1] + p11[i][yc][1]
			+ p12[i][yc][1] + p13[i][yc][1] + p14[i][yc][1] + p15[i][yc][1]
			+ p16[i][yc][1] + p17[i][yc][1] + p18[i][yc][1];

		A = 1.0 / 4.0*(p1[i][yc][1] - p3[i][yc][1]);

		p7[i][yc][1] = p9[i][yc][1] - (A + B);
		p8[i][yc][1] = p10[i][yc][1] - (-A + B);
		p11[i][yc][1] = p13[i][yc][1] - (A + C);
		p12[i][yc][1] = p14[i][yc][1] - (-A + C);

	}
	// All corners as shown in the schematic 
	// Total no.of.corners 16

	
	#pragma omp parallel sections private(A,xc,yc)
	{
		#pragma omp section
		{
			//Corner1: G'as shown in the schematic 

	//		known: f0,f2,f3,f5,f8,f12,f15,rho_in
			A = 0;

			f1[1][iny][nz] = f3[1][iny][nz] + A;
			f4[1][iny][nz] = f2[1][iny][nz];
			f6[1][iny][nz] = f5[1][iny][nz];
			f10[1][iny][nz] = f8[1][iny][nz];
			f14[1][iny][nz] = f12[1][iny][nz];
			f17[1][iny][nz] = f15[1][iny][nz];

			f7[1][iny][nz] = fn7[1][iny][nz];//undo streaming
			f9[1][iny][nz] = fn9[1][iny][nz];
			f11[1][iny][nz] = fn11[1][iny][nz];
			f13[1][iny][nz] = fn13[1][iny][nz];
			f16[1][iny][nz] = fn16[1][iny][nz];
			f18[1][iny][nz] = fn18[1][iny][nz];

			f9[1][iny][nz] = 1.0 / 2.0*(f9[1][iny][nz] + f7[1][iny][nz]);
			f7[1][iny][nz] = f9[1][iny][nz];
			f11[1][iny][nz] = 1.0 / 2.0*(f11[1][iny][nz] + f13[1][iny][nz]);
			f13[1][iny][nz] = f11[1][iny][nz];

			f16[1][iny][nz] = (1.0 / 2.0)*(rho[1][iny][nz - 1] -
				(f0[1][iny][nz] + f1[1][iny][nz] + f2[1][iny][nz] + f3[1][iny][nz]
				+ f4[1][iny][nz] + f5[1][iny][nz] + f6[1][iny][nz] + f7[1][iny][nz]
				+ f8[1][iny][nz] + f9[1][iny][nz] + f10[1][iny][nz] + f11[1][iny][nz]
				+ f12[1][iny][nz] + f13[1][iny][nz] + f14[1][iny][nz] + f15[1][iny][nz]
				+ f17[1][iny][nz]));

			f18[1][iny][nz] = f16[1][iny][nz];
		}
		#pragma omp section
		{
			//Corner1: G'as shown in the schematic 

	//		known: f0,f2,f3,f5,f8,f12,f15,rho_in
			A = 0;

			g1[1][iny][nz] = g3[1][iny][nz] + A;
			g4[1][iny][nz] = g2[1][iny][nz];
			g6[1][iny][nz] = g5[1][iny][nz];
			g10[1][iny][nz] = g8[1][iny][nz];
			g14[1][iny][nz] = g12[1][iny][nz];
			g17[1][iny][nz] = g15[1][iny][nz];

			g7[1][iny][nz] = gn7[1][iny][nz];//undo streaming
			g9[1][iny][nz] = gn9[1][iny][nz];
			g11[1][iny][nz] = gn11[1][iny][nz];
			g13[1][iny][nz] = gn13[1][iny][nz];
			g16[1][iny][nz] = gn16[1][iny][nz];
			g18[1][iny][nz] = gn18[1][iny][nz];

			g9[1][iny][nz] = 1.0 / 2.0*(g9[1][iny][nz] + g7[1][iny][nz]);
			g7[1][iny][nz] = g9[1][iny][nz];
			g11[1][iny][nz] = 1.0 / 2.0*(g11[1][iny][nz] + g13[1][iny][nz]);
			g13[1][iny][nz] = g11[1][iny][nz];

			g16[1][iny][nz] = (1.0 / 2.0)*(grho[1][iny][nz - 1] -
				(g0[1][iny][nz] + g1[1][iny][nz] + g2[1][iny][nz] + g3[1][iny][nz]
				+ g4[1][iny][nz] + g5[1][iny][nz] + g6[1][iny][nz] + g7[1][iny][nz]
				+ g8[1][iny][nz] + g9[1][iny][nz] + g10[1][iny][nz] + g11[1][iny][nz]
				+ g12[1][iny][nz] + g13[1][iny][nz] + g14[1][iny][nz] + g15[1][iny][nz]
				+ g17[1][iny][nz]));

			g18[1][iny][nz] = g16[1][iny][nz];
		}
		#pragma omp section
		{
			//Corner1: G'as shown in the schematic 

	//		known: f0,f2,f3,f5,f8,f12,f15,rho_in
			A = 0;

			h1[1][iny][nz] = h3[1][iny][nz] + A;
			h4[1][iny][nz] = h2[1][iny][nz];
			h6[1][iny][nz] = h5[1][iny][nz];
			h10[1][iny][nz] = h8[1][iny][nz];
			h14[1][iny][nz] = h12[1][iny][nz];
			h17[1][iny][nz] = h15[1][iny][nz];

			h7[1][iny][nz] = hn7[1][iny][nz];//undo streaminh
			h9[1][iny][nz] = hn9[1][iny][nz];
			h11[1][iny][nz] = hn11[1][iny][nz];
			h13[1][iny][nz] = hn13[1][iny][nz];
			h16[1][iny][nz] = hn16[1][iny][nz];
			h18[1][iny][nz] = hn18[1][iny][nz];

			h9[1][iny][nz] = 1.0 / 2.0*(h9[1][iny][nz] + h7[1][iny][nz]);
			h7[1][iny][nz] = h9[1][iny][nz];
			h11[1][iny][nz] = 1.0 / 2.0*(h11[1][iny][nz] + h13[1][iny][nz]);
			h13[1][iny][nz] = h11[1][iny][nz];

			h16[1][iny][nz] = (1.0 / 2.0)*(hrho[1][iny][nz - 1] -
				(h0[1][iny][nz] + h1[1][iny][nz] + h2[1][iny][nz] + h3[1][iny][nz]
				+ h4[1][iny][nz] + h5[1][iny][nz] + h6[1][iny][nz] + h7[1][iny][nz]
				+ h8[1][iny][nz] + h9[1][iny][nz] + h10[1][iny][nz] + h11[1][iny][nz]
				+ h12[1][iny][nz] + h13[1][iny][nz] + h14[1][iny][nz] + h15[1][iny][nz]
				+ h17[1][iny][nz]));

			h18[1][iny][nz] = h16[1][iny][nz];
		}
		#pragma omp section
		{
			//Corner1: G'as shown in the schematic 

	//		known: f0,f2,f3,f5,f8,f12,f15,rho_in
			A = 0;

			p1[1][iny][nz] = p3[1][iny][nz] + A;
			p4[1][iny][nz] = p2[1][iny][nz];
			p6[1][iny][nz] = p5[1][iny][nz];
			p10[1][iny][nz] = p8[1][iny][nz];
			p14[1][iny][nz] = p12[1][iny][nz];
			p17[1][iny][nz] = p15[1][iny][nz];

			p7[1][iny][nz] = pn7[1][iny][nz];//undo streaminp
			p9[1][iny][nz] = pn9[1][iny][nz];
			p11[1][iny][nz] = pn11[1][iny][nz];
			p13[1][iny][nz] = pn13[1][iny][nz];
			p16[1][iny][nz] = pn16[1][iny][nz];
			p18[1][iny][nz] = pn18[1][iny][nz];

			p9[1][iny][nz] = 1.0 / 2.0*(p9[1][iny][nz] + p7[1][iny][nz]);
			p7[1][iny][nz] = p9[1][iny][nz];
			p11[1][iny][nz] = 1.0 / 2.0*(p11[1][iny][nz] + p13[1][iny][nz]);
			p13[1][iny][nz] = p11[1][iny][nz];

			p16[1][iny][nz] = (1.0 / 2.0)*(prho[1][iny][nz - 1] -
				(p0[1][iny][nz] + p1[1][iny][nz] + p2[1][iny][nz] + p3[1][iny][nz]
				+ p4[1][iny][nz] + p5[1][iny][nz] + p6[1][iny][nz] + p7[1][iny][nz]
				+ p8[1][iny][nz] + p9[1][iny][nz] + p10[1][iny][nz] + p11[1][iny][nz]
				+ p12[1][iny][nz] + p13[1][iny][nz] + p14[1][iny][nz] + p15[1][iny][nz]
				+ p17[1][iny][nz]));

			p18[1][iny][nz] = p16[1][iny][nz];
		}
		#pragma omp section
		{
			//Corner2 : F'as shown in the schematic

			A = 0;

			f3[inx][iny][nz] = f1[inx][iny][nz] - A;
			f4[inx][iny][nz] = f2[inx][iny][nz];
			f6[inx][iny][nz] = f5[inx][iny][nz];
			f17[inx][iny][nz] = f15[inx][iny][nz];
			f13[inx][iny][nz] = f11[inx][iny][nz];
			f9[inx][iny][nz] = f7[inx][iny][nz];

			f10[inx][iny][nz] = fn10[inx][iny][nz];//undo streaming
			f8[inx][iny][nz] = fn8[inx][iny][nz];
			f12[inx][iny][nz] = fn12[inx][iny][nz];
			f14[inx][iny][nz] = fn14[inx][iny][nz];
			f16[inx][iny][nz] = fn16[inx][iny][nz];
			f18[inx][iny][nz] = fn18[inx][iny][nz];

			f10[inx][iny][nz] = 1.0 / 2.0*(f10[inx][iny][nz] + f8[inx][iny][nz]);
			f8[inx][iny][nz] = f10[inx][iny][nz];
			f12[inx][iny][nz] = 1.0 / 2.0*(f12[inx][iny][nz] + f14[inx][iny][nz]);
			f14[inx][iny][nz] = f12[inx][iny][nz];

			f16[inx][iny][nz] = (1.0 / 2.0)*(rho[inx][iny][nz - 1] -
				(f0[inx][iny][nz] + f1[inx][iny][nz] + f2[inx][iny][nz] + f3[inx][iny][nz]
				+ f4[inx][iny][nz] + f5[inx][iny][nz] + f6[inx][iny][nz] + f7[inx][iny][nz]
				+ f8[inx][iny][nz] + f9[inx][iny][nz] + f10[inx][iny][nz] + f11[inx][iny][nz]
				+ f12[inx][iny][nz] + f13[inx][iny][nz] + f14[inx][iny][nz] + f15[inx][iny][nz]
				+ f17[inx][iny][nz]));

			f18[inx][iny][nz] = f16[inx][iny][nz];
		}
		#pragma omp section
		{
			//Corner2 : F'as shown in the schematic

			A = 0;

			g3[inx][iny][nz] = g1[inx][iny][nz] - A;
			g4[inx][iny][nz] = g2[inx][iny][nz];
			g6[inx][iny][nz] = g5[inx][iny][nz];
			g17[inx][iny][nz] = g15[inx][iny][nz];
			g13[inx][iny][nz] = g11[inx][iny][nz];
			g9[inx][iny][nz] = g7[inx][iny][nz];

			g10[inx][iny][nz] = gn10[inx][iny][nz];//undo streaming
			g8[inx][iny][nz] = gn8[inx][iny][nz];
			g12[inx][iny][nz] = gn12[inx][iny][nz];
			g14[inx][iny][nz] = gn14[inx][iny][nz];
			g16[inx][iny][nz] = gn16[inx][iny][nz];
			g18[inx][iny][nz] = gn18[inx][iny][nz];

			g10[inx][iny][nz] = 1.0 / 2.0*(g10[inx][iny][nz] + g8[inx][iny][nz]);
			g8[inx][iny][nz] = g10[inx][iny][nz];
			g12[inx][iny][nz] = 1.0 / 2.0*(g12[inx][iny][nz] + g14[inx][iny][nz]);
			g14[inx][iny][nz] = g12[inx][iny][nz];

			g16[inx][iny][nz] = (1.0 / 2.0)*(grho[inx][iny][nz - 1] -
				(g0[inx][iny][nz] + g1[inx][iny][nz] + g2[inx][iny][nz] + g3[inx][iny][nz]
				+ g4[inx][iny][nz] + g5[inx][iny][nz] + g6[inx][iny][nz] + g7[inx][iny][nz]
				+ g8[inx][iny][nz] + g9[inx][iny][nz] + g10[inx][iny][nz] + g11[inx][iny][nz]
				+ g12[inx][iny][nz] + g13[inx][iny][nz] + g14[inx][iny][nz] + g15[inx][iny][nz]
				+ g17[inx][iny][nz]));

			g18[inx][iny][nz] = g16[inx][iny][nz];
		}
		#pragma omp section
		{
			//Corner2 : F'as shown in the schematic

			A = 0;

			h3[inx][iny][nz] = h1[inx][iny][nz] - A;
			h4[inx][iny][nz] = h2[inx][iny][nz];
			h6[inx][iny][nz] = h5[inx][iny][nz];
			h17[inx][iny][nz] = h15[inx][iny][nz];
			h13[inx][iny][nz] = h11[inx][iny][nz];
			h9[inx][iny][nz] = h7[inx][iny][nz];

			h10[inx][iny][nz] = hn10[inx][iny][nz];//undo streaminh
			h8[inx][iny][nz] = hn8[inx][iny][nz];
			h12[inx][iny][nz] = hn12[inx][iny][nz];
			h14[inx][iny][nz] = hn14[inx][iny][nz];
			h16[inx][iny][nz] = hn16[inx][iny][nz];
			h18[inx][iny][nz] = hn18[inx][iny][nz];

			h10[inx][iny][nz] = 1.0 / 2.0*(h10[inx][iny][nz] + h8[inx][iny][nz]);
			h8[inx][iny][nz] = h10[inx][iny][nz];
			h12[inx][iny][nz] = 1.0 / 2.0*(h12[inx][iny][nz] + h14[inx][iny][nz]);
			h14[inx][iny][nz] = h12[inx][iny][nz];

			h16[inx][iny][nz] = (1.0 / 2.0)*(hrho[inx][iny][nz - 1] -
				(h0[inx][iny][nz] + h1[inx][iny][nz] + h2[inx][iny][nz] + h3[inx][iny][nz]
				+ h4[inx][iny][nz] + h5[inx][iny][nz] + h6[inx][iny][nz] + h7[inx][iny][nz]
				+ h8[inx][iny][nz] + h9[inx][iny][nz] + h10[inx][iny][nz] + h11[inx][iny][nz]
				+ h12[inx][iny][nz] + h13[inx][iny][nz] + h14[inx][iny][nz] + h15[inx][iny][nz]
				+ h17[inx][iny][nz]));

			h18[inx][iny][nz] = h16[inx][iny][nz];
		}
		#pragma omp section
		{
			//Corner2 : F'as shown in the schematic

			A = 0;

			p3[inx][iny][nz] = p1[inx][iny][nz] - A;
			p4[inx][iny][nz] = p2[inx][iny][nz];
			p6[inx][iny][nz] = p5[inx][iny][nz];
			p17[inx][iny][nz] = p15[inx][iny][nz];
			p13[inx][iny][nz] = p11[inx][iny][nz];
			p9[inx][iny][nz] = p7[inx][iny][nz];

			p10[inx][iny][nz] = pn10[inx][iny][nz];//undo streaminp
			p8[inx][iny][nz] = pn8[inx][iny][nz];
			p12[inx][iny][nz] = pn12[inx][iny][nz];
			p14[inx][iny][nz] = pn14[inx][iny][nz];
			p16[inx][iny][nz] = pn16[inx][iny][nz];
			p18[inx][iny][nz] = pn18[inx][iny][nz];

			p10[inx][iny][nz] = 1.0 / 2.0*(p10[inx][iny][nz] + p8[inx][iny][nz]);
			p8[inx][iny][nz] = p10[inx][iny][nz];
			p12[inx][iny][nz] = 1.0 / 2.0*(p12[inx][iny][nz] + p14[inx][iny][nz]);
			p14[inx][iny][nz] = p12[inx][iny][nz];

			p16[inx][iny][nz] = (1.0 / 2.0)*(prho[inx][iny][nz - 1] -
				(p0[inx][iny][nz] + p1[inx][iny][nz] + p2[inx][iny][nz] + p3[inx][iny][nz]
				+ p4[inx][iny][nz] + p5[inx][iny][nz] + p6[inx][iny][nz] + p7[inx][iny][nz]
				+ p8[inx][iny][nz] + p9[inx][iny][nz] + p10[inx][iny][nz] + p11[inx][iny][nz]
				+ p12[inx][iny][nz] + p13[inx][iny][nz] + p14[inx][iny][nz] + p15[inx][iny][nz]
				+ p17[inx][iny][nz]));

			p18[inx][iny][nz] = p16[inx][iny][nz];
		}
		#pragma omp section
		{
			// Corner3: H'as shown in the schematic

			A = 0;

			yc = 1;

			f1[1][yc][nz] = f3[1][yc][nz] + A;
			f2[1][yc][nz] = f4[1][yc][nz];
			f6[1][yc][nz] = f5[1][yc][nz];
			f7[1][yc][nz] = f9[1][yc][nz];
			f18[1][yc][nz] = f16[1][yc][nz];
			f14[1][yc][nz] = f12[1][yc][nz];

			f10[1][yc][nz] = fn10[1][yc][nz];//undo streaming
			f8[1][yc][nz] = fn8[1][yc][nz];
			f11[1][yc][nz] = fn11[1][yc][nz];
			f13[1][yc][nz] = fn13[1][yc][nz];
			f15[1][yc][nz] = fn15[1][yc][nz];
			f17[1][yc][nz] = fn17[1][yc][nz];

			f10[1][yc][nz] = 1.0 / 2.0*(f10[1][yc][nz] + f8[1][yc][nz]);
			f8[1][yc][nz] = f10[1][yc][nz];
			f11[1][yc][nz] = 1.0 / 2.0*(f11[1][yc][nz] + f13[1][yc][nz]);
			f13[1][yc][nz] = f11[1][yc][nz];

			f15[1][yc][nz] = (1.0 / 2.0)*(rho[1][yc][nz - 1] -
				(f0[1][yc][nz] + f1[1][yc][nz] + f2[1][yc][nz] + f3[1][yc][nz]
				+ f4[1][yc][nz] + f5[1][yc][nz] + f6[1][yc][nz] + f7[1][yc][nz]
				+ f8[1][yc][nz] + f9[1][yc][nz] + f10[1][yc][nz] + f11[1][yc][nz]
				+ f12[1][yc][nz] + f13[1][yc][nz] + f14[1][yc][nz] + f16[1][yc][nz]
				+ f18[1][yc][nz]));

			f17[1][yc][nz] = f15[1][yc][nz];
		}
		#pragma omp section
		{
			// Corner3: H'as shown in the schematic

			A = 0;

			yc = 1;

			g1[1][yc][nz] = g3[1][yc][nz] + A;
			g2[1][yc][nz] = g4[1][yc][nz];
			g6[1][yc][nz] = g5[1][yc][nz];
			g7[1][yc][nz] = g9[1][yc][nz];
			g18[1][yc][nz] = g16[1][yc][nz];
			g14[1][yc][nz] = g12[1][yc][nz];

			g10[1][yc][nz] = gn10[1][yc][nz];//undo streaming
			g8[1][yc][nz] = gn8[1][yc][nz];
			g11[1][yc][nz] = gn11[1][yc][nz];
			g13[1][yc][nz] = gn13[1][yc][nz];
			g15[1][yc][nz] = gn15[1][yc][nz];
			g17[1][yc][nz] = gn17[1][yc][nz];

			g10[1][yc][nz] = 1.0 / 2.0*(g10[1][yc][nz] + g8[1][yc][nz]);
			g8[1][yc][nz] = g10[1][yc][nz];
			g11[1][yc][nz] = 1.0 / 2.0*(g11[1][yc][nz] + g13[1][yc][nz]);
			g13[1][yc][nz] = g11[1][yc][nz];

			g15[1][yc][nz] = (1.0 / 2.0)*(grho[1][yc][nz - 1] -
				(g0[1][yc][nz] + g1[1][yc][nz] + g2[1][yc][nz] + g3[1][yc][nz]
				+ g4[1][yc][nz] + g5[1][yc][nz] + g6[1][yc][nz] + g7[1][yc][nz]
				+ g8[1][yc][nz] + g9[1][yc][nz] + g10[1][yc][nz] + g11[1][yc][nz]
				+ g12[1][yc][nz] + g13[1][yc][nz] + g14[1][yc][nz] + g16[1][yc][nz]
				+ g18[1][yc][nz]));

			g17[1][yc][nz] = g15[1][yc][nz];
		}

		#pragma omp section
		{
			// Corner3: H'as shown in the schematic

			A = 0;

			yc = 1;

			h1[1][yc][nz] = h3[1][yc][nz] + A;
			h2[1][yc][nz] = h4[1][yc][nz];
			h6[1][yc][nz] = h5[1][yc][nz];
			h7[1][yc][nz] = h9[1][yc][nz];
			h18[1][yc][nz] = h16[1][yc][nz];
			h14[1][yc][nz] = h12[1][yc][nz];

			h10[1][yc][nz] = hn10[1][yc][nz];//undo streaminh
			h8[1][yc][nz] = hn8[1][yc][nz];
			h11[1][yc][nz] = hn11[1][yc][nz];
			h13[1][yc][nz] = hn13[1][yc][nz];
			h15[1][yc][nz] = hn15[1][yc][nz];
			h17[1][yc][nz] = hn17[1][yc][nz];

			h10[1][yc][nz] = 1.0 / 2.0*(h10[1][yc][nz] + h8[1][yc][nz]);
			h8[1][yc][nz] = h10[1][yc][nz];
			h11[1][yc][nz] = 1.0 / 2.0*(h11[1][yc][nz] + h13[1][yc][nz]);
			h13[1][yc][nz] = h11[1][yc][nz];

			h15[1][yc][nz] = (1.0 / 2.0)*(hrho[1][yc][nz - 1] -
				(h0[1][yc][nz] + h1[1][yc][nz] + h2[1][yc][nz] + h3[1][yc][nz]
				+ h4[1][yc][nz] + h5[1][yc][nz] + h6[1][yc][nz] + h7[1][yc][nz]
				+ h8[1][yc][nz] + h9[1][yc][nz] + h10[1][yc][nz] + h11[1][yc][nz]
				+ h12[1][yc][nz] + h13[1][yc][nz] + h14[1][yc][nz] + h16[1][yc][nz]
				+ h18[1][yc][nz]));

			h17[1][yc][nz] = h15[1][yc][nz];
		}
		#pragma omp section
		{
			// Corner3: H'as shown in the schematic

			A = 0;

			yc = 1;

			p1[1][yc][nz] = p3[1][yc][nz] + A;
			p2[1][yc][nz] = p4[1][yc][nz];
			p6[1][yc][nz] = p5[1][yc][nz];
			p7[1][yc][nz] = p9[1][yc][nz];
			p18[1][yc][nz] = p16[1][yc][nz];
			p14[1][yc][nz] = p12[1][yc][nz];

			p10[1][yc][nz] = pn10[1][yc][nz];//undo streaminp
			p8[1][yc][nz] = pn8[1][yc][nz];
			p11[1][yc][nz] = pn11[1][yc][nz];
			p13[1][yc][nz] = pn13[1][yc][nz];
			p15[1][yc][nz] = pn15[1][yc][nz];
			p17[1][yc][nz] = pn17[1][yc][nz];

			p10[1][yc][nz] = 1.0 / 2.0*(p10[1][yc][nz] + p8[1][yc][nz]);
			p8[1][yc][nz] = p10[1][yc][nz];
			p11[1][yc][nz] = 1.0 / 2.0*(p11[1][yc][nz] + p13[1][yc][nz]);
			p13[1][yc][nz] = p11[1][yc][nz];

			p15[1][yc][nz] = (1.0 / 2.0)*(prho[1][yc][nz - 1] -
				(p0[1][yc][nz] + p1[1][yc][nz] + p2[1][yc][nz] + p3[1][yc][nz]
				+ p4[1][yc][nz] + p5[1][yc][nz] + p6[1][yc][nz] + p7[1][yc][nz]
				+ p8[1][yc][nz] + p9[1][yc][nz] + p10[1][yc][nz] + p11[1][yc][nz]
				+ p12[1][yc][nz] + p13[1][yc][nz] + p14[1][yc][nz] + p16[1][yc][nz]
				+ p18[1][yc][nz]));

			p17[1][yc][nz] = p15[1][yc][nz];
		}
		#pragma omp section
		{
			// Corner4: E'as shown in the schematic

			A = 0;//1.0/3.0*rho_in2*uxwall;

			yc = 1;

			f3[inx][yc][nz] = f1[inx][yc][nz] - A;
			f2[inx][yc][nz] = f4[inx][yc][nz];
			f6[inx][yc][nz] = f5[inx][yc][nz];
			f8[inx][yc][nz] = f10[inx][yc][nz];
			f13[inx][yc][nz] = f11[inx][yc][nz];
			f18[inx][yc][nz] = f16[inx][yc][nz];

			f9[inx][yc][nz] = fn9[inx][yc][nz];//undo streaming
			f7[inx][yc][nz] = fn7[inx][yc][nz];
			f12[inx][yc][nz] = fn12[inx][yc][nz];
			f14[inx][yc][nz] = fn14[inx][yc][nz];
			f15[inx][yc][nz] = fn15[inx][yc][nz];
			f17[inx][yc][nz] = fn17[inx][yc][nz];

			f9[inx][yc][nz] = 1.0 / 2.0*(f9[inx][yc][nz] + f7[inx][yc][nz]);
			f7[inx][yc][nz] = f9[inx][yc][nz];

			f12[inx][yc][nz] = 1.0 / 2.0*(f12[inx][yc][nz] + f14[inx][yc][nz]);
			f14[inx][yc][nz] = f12[inx][yc][nz];

			f15[inx][yc][nz] = (1.0 / 2.0)*(rho[inx][yc][nz - 1] - (f0[inx][yc][nz]
				+ f1[inx][yc][nz] + f2[inx][yc][nz] + f3[inx][yc][nz]
				+ f4[inx][yc][nz] + f5[inx][yc][nz] + f6[inx][yc][nz]
				+ f7[inx][yc][nz] + f8[inx][yc][nz] + f9[inx][yc][nz]
				+ f10[inx][yc][nz] + f11[inx][yc][nz] + f12[inx][yc][nz]
				+ f13[inx][yc][nz] + f14[inx][yc][nz] + f16[inx][yc][nz]
				+ f18[inx][yc][nz]));

			f17[inx][yc][nz] = f15[inx][yc][nz];

		}
		#pragma omp section
		{
			// Corner4: E'as shown in the schematic

			A = 0;//1.0/3.0*rho_in2*uxwall;

			yc = 1;

			g3[inx][yc][nz] = g1[inx][yc][nz] - A;
			g2[inx][yc][nz] = g4[inx][yc][nz];
			g6[inx][yc][nz] = g5[inx][yc][nz];
			g8[inx][yc][nz] = g10[inx][yc][nz];
			g13[inx][yc][nz] = g11[inx][yc][nz];
			g18[inx][yc][nz] = g16[inx][yc][nz];

			g9[inx][yc][nz] = gn9[inx][yc][nz];//undo streaming
			g7[inx][yc][nz] = gn7[inx][yc][nz];
			g12[inx][yc][nz] = gn12[inx][yc][nz];
			g14[inx][yc][nz] = gn14[inx][yc][nz];
			g15[inx][yc][nz] = gn15[inx][yc][nz];
			g17[inx][yc][nz] = gn17[inx][yc][nz];

			g9[inx][yc][nz] = 1.0 / 2.0*(g9[inx][yc][nz] + g7[inx][yc][nz]);
			g7[inx][yc][nz] = g9[inx][yc][nz];

			g12[inx][yc][nz] = 1.0 / 2.0*(g12[inx][yc][nz] + g14[inx][yc][nz]);
			g14[inx][yc][nz] = g12[inx][yc][nz];

			g15[inx][yc][nz] = (1.0 / 2.0)*(grho[inx][yc][nz - 1] - (g0[inx][yc][nz]
				+ g1[inx][yc][nz] + g2[inx][yc][nz] + g3[inx][yc][nz]
				+ g4[inx][yc][nz] + g5[inx][yc][nz] + g6[inx][yc][nz]
				+ g7[inx][yc][nz] + g8[inx][yc][nz] + g9[inx][yc][nz]
				+ g10[inx][yc][nz] + g11[inx][yc][nz] + g12[inx][yc][nz]
				+ g13[inx][yc][nz] + g14[inx][yc][nz] + g16[inx][yc][nz]
				+ g18[inx][yc][nz]));

			g17[inx][yc][nz] = g15[inx][yc][nz];

		}
		#pragma omp section
		{
			// Corner4: E'as shown in the schematic

			A = 0;//1.0/3.0*rho_in2*uxwall;

			yc = 1;

			h3[inx][yc][nz] = h1[inx][yc][nz] - A;
			h2[inx][yc][nz] = h4[inx][yc][nz];
			h6[inx][yc][nz] = h5[inx][yc][nz];
			h8[inx][yc][nz] = h10[inx][yc][nz];
			h13[inx][yc][nz] = h11[inx][yc][nz];
			h18[inx][yc][nz] = h16[inx][yc][nz];

			h9[inx][yc][nz] = hn9[inx][yc][nz];//undo streaminh
			h7[inx][yc][nz] = hn7[inx][yc][nz];
			h12[inx][yc][nz] = hn12[inx][yc][nz];
			h14[inx][yc][nz] = hn14[inx][yc][nz];
			h15[inx][yc][nz] = hn15[inx][yc][nz];
			h17[inx][yc][nz] = hn17[inx][yc][nz];

			h9[inx][yc][nz] = 1.0 / 2.0*(h9[inx][yc][nz] + h7[inx][yc][nz]);
			h7[inx][yc][nz] = h9[inx][yc][nz];

			h12[inx][yc][nz] = 1.0 / 2.0*(h12[inx][yc][nz] + h14[inx][yc][nz]);
			h14[inx][yc][nz] = h12[inx][yc][nz];

			h15[inx][yc][nz] = (1.0 / 2.0)*(hrho[inx][yc][nz - 1] - (h0[inx][yc][nz]
				+ h1[inx][yc][nz] + h2[inx][yc][nz] + h3[inx][yc][nz]
				+ h4[inx][yc][nz] + h5[inx][yc][nz] + h6[inx][yc][nz]
				+ h7[inx][yc][nz] + h8[inx][yc][nz] + h9[inx][yc][nz]
				+ h10[inx][yc][nz] + h11[inx][yc][nz] + h12[inx][yc][nz]
				+ h13[inx][yc][nz] + h14[inx][yc][nz] + h16[inx][yc][nz]
				+ h18[inx][yc][nz]));

			h17[inx][yc][nz] = h15[inx][yc][nz];

		}
		#pragma omp section
		{
			// Corner4: E'as shown in the schematic

			A = 0;//1.0/3.0*rho_in2*uxwall;

			yc = 1;

			p3[inx][yc][nz] = p1[inx][yc][nz] - A;
			p2[inx][yc][nz] = p4[inx][yc][nz];
			p6[inx][yc][nz] = p5[inx][yc][nz];
			p8[inx][yc][nz] = p10[inx][yc][nz];
			p13[inx][yc][nz] = p11[inx][yc][nz];
			p18[inx][yc][nz] = p16[inx][yc][nz];

			p9[inx][yc][nz] = pn9[inx][yc][nz];//undo streaminp
			p7[inx][yc][nz] = pn7[inx][yc][nz];
			p12[inx][yc][nz] = pn12[inx][yc][nz];
			p14[inx][yc][nz] = pn14[inx][yc][nz];
			p15[inx][yc][nz] = pn15[inx][yc][nz];
			p17[inx][yc][nz] = pn17[inx][yc][nz];

			p9[inx][yc][nz] = 1.0 / 2.0*(p9[inx][yc][nz] + p7[inx][yc][nz]);
			p7[inx][yc][nz] = p9[inx][yc][nz];

			p12[inx][yc][nz] = 1.0 / 2.0*(p12[inx][yc][nz] + p14[inx][yc][nz]);
			p14[inx][yc][nz] = p12[inx][yc][nz];

			p15[inx][yc][nz] = (1.0 / 2.0)*(prho[inx][yc][nz - 1] - (p0[inx][yc][nz]
				+ p1[inx][yc][nz] + p2[inx][yc][nz] + p3[inx][yc][nz]
				+ p4[inx][yc][nz] + p5[inx][yc][nz] + p6[inx][yc][nz]
				+ p7[inx][yc][nz] + p8[inx][yc][nz] + p9[inx][yc][nz]
				+ p10[inx][yc][nz] + p11[inx][yc][nz] + p12[inx][yc][nz]
				+ p13[inx][yc][nz] + p14[inx][yc][nz] + p16[inx][yc][nz]
				+ p18[inx][yc][nz]));

			p17[inx][yc][nz] = p15[inx][yc][nz];

		}
	}

	#pragma omp parallel sections private(A,xc,yc)
	{
		#pragma omp section
		{
			//Corner5: G as shown in the schematic

			A = 0;// 1.0/3.0*rho_in*uxwall;

			f1[1][iny][1] = f3[1][iny][1] + A;
			f4[1][iny][1] = f2[1][iny][1];
			f5[1][iny][1] = f6[1][iny][1];
			f16[1][iny][1] = f18[1][iny][1];
			f11[1][iny][1] = f13[1][iny][1];
			f10[1][iny][1] = f8[1][iny][1];

			f9[1][iny][1] = fn9[1][iny][1];//undo streaming
			f7[1][iny][1] = fn7[1][iny][1];
			f14[1][iny][1] = fn14[1][iny][1];
			f12[1][iny][1] = fn12[1][iny][1];
			f15[1][iny][1] = fn15[1][iny][1];
			f17[1][iny][1] = fn17[1][iny][1];

			f9[1][iny][1] = 1.0 / 2.0*(f9[1][iny][1] + f7[1][iny][1]);
			f7[1][iny][1] = f9[1][iny][1];
			f14[1][iny][1] = 1.0 / 2.0*(f14[1][iny][1] + f12[1][iny][1]);
			f12[1][iny][1] = f14[1][iny][1];

			f15[1][iny][1] = (1.0 / 2.0)*(rho[1][iny][2] - (f0[1][iny][1]
				+ f1[1][iny][1] + f2[1][iny][1] + f3[1][iny][1]
				+ f4[1][iny][1] + f5[1][iny][1] + f6[1][iny][1]
				+ f7[1][iny][1] + f8[1][iny][1] + f9[1][iny][1]
				+ f10[1][iny][1] + f11[1][iny][1] + f12[1][iny][1]
				+ f13[1][iny][1] + f14[1][iny][1] + f16[1][iny][1]
				+ f18[1][iny][1]));

			f17[1][iny][1] = f15[1][iny][1];
		}
		#pragma omp section
		{
			//Corner5: G as shown in the schematic

			A = 0;// 1.0/3.0*rho_in*uxwall;

			g1[1][iny][1] = g3[1][iny][1] + A;
			g4[1][iny][1] = g2[1][iny][1];
			g5[1][iny][1] = g6[1][iny][1];
			g16[1][iny][1] = g18[1][iny][1];
			g11[1][iny][1] = g13[1][iny][1];
			g10[1][iny][1] = g8[1][iny][1];

			g9[1][iny][1] = gn9[1][iny][1];//undo streaming
			g7[1][iny][1] = gn7[1][iny][1];
			g14[1][iny][1] = gn14[1][iny][1];
			g12[1][iny][1] = gn12[1][iny][1];
			g15[1][iny][1] = gn15[1][iny][1];
			g17[1][iny][1] = gn17[1][iny][1];

			g9[1][iny][1] = 1.0 / 2.0*(g9[1][iny][1] + g7[1][iny][1]);
			g7[1][iny][1] = g9[1][iny][1];
			g14[1][iny][1] = 1.0 / 2.0*(g14[1][iny][1] + g12[1][iny][1]);
			g12[1][iny][1] = g14[1][iny][1];

			g15[1][iny][1] = (1.0 / 2.0)*(grho[1][iny][2] - (g0[1][iny][1]
				+ g1[1][iny][1] + g2[1][iny][1] + g3[1][iny][1]
				+ g4[1][iny][1] + g5[1][iny][1] + g6[1][iny][1]
				+ g7[1][iny][1] + g8[1][iny][1] + g9[1][iny][1]
				+ g10[1][iny][1] + g11[1][iny][1] + g12[1][iny][1]
				+ g13[1][iny][1] + g14[1][iny][1] + g16[1][iny][1]
				+ g18[1][iny][1]));

			g17[1][iny][1] = g15[1][iny][1];
		}

		#pragma omp section
		{
			//Corner5: G as shown in the schematic

			A = 0;// 1.0/3.0*rho_in*uxwall;

			h1[1][iny][1] = h3[1][iny][1] + A;
			h4[1][iny][1] = h2[1][iny][1];
			h5[1][iny][1] = h6[1][iny][1];
			h16[1][iny][1] = h18[1][iny][1];
			h11[1][iny][1] = h13[1][iny][1];
			h10[1][iny][1] = h8[1][iny][1];

			h9[1][iny][1] = hn9[1][iny][1];//undo streaminh
			h7[1][iny][1] = hn7[1][iny][1];
			h14[1][iny][1] = hn14[1][iny][1];
			h12[1][iny][1] = hn12[1][iny][1];
			h15[1][iny][1] = hn15[1][iny][1];
			h17[1][iny][1] = hn17[1][iny][1];

			h9[1][iny][1] = 1.0 / 2.0*(h9[1][iny][1] + h7[1][iny][1]);
			h7[1][iny][1] = h9[1][iny][1];
			h14[1][iny][1] = 1.0 / 2.0*(h14[1][iny][1] + h12[1][iny][1]);
			h12[1][iny][1] = h14[1][iny][1];

			h15[1][iny][1] = (1.0 / 2.0)*(hrho[1][iny][2] - (h0[1][iny][1]
				+ h1[1][iny][1] + h2[1][iny][1] + h3[1][iny][1]
				+ h4[1][iny][1] + h5[1][iny][1] + h6[1][iny][1]
				+ h7[1][iny][1] + h8[1][iny][1] + h9[1][iny][1]
				+ h10[1][iny][1] + h11[1][iny][1] + h12[1][iny][1]
				+ h13[1][iny][1] + h14[1][iny][1] + h16[1][iny][1]
				+ h18[1][iny][1]));

			h17[1][iny][1] = h15[1][iny][1];
		}
		#pragma omp section
		{
			//Corner5: G as shown in the schematic

			A = 0;// 1.0/3.0*rho_in*uxwall;

			p1[1][iny][1] = p3[1][iny][1] + A;
			p4[1][iny][1] = p2[1][iny][1];
			p5[1][iny][1] = p6[1][iny][1];
			p16[1][iny][1] = p18[1][iny][1];
			p11[1][iny][1] = p13[1][iny][1];
			p10[1][iny][1] = p8[1][iny][1];

			p9[1][iny][1] = pn9[1][iny][1];//undo streaminp
			p7[1][iny][1] = pn7[1][iny][1];
			p14[1][iny][1] = pn14[1][iny][1];
			p12[1][iny][1] = pn12[1][iny][1];
			p15[1][iny][1] = pn15[1][iny][1];
			p17[1][iny][1] = pn17[1][iny][1];

			p9[1][iny][1] = 1.0 / 2.0*(p9[1][iny][1] + p7[1][iny][1]);
			p7[1][iny][1] = p9[1][iny][1];
			p14[1][iny][1] = 1.0 / 2.0*(p14[1][iny][1] + p12[1][iny][1]);
			p12[1][iny][1] = p14[1][iny][1];

			p15[1][iny][1] = (1.0 / 2.0)*(prho[1][iny][2] - (p0[1][iny][1]
				+ p1[1][iny][1] + p2[1][iny][1] + p3[1][iny][1]
				+ p4[1][iny][1] + p5[1][iny][1] + p6[1][iny][1]
				+ p7[1][iny][1] + p8[1][iny][1] + p9[1][iny][1]
				+ p10[1][iny][1] + p11[1][iny][1] + p12[1][iny][1]
				+ p13[1][iny][1] + p14[1][iny][1] + p16[1][iny][1]
				+ p18[1][iny][1]));

			p17[1][iny][1] = p15[1][iny][1];
		}
		#pragma omp section
		{
			//Corner6: F as shown in the schematic

			A = 0;

			f3[inx][iny][1] = f1[inx][iny][1] - A;
			f4[inx][iny][1] = f2[inx][iny][1];
			f5[inx][iny][1] = f6[inx][iny][1];
			f9[inx][iny][1] = f7[inx][iny][1];
			f12[inx][iny][1] = f14[inx][iny][1];
			f16[inx][iny][1] = f18[inx][iny][1];

			f10[inx][iny][1] = fn10[inx][iny][1];//undo streaming
			f8[inx][iny][1] = fn8[inx][iny][1];
			f11[inx][iny][1] = fn11[inx][iny][1];
			f13[inx][iny][1] = fn13[inx][iny][1];
			f15[inx][iny][1] = fn15[inx][iny][1];
			f17[inx][iny][1] = fn17[inx][iny][1];

			f10[inx][iny][1] = 1.0 / 2.0*(f10[inx][iny][1] + f8[inx][iny][1]);
			f8[inx][iny][1] = f10[inx][iny][1];
			f11[inx][iny][1] = 1.0 / 2.0*(f11[inx][iny][1] + f13[inx][iny][1]);
			f13[inx][iny][1] = f11[inx][iny][1];

			f15[inx][iny][1] = (1.0 / 2.0)*(rho[inx][iny][2] - (f0[inx][iny][1]
				+ f1[inx][iny][1] + f2[inx][iny][1] + f3[inx][iny][1]
				+ f4[inx][iny][1] + f5[inx][iny][1] + f6[inx][iny][1]
				+ f7[inx][iny][1] + f8[inx][iny][1] + f9[inx][iny][1]
				+ f10[inx][iny][1] + f11[inx][iny][1] + f12[inx][iny][1]
				+ f13[inx][iny][1] + f14[inx][iny][1] + f16[inx][iny][1]
				+ f18[inx][iny][1]));

			f17[inx][iny][1] = f15[inx][iny][1];
		}
		#pragma omp section
		{
			//Corner6: F as shown in the schematic

			A = 0;

			g3[inx][iny][1] = g1[inx][iny][1] - A;
			g4[inx][iny][1] = g2[inx][iny][1];
			g5[inx][iny][1] = g6[inx][iny][1];
			g9[inx][iny][1] = g7[inx][iny][1];
			g12[inx][iny][1] = g14[inx][iny][1];
			g16[inx][iny][1] = g18[inx][iny][1];

			g10[inx][iny][1] = gn10[inx][iny][1];//undo streaming
			g8[inx][iny][1] = gn8[inx][iny][1];
			g11[inx][iny][1] = gn11[inx][iny][1];
			g13[inx][iny][1] = gn13[inx][iny][1];
			g15[inx][iny][1] = gn15[inx][iny][1];
			g17[inx][iny][1] = gn17[inx][iny][1];

			g10[inx][iny][1] = 1.0 / 2.0*(g10[inx][iny][1] + g8[inx][iny][1]);
			g8[inx][iny][1] = g10[inx][iny][1];
			g11[inx][iny][1] = 1.0 / 2.0*(g11[inx][iny][1] + g13[inx][iny][1]);
			g13[inx][iny][1] = g11[inx][iny][1];

			g15[inx][iny][1] = (1.0 / 2.0)*(grho[inx][iny][2] - (g0[inx][iny][1]
				+ g1[inx][iny][1] + g2[inx][iny][1] + g3[inx][iny][1]
				+ g4[inx][iny][1] + g5[inx][iny][1] + g6[inx][iny][1]
				+ g7[inx][iny][1] + g8[inx][iny][1] + g9[inx][iny][1]
				+ g10[inx][iny][1] + g11[inx][iny][1] + g12[inx][iny][1]
				+ g13[inx][iny][1] + g14[inx][iny][1] + g16[inx][iny][1]
				+ g18[inx][iny][1]));

			g17[inx][iny][1] = g15[inx][iny][1];
		}
		#pragma omp section
		{
			//Corner6: F as shown in the schematic

			A = 0;

			h3[inx][iny][1] = h1[inx][iny][1] - A;
			h4[inx][iny][1] = h2[inx][iny][1];
			h5[inx][iny][1] = h6[inx][iny][1];
			h9[inx][iny][1] = h7[inx][iny][1];
			h12[inx][iny][1] = h14[inx][iny][1];
			h16[inx][iny][1] = h18[inx][iny][1];

			h10[inx][iny][1] = hn10[inx][iny][1];//undo streaminh
			h8[inx][iny][1] = hn8[inx][iny][1];
			h11[inx][iny][1] = hn11[inx][iny][1];
			h13[inx][iny][1] = hn13[inx][iny][1];
			h15[inx][iny][1] = hn15[inx][iny][1];
			h17[inx][iny][1] = hn17[inx][iny][1];

			h10[inx][iny][1] = 1.0 / 2.0*(h10[inx][iny][1] + h8[inx][iny][1]);
			h8[inx][iny][1] = h10[inx][iny][1];
			h11[inx][iny][1] = 1.0 / 2.0*(h11[inx][iny][1] + h13[inx][iny][1]);
			h13[inx][iny][1] = h11[inx][iny][1];

			h15[inx][iny][1] = (1.0 / 2.0)*(hrho[inx][iny][2] - (h0[inx][iny][1]
				+ h1[inx][iny][1] + h2[inx][iny][1] + h3[inx][iny][1]
				+ h4[inx][iny][1] + h5[inx][iny][1] + h6[inx][iny][1]
				+ h7[inx][iny][1] + h8[inx][iny][1] + h9[inx][iny][1]
				+ h10[inx][iny][1] + h11[inx][iny][1] + h12[inx][iny][1]
				+ h13[inx][iny][1] + h14[inx][iny][1] + h16[inx][iny][1]
				+ h18[inx][iny][1]));

			h17[inx][iny][1] = h15[inx][iny][1];
		}
		#pragma omp section
		{
			//Corner6: F as shown in the schematic

			A = 0;

			p3[inx][iny][1] = p1[inx][iny][1] - A;
			p4[inx][iny][1] = p2[inx][iny][1];
			p5[inx][iny][1] = p6[inx][iny][1];
			p9[inx][iny][1] = p7[inx][iny][1];
			p12[inx][iny][1] = p14[inx][iny][1];
			p16[inx][iny][1] = p18[inx][iny][1];

			p10[inx][iny][1] = pn10[inx][iny][1];//undo streaminp
			p8[inx][iny][1] = pn8[inx][iny][1];
			p11[inx][iny][1] = pn11[inx][iny][1];
			p13[inx][iny][1] = pn13[inx][iny][1];
			p15[inx][iny][1] = pn15[inx][iny][1];
			p17[inx][iny][1] = pn17[inx][iny][1];

			p10[inx][iny][1] = 1.0 / 2.0*(p10[inx][iny][1] + p8[inx][iny][1]);
			p8[inx][iny][1] = p10[inx][iny][1];
			p11[inx][iny][1] = 1.0 / 2.0*(p11[inx][iny][1] + p13[inx][iny][1]);
			p13[inx][iny][1] = p11[inx][iny][1];

			p15[inx][iny][1] = (1.0 / 2.0)*(prho[inx][iny][2] - (p0[inx][iny][1]
				+ p1[inx][iny][1] + p2[inx][iny][1] + p3[inx][iny][1]
				+ p4[inx][iny][1] + p5[inx][iny][1] + p6[inx][iny][1]
				+ p7[inx][iny][1] + p8[inx][iny][1] + p9[inx][iny][1]
				+ p10[inx][iny][1] + p11[inx][iny][1] + p12[inx][iny][1]
				+ p13[inx][iny][1] + p14[inx][iny][1] + p16[inx][iny][1]
				+ p18[inx][iny][1]));

			p17[inx][iny][1] = p15[inx][iny][1];
		}
		#pragma omp section
		{
			//Corner7: H as shown in the schematic
			//
			A = 0;

			yc = 1;

			f1[1][yc][1] = f3[1][yc][1] + A;
			f2[1][yc][1] = f4[1][yc][1];
			f5[1][yc][1] = f6[1][yc][1];
			f7[1][yc][1] = f9[1][yc][1];
			f11[1][yc][1] = f13[1][yc][1];
			f15[1][yc][1] = f17[1][yc][1];

			f10[1][yc][1] = fn10[1][yc][1];//undo streaming
			f8[1][yc][1] = fn8[1][yc][1];
			f12[1][yc][1] = fn12[1][yc][1];
			f14[1][yc][1] = fn14[1][yc][1];
			f16[1][yc][1] = fn16[1][yc][1];
			f18[1][yc][1] = fn18[1][yc][1];

			f10[1][yc][1] = 1.0 / 2.0*(f10[1][yc][1] + f8[1][yc][1]);
			f8[1][yc][1] = f10[1][yc][1];
			f12[1][yc][1] = 1.0 / 2.0*(f12[1][yc][1] + f14[1][yc][1]);
			f14[1][yc][1] = f12[1][yc][1];

			f16[1][yc][1] = (1.0 / 2.0)*(rho[1][yc][2] - (f0[1][yc][1]
				+ f1[1][yc][1] + f2[1][yc][1] + f3[1][yc][1]
				+ f4[1][yc][1] + f5[1][yc][1] + f6[1][yc][1]
				+ f7[1][yc][1] + f8[1][yc][1] + f9[1][yc][1]
				+ f10[1][yc][1] + f11[1][yc][1] + f12[1][yc][1]
				+ f13[1][yc][1] + f14[1][yc][1] + f15[1][yc][1]
				+ f17[1][yc][1]));

			f18[1][yc][1] = f16[1][yc][1];
		}
		#pragma omp section
		{
			//Corner7: H as shown in the schematic

			A = 0;

			yc = 1;

			g1[1][yc][1] = g3[1][yc][1] + A;
			g2[1][yc][1] = g4[1][yc][1];
			g5[1][yc][1] = g6[1][yc][1];
			g7[1][yc][1] = g9[1][yc][1];
			g11[1][yc][1] = g13[1][yc][1];
			g15[1][yc][1] = g17[1][yc][1];

			g10[1][yc][1] = gn10[1][yc][1];//undo streaming
			g8[1][yc][1] = gn8[1][yc][1];
			g12[1][yc][1] = gn12[1][yc][1];
			g14[1][yc][1] = gn14[1][yc][1];
			g16[1][yc][1] = gn16[1][yc][1];
			g18[1][yc][1] = gn18[1][yc][1];

			g10[1][yc][1] = 1.0 / 2.0*(g10[1][yc][1] + g8[1][yc][1]);
			g8[1][yc][1] = g10[1][yc][1];
			g12[1][yc][1] = 1.0 / 2.0*(g12[1][yc][1] + g14[1][yc][1]);
			g14[1][yc][1] = g12[1][yc][1];

			g16[1][yc][1] = (1.0 / 2.0)*(grho[1][yc][2] - (g0[1][yc][1]
				+ g1[1][yc][1] + g2[1][yc][1] + g3[1][yc][1]
				+ g4[1][yc][1] + g5[1][yc][1] + g6[1][yc][1]
				+ g7[1][yc][1] + g8[1][yc][1] + g9[1][yc][1]
				+ g10[1][yc][1] + g11[1][yc][1] + g12[1][yc][1]
				+ g13[1][yc][1] + g14[1][yc][1] + g15[1][yc][1]
				+ g17[1][yc][1]));

			g18[1][yc][1] = g16[1][yc][1];
		}
		#pragma omp section
		{
			//Corner7: H as shown in the schematic

			A = 0;

			yc = 1;

			h1[1][yc][1] = h3[1][yc][1] + A;
			h2[1][yc][1] = h4[1][yc][1];
			h5[1][yc][1] = h6[1][yc][1];
			h7[1][yc][1] = h9[1][yc][1];
			h11[1][yc][1] = h13[1][yc][1];
			h15[1][yc][1] = h17[1][yc][1];

			h10[1][yc][1] = hn10[1][yc][1];//undo streaminh
			h8[1][yc][1] = hn8[1][yc][1];
			h12[1][yc][1] = hn12[1][yc][1];
			h14[1][yc][1] = hn14[1][yc][1];
			h16[1][yc][1] = hn16[1][yc][1];
			h18[1][yc][1] = hn18[1][yc][1];

			h10[1][yc][1] = 1.0 / 2.0*(h10[1][yc][1] + h8[1][yc][1]);
			h8[1][yc][1] = h10[1][yc][1];
			h12[1][yc][1] = 1.0 / 2.0*(h12[1][yc][1] + h14[1][yc][1]);
			h14[1][yc][1] = h12[1][yc][1];

			h16[1][yc][1] = (1.0 / 2.0)*(hrho[1][yc][2] - (h0[1][yc][1]
				+ h1[1][yc][1] + h2[1][yc][1] + h3[1][yc][1]
				+ h4[1][yc][1] + h5[1][yc][1] + h6[1][yc][1]
				+ h7[1][yc][1] + h8[1][yc][1] + h9[1][yc][1]
				+ h10[1][yc][1] + h11[1][yc][1] + h12[1][yc][1]
				+ h13[1][yc][1] + h14[1][yc][1] + h15[1][yc][1]
				+ h17[1][yc][1]));

			h18[1][yc][1] = h16[1][yc][1];
		}
		#pragma omp section
		{
			//Corner7: H as shown in the schematic

			A = 0;

			yc = 1;

			p1[1][yc][1] = p3[1][yc][1] + A;
			p2[1][yc][1] = p4[1][yc][1];
			p5[1][yc][1] = p6[1][yc][1];
			p7[1][yc][1] = p9[1][yc][1];
			p11[1][yc][1] = p13[1][yc][1];
			p15[1][yc][1] = p17[1][yc][1];

			p10[1][yc][1] = pn10[1][yc][1];//undo streaminp
			p8[1][yc][1] = pn8[1][yc][1];
			p12[1][yc][1] = pn12[1][yc][1];
			p14[1][yc][1] = pn14[1][yc][1];
			p16[1][yc][1] = pn16[1][yc][1];
			p18[1][yc][1] = pn18[1][yc][1];

			p10[1][yc][1] = 1.0 / 2.0*(p10[1][yc][1] + p8[1][yc][1]);
			p8[1][yc][1] = p10[1][yc][1];
			p12[1][yc][1] = 1.0 / 2.0*(p12[1][yc][1] + p14[1][yc][1]);
			p14[1][yc][1] = p12[1][yc][1];

			p16[1][yc][1] = (1.0 / 2.0)*(prho[1][yc][2] - (p0[1][yc][1]
				+ p1[1][yc][1] + p2[1][yc][1] + p3[1][yc][1]
				+ p4[1][yc][1] + p5[1][yc][1] + p6[1][yc][1]
				+ p7[1][yc][1] + p8[1][yc][1] + p9[1][yc][1]
				+ p10[1][yc][1] + p11[1][yc][1] + p12[1][yc][1]
				+ p13[1][yc][1] + p14[1][yc][1] + p15[1][yc][1]
				+ p17[1][yc][1]));

			p18[1][yc][1] = p16[1][yc][1];
		}
		#pragma omp section
		{
			//Corner8: E as shown in the schematic 

			A = 0;// 1.0/3.0*rho_in2*uxwall;

			yc = 1;

			f3[inx][yc][1] = f1[inx][yc][1] - A;
			f2[inx][yc][1] = f4[inx][yc][1];
			f5[inx][yc][1] = f6[inx][yc][1];
			f8[inx][yc][1] = f10[inx][yc][1];
			f12[inx][yc][1] = f14[inx][yc][1];
			f15[inx][yc][1] = f17[inx][yc][1];

			f9[inx][yc][1] = fn9[inx][yc][1];//undo streaming
			f7[inx][yc][1] = fn7[inx][yc][1];
			f11[inx][yc][1] = fn11[inx][yc][1];
			f13[inx][yc][1] = fn13[inx][yc][1];
			f16[inx][yc][1] = fn16[inx][yc][1];
			f18[inx][yc][1] = fn18[inx][yc][1];

			f9[inx][yc][1] = 1.0 / 2.0*(f9[inx][yc][1] + f7[inx][yc][1]);
			f7[inx][yc][1] = f9[inx][yc][1];
			f11[inx][yc][1] = 1.0 / 2.0*(f11[inx][yc][1] + f13[inx][yc][1]);
			f13[inx][yc][1] = f11[inx][yc][1];

			f16[inx][yc][1] = (1.0 / 2.0)*(rho[inx][yc][2] - (f0[inx][yc][1]
				+ f1[inx][yc][1] + f2[inx][yc][1] + f3[inx][yc][1]
				+ f4[inx][yc][1] + f5[inx][yc][1] + f6[inx][yc][1]
				+ f7[inx][yc][1] + f8[inx][yc][1] + f9[inx][yc][1]
				+ f10[inx][yc][1] + f11[inx][yc][1] + f12[inx][yc][1]
				+ f13[inx][yc][1] + f14[inx][yc][1] + f15[inx][yc][1]
				+ f17[inx][yc][1]));

			f18[inx][yc][1] = f16[inx][yc][1];
		}
		#pragma omp section
		{
			//Corner8: E as shown in the schematic 

			A = 0;// 1.0/3.0*rho_in2*uxwall;

			yc = 1;

			g3[inx][yc][1] = g1[inx][yc][1] - A;
			g2[inx][yc][1] = g4[inx][yc][1];
			g5[inx][yc][1] = g6[inx][yc][1];
			g8[inx][yc][1] = g10[inx][yc][1];
			g12[inx][yc][1] = g14[inx][yc][1];
			g15[inx][yc][1] = g17[inx][yc][1];

			g9[inx][yc][1] = gn9[inx][yc][1];//undo streaming
			g7[inx][yc][1] = gn7[inx][yc][1];
			g11[inx][yc][1] = gn11[inx][yc][1];
			g13[inx][yc][1] = gn13[inx][yc][1];
			g16[inx][yc][1] = gn16[inx][yc][1];
			g18[inx][yc][1] = gn18[inx][yc][1];

			g9[inx][yc][1] = 1.0 / 2.0*(g9[inx][yc][1] + g7[inx][yc][1]);
			g7[inx][yc][1] = g9[inx][yc][1];
			g11[inx][yc][1] = 1.0 / 2.0*(g11[inx][yc][1] + g13[inx][yc][1]);
			g13[inx][yc][1] = g11[inx][yc][1];

			g16[inx][yc][1] = (1.0 / 2.0)*(grho[inx][yc][2] - (g0[inx][yc][1]
				+ g1[inx][yc][1] + g2[inx][yc][1] + g3[inx][yc][1]
				+ g4[inx][yc][1] + g5[inx][yc][1] + g6[inx][yc][1]
				+ g7[inx][yc][1] + g8[inx][yc][1] + g9[inx][yc][1]
				+ g10[inx][yc][1] + g11[inx][yc][1] + g12[inx][yc][1]
				+ g13[inx][yc][1] + g14[inx][yc][1] + g15[inx][yc][1]
				+ g17[inx][yc][1]));

			g18[inx][yc][1] = g16[inx][yc][1];
		}
		#pragma omp section
		{
			//Corner8: E as shown in the schematic 

			A = 0;// 1.0/3.0*rho_in2*uxwall;

			yc = 1;

			h3[inx][yc][1] = h1[inx][yc][1] - A;
			h2[inx][yc][1] = h4[inx][yc][1];
			h5[inx][yc][1] = h6[inx][yc][1];
			h8[inx][yc][1] = h10[inx][yc][1];
			h12[inx][yc][1] = h14[inx][yc][1];
			h15[inx][yc][1] = h17[inx][yc][1];

			h9[inx][yc][1] = hn9[inx][yc][1];//undo streaminh
			h7[inx][yc][1] = hn7[inx][yc][1];
			h11[inx][yc][1] = hn11[inx][yc][1];
			h13[inx][yc][1] = hn13[inx][yc][1];
			h16[inx][yc][1] = hn16[inx][yc][1];
			h18[inx][yc][1] = hn18[inx][yc][1];

			h9[inx][yc][1] = 1.0 / 2.0*(h9[inx][yc][1] + h7[inx][yc][1]);
			h7[inx][yc][1] = h9[inx][yc][1];
			h11[inx][yc][1] = 1.0 / 2.0*(h11[inx][yc][1] + h13[inx][yc][1]);
			h13[inx][yc][1] = h11[inx][yc][1];

			h16[inx][yc][1] = (1.0 / 2.0)*(hrho[inx][yc][2] - (h0[inx][yc][1]
				+ h1[inx][yc][1] + h2[inx][yc][1] + h3[inx][yc][1]
				+ h4[inx][yc][1] + h5[inx][yc][1] + h6[inx][yc][1]
				+ h7[inx][yc][1] + h8[inx][yc][1] + h9[inx][yc][1]
				+ h10[inx][yc][1] + h11[inx][yc][1] + h12[inx][yc][1]
				+ h13[inx][yc][1] + h14[inx][yc][1] + h15[inx][yc][1]
				+ h17[inx][yc][1]));

			h18[inx][yc][1] = h16[inx][yc][1];
		}
		#pragma omp section
		{
			//Corner8: E as shown in the schematic 

			A = 0;// 1.0/3.0*rho_in2*uxwall;

			yc = 1;

			p3[inx][yc][1] = p1[inx][yc][1] - A;
			p2[inx][yc][1] = p4[inx][yc][1];
			p5[inx][yc][1] = p6[inx][yc][1];
			p8[inx][yc][1] = p10[inx][yc][1];
			p12[inx][yc][1] = p14[inx][yc][1];
			p15[inx][yc][1] = p17[inx][yc][1];

			p9[inx][yc][1] = pn9[inx][yc][1];//undo streaminp
			p7[inx][yc][1] = pn7[inx][yc][1];
			p11[inx][yc][1] = pn11[inx][yc][1];
			p13[inx][yc][1] = pn13[inx][yc][1];
			p16[inx][yc][1] = pn16[inx][yc][1];
			p18[inx][yc][1] = pn18[inx][yc][1];

			p9[inx][yc][1] = 1.0 / 2.0*(p9[inx][yc][1] + p7[inx][yc][1]);
			p7[inx][yc][1] = p9[inx][yc][1];
			p11[inx][yc][1] = 1.0 / 2.0*(p11[inx][yc][1] + p13[inx][yc][1]);
			p13[inx][yc][1] = p11[inx][yc][1];

			p16[inx][yc][1] = (1.0 / 2.0)*(prho[inx][yc][2] - (p0[inx][yc][1]
				+ p1[inx][yc][1] + p2[inx][yc][1] + p3[inx][yc][1]
				+ p4[inx][yc][1] + p5[inx][yc][1] + p6[inx][yc][1]
				+ p7[inx][yc][1] + p8[inx][yc][1] + p9[inx][yc][1]
				+ p10[inx][yc][1] + p11[inx][yc][1] + p12[inx][yc][1]
				+ p13[inx][yc][1] + p14[inx][yc][1] + p15[inx][yc][1]
				+ p17[inx][yc][1]));

			p18[inx][yc][1] = p16[inx][yc][1];
		}
	}
	#pragma omp parallel sections private(A,xc,yc)
	{
		#pragma omp section
		{
			//Corner9: A as shown in the schematic                                    

			A = 0;

			yc = 1;//corners_y0[0];
			xc = 1;//corners_x0[0];

			fm1[xc][yc][1] = fm3[xc][yc][1] + A;
			fm2[xc][yc][1] = fm4[xc][yc][1];
			fm5[xc][yc][1] = fm6[xc][yc][1];
			fm7[xc][yc][1] = fm9[xc][yc][1];
			fm11[xc][yc][1] = fm13[xc][yc][1];
			fm15[xc][yc][1] = fm17[xc][yc][1];

			fm10[xc][yc][1] = fmn10[xc][yc][1];//undo streaming
			fm8[xc][yc][1] = fmn8[xc][yc][1];
			fm12[xc][yc][1] = fmn12[xc][yc][1];
			fm14[xc][yc][1] = fmn14[xc][yc][1];
			fm16[xc][yc][1] = fmn16[xc][yc][1];
			fm18[xc][yc][1] = fmn18[xc][yc][1];

			fm10[xc][yc][1] = 1.0 / 2.0*(fm10[xc][yc][1] + fm8[xc][yc][1]);
			fm8[xc][yc][1] = fm10[xc][yc][1];
			fm12[xc][yc][1] = 1.0 / 2.0*(fm12[xc][yc][1] + fm14[xc][yc][1]);
			fm14[xc][yc][1] = fm12[xc][yc][1];

			fm16[xc][yc][1] = (1.0 / 2.0)*(rho_out - (fm0[xc][yc][1]
				+ fm1[xc][yc][1] + fm2[xc][yc][1] + fm3[xc][yc][1]
				+ fm4[xc][yc][1] + fm5[xc][yc][1] + fm6[xc][yc][1]
				+ fm7[xc][yc][1] + fm8[xc][yc][1] + fm9[xc][yc][1]
				+ fm10[xc][yc][1] + fm11[xc][yc][1] + fm12[xc][yc][1]
				+ fm13[xc][yc][1] + fm14[xc][yc][1] + fm15[xc][yc][1]
				+ fm17[xc][yc][1]));

			fm18[xc][yc][1] = fm16[xc][yc][1];

		}
		#pragma omp section
		{
			//Corner9: A as shown in the schematic                                    

			A = 0;

			yc = 1;//corners_y0[0];
			xc = 1;//corners_x0[0];

			gm1[xc][yc][1] = gm3[xc][yc][1] + A;
			gm2[xc][yc][1] = gm4[xc][yc][1];
			gm5[xc][yc][1] = gm6[xc][yc][1];
			gm7[xc][yc][1] = gm9[xc][yc][1];
			gm11[xc][yc][1] = gm13[xc][yc][1];
			gm15[xc][yc][1] = gm17[xc][yc][1];

			gm10[xc][yc][1] = gmn10[xc][yc][1];//undo streaming
			gm8[xc][yc][1] = gmn8[xc][yc][1];
			gm12[xc][yc][1] = gmn12[xc][yc][1];
			gm14[xc][yc][1] = gmn14[xc][yc][1];
			gm16[xc][yc][1] = gmn16[xc][yc][1];
			gm18[xc][yc][1] = gmn18[xc][yc][1];

			gm10[xc][yc][1] = 1.0 / 2.0*(gm10[xc][yc][1] + gm8[xc][yc][1]);
			gm8[xc][yc][1] = gm10[xc][yc][1];
			gm12[xc][yc][1] = 1.0 / 2.0*(gm12[xc][yc][1] + gm14[xc][yc][1]);
			gm14[xc][yc][1] = gm12[xc][yc][1];

			gm16[xc][yc][1] = (1.0 / 2.0)*(gmrho[xc][yc-1][1] - (gm0[xc][yc][1]
				+ gm1[xc][yc][1] + gm2[xc][yc][1] + gm3[xc][yc][1]
				+ gm4[xc][yc][1] + gm5[xc][yc][1] + gm6[xc][yc][1]
				+ gm7[xc][yc][1] + gm8[xc][yc][1] + gm9[xc][yc][1]
				+ gm10[xc][yc][1] + gm11[xc][yc][1] + gm12[xc][yc][1]
				+ gm13[xc][yc][1] + gm14[xc][yc][1] + gm15[xc][yc][1]
				+ gm17[xc][yc][1]));

			gm18[xc][yc][1] = gm16[xc][yc][1];

		}
		#pragma omp section
		{
			//Corner9: A as shown in the schematic                                    

			A = 0;

			yc = 1;//corners_y0[0];
			xc = 1;//corners_x0[0];

			hm1[xc][yc][1] = hm3[xc][yc][1] + A;
			hm2[xc][yc][1] = hm4[xc][yc][1];
			hm5[xc][yc][1] = hm6[xc][yc][1];
			hm7[xc][yc][1] = hm9[xc][yc][1];
			hm11[xc][yc][1] = hm13[xc][yc][1];
			hm15[xc][yc][1] = hm17[xc][yc][1];

			hm10[xc][yc][1] = hmn10[xc][yc][1];//undo streaminh
			hm8[xc][yc][1] = hmn8[xc][yc][1];
			hm12[xc][yc][1] = hmn12[xc][yc][1];
			hm14[xc][yc][1] = hmn14[xc][yc][1];
			hm16[xc][yc][1] = hmn16[xc][yc][1];
			hm18[xc][yc][1] = hmn18[xc][yc][1];

			hm10[xc][yc][1] = 1.0 / 2.0*(hm10[xc][yc][1] + hm8[xc][yc][1]);
			hm8[xc][yc][1] = hm10[xc][yc][1];
			hm12[xc][yc][1] = 1.0 / 2.0*(hm12[xc][yc][1] + hm14[xc][yc][1]);
			hm14[xc][yc][1] = hm12[xc][yc][1];

			hm16[xc][yc][1] = (1.0 / 2.0)*(hmrho[xc][yc-1][1] - (hm0[xc][yc][1]
				+ hm1[xc][yc][1] + hm2[xc][yc][1] + hm3[xc][yc][1]
				+ hm4[xc][yc][1] + hm5[xc][yc][1] + hm6[xc][yc][1]
				+ hm7[xc][yc][1] + hm8[xc][yc][1] + hm9[xc][yc][1]
				+ hm10[xc][yc][1] + hm11[xc][yc][1] + hm12[xc][yc][1]
				+ hm13[xc][yc][1] + hm14[xc][yc][1] + hm15[xc][yc][1]
				+ hm17[xc][yc][1]));

			hm18[xc][yc][1] = hm16[xc][yc][1];

		}
		#pragma omp section
		{
			//Corner9: A as shown in the schematic                                    

			A = 0;

			yc = 1;//corners_y0[0];
			xc = 1;//corners_x0[0];

			pm1[xc][yc][1] = pm3[xc][yc][1] + A;
			pm2[xc][yc][1] = pm4[xc][yc][1];
			pm5[xc][yc][1] = pm6[xc][yc][1];
			pm7[xc][yc][1] = pm9[xc][yc][1];
			pm11[xc][yc][1] = pm13[xc][yc][1];
			pm15[xc][yc][1] = pm17[xc][yc][1];

			pm10[xc][yc][1] = pmn10[xc][yc][1];//undo streaminp
			pm8[xc][yc][1] = pmn8[xc][yc][1];
			pm12[xc][yc][1] = pmn12[xc][yc][1];
			pm14[xc][yc][1] = pmn14[xc][yc][1];
			pm16[xc][yc][1] = pmn16[xc][yc][1];
			pm18[xc][yc][1] = pmn18[xc][yc][1];

			pm10[xc][yc][1] = 1.0 / 2.0*(pm10[xc][yc][1] + pm8[xc][yc][1]);
			pm8[xc][yc][1] = pm10[xc][yc][1];
			pm12[xc][yc][1] = 1.0 / 2.0*(pm12[xc][yc][1] + pm14[xc][yc][1]);
			pm14[xc][yc][1] = pm12[xc][yc][1];

			pm16[xc][yc][1] = (1.0 / 2.0)*(pmrho[xc][yc-1][1] - (pm0[xc][yc][1]
				+ pm1[xc][yc][1] + pm2[xc][yc][1] + pm3[xc][yc][1]
				+ pm4[xc][yc][1] + pm5[xc][yc][1] + pm6[xc][yc][1]
				+ pm7[xc][yc][1] + pm8[xc][yc][1] + pm9[xc][yc][1]
				+ pm10[xc][yc][1] + pm11[xc][yc][1] + pm12[xc][yc][1]
				+ pm13[xc][yc][1] + pm14[xc][yc][1] + pm15[xc][yc][1]
				+ pm17[xc][yc][1]));

			pm18[xc][yc][1] = pm16[xc][yc][1];

		}
		#pragma omp section
		{
			//Corner10: B as shown in the schematic

			A = 0;

			yc = 1;
			xc = mnx;

			fm3[xc][yc][1] = fm1[xc][yc][1] - A;
			fm2[xc][yc][1] = fm4[xc][yc][1];
			fm5[xc][yc][1] = fm6[xc][yc][1];
			fm8[xc][yc][1] = fm10[xc][yc][1];
			fm12[xc][yc][1] = fm14[xc][yc][1];
			fm15[xc][yc][1] = fm17[xc][yc][1];

			fm9[xc][yc][1] = fmn9[xc][yc][1];//undo streaming
			fm7[xc][yc][1] = fmn7[xc][yc][1];
			fm11[xc][yc][1] = fmn11[xc][yc][1];
			fm13[xc][yc][1] = fmn13[xc][yc][1];
			fm16[xc][yc][1] = fmn16[xc][yc][1];
			fm18[xc][yc][1] = fmn18[xc][yc][1];

			fm9[xc][yc][1] = 1.0 / 2.0*(fm9[xc][yc][1] + fm7[xc][yc][1]);
			fm7[xc][yc][1] = fm9[xc][yc][1];
			fm11[xc][yc][1] = 1.0 / 2.0*(fm11[xc][yc][1] + fm13[xc][yc][1]);
			fm13[xc][yc][1] = fm11[xc][yc][1];

			fm16[xc][yc][1] = (1.0 / 2.0)*(rho_out - (fm0[xc][yc][1] + fm1[xc][yc][1] + fm2[xc][yc][1] + fm3[xc][yc][1] + fm4[xc][yc][1] + fm5[xc][yc][1] + fm6[xc][yc][1] + fm7[xc][yc][1] + fm8[xc][yc][1] + fm9[xc][yc][1] + fm10[xc][yc][1] + fm11[xc][yc][1] + fm12[xc][yc][1] + fm13[xc][yc][1] + fm14[xc][yc][1] + fm15[xc][yc][1] + fm17[xc][yc][1]));

			fm18[xc][yc][1] = fm16[xc][yc][1];
		}
		#pragma omp section
		{
			//Corner10: B as shown in the schematic

			A = 0;

			yc = 1;
			xc = mnx;

			gm3[xc][yc][1] = gm1[xc][yc][1] - A;
			gm2[xc][yc][1] = gm4[xc][yc][1];
			gm5[xc][yc][1] = gm6[xc][yc][1];
			gm8[xc][yc][1] = gm10[xc][yc][1];
			gm12[xc][yc][1] = gm14[xc][yc][1];
			gm15[xc][yc][1] = gm17[xc][yc][1];

			gm9[xc][yc][1] = gmn9[xc][yc][1];//undo streaming
			gm7[xc][yc][1] = gmn7[xc][yc][1];
			gm11[xc][yc][1] = gmn11[xc][yc][1];
			gm13[xc][yc][1] = gmn13[xc][yc][1];
			gm16[xc][yc][1] = gmn16[xc][yc][1];
			gm18[xc][yc][1] = gmn18[xc][yc][1];

			gm9[xc][yc][1] = 1.0 / 2.0*(gm9[xc][yc][1] + gm7[xc][yc][1]);
			gm7[xc][yc][1] = gm9[xc][yc][1];
			gm11[xc][yc][1] = 1.0 / 2.0*(gm11[xc][yc][1] + gm13[xc][yc][1]);
			gm13[xc][yc][1] = gm11[xc][yc][1];

			gm16[xc][yc][1] = (1.0 / 2.0)*(gmrho[xc][yc-1][1] - (gm0[xc][yc][1] + gm1[xc][yc][1] + gm2[xc][yc][1] + gm3[xc][yc][1] + gm4[xc][yc][1] + gm5[xc][yc][1] + gm6[xc][yc][1] + gm7[xc][yc][1] + gm8[xc][yc][1] + gm9[xc][yc][1] + gm10[xc][yc][1] + gm11[xc][yc][1] + gm12[xc][yc][1] + gm13[xc][yc][1] + gm14[xc][yc][1] + gm15[xc][yc][1] + gm17[xc][yc][1]));

			gm18[xc][yc][1] = gm16[xc][yc][1];
		}
		#pragma omp section
		{
			//Corner10: B as shown in the schematic

			A = 0;

			yc = 1;
			xc = mnx;

			hm3[xc][yc][1] = hm1[xc][yc][1] - A;
			hm2[xc][yc][1] = hm4[xc][yc][1];
			hm5[xc][yc][1] = hm6[xc][yc][1];
			hm8[xc][yc][1] = hm10[xc][yc][1];
			hm12[xc][yc][1] = hm14[xc][yc][1];
			hm15[xc][yc][1] = hm17[xc][yc][1];

			hm9[xc][yc][1] = hmn9[xc][yc][1];//undo streaminh
			hm7[xc][yc][1] = hmn7[xc][yc][1];
			hm11[xc][yc][1] = hmn11[xc][yc][1];
			hm13[xc][yc][1] = hmn13[xc][yc][1];
			hm16[xc][yc][1] = hmn16[xc][yc][1];
			hm18[xc][yc][1] = hmn18[xc][yc][1];

			hm9[xc][yc][1] = 1.0 / 2.0*(hm9[xc][yc][1] + hm7[xc][yc][1]);
			hm7[xc][yc][1] = hm9[xc][yc][1];
			hm11[xc][yc][1] = 1.0 / 2.0*(hm11[xc][yc][1] + hm13[xc][yc][1]);
			hm13[xc][yc][1] = hm11[xc][yc][1];

			hm16[xc][yc][1] = (1.0 / 2.0)*(hmrho[xc][yc-1][1] - (hm0[xc][yc][1] + hm1[xc][yc][1] + hm2[xc][yc][1] + hm3[xc][yc][1] + hm4[xc][yc][1] + hm5[xc][yc][1] + hm6[xc][yc][1] + hm7[xc][yc][1] + hm8[xc][yc][1] + hm9[xc][yc][1] + hm10[xc][yc][1] + hm11[xc][yc][1] + hm12[xc][yc][1] + hm13[xc][yc][1] + hm14[xc][yc][1] + hm15[xc][yc][1] + hm17[xc][yc][1]));

			hm18[xc][yc][1] = hm16[xc][yc][1];
		}
		#pragma omp section
		{
			//Corner10: B as shown in the schematic

			A = 0;

			yc = 1;
			xc = mnx;

			pm3[xc][yc][1] = pm1[xc][yc][1] - A;
			pm2[xc][yc][1] = pm4[xc][yc][1];
			pm5[xc][yc][1] = pm6[xc][yc][1];
			pm8[xc][yc][1] = pm10[xc][yc][1];
			pm12[xc][yc][1] = pm14[xc][yc][1];
			pm15[xc][yc][1] = pm17[xc][yc][1];

			pm9[xc][yc][1] = pmn9[xc][yc][1];//undo streaminp
			pm7[xc][yc][1] = pmn7[xc][yc][1];
			pm11[xc][yc][1] = pmn11[xc][yc][1];
			pm13[xc][yc][1] = pmn13[xc][yc][1];
			pm16[xc][yc][1] = pmn16[xc][yc][1];
			pm18[xc][yc][1] = pmn18[xc][yc][1];

			pm9[xc][yc][1] = 1.0 / 2.0*(pm9[xc][yc][1] + pm7[xc][yc][1]);
			pm7[xc][yc][1] = pm9[xc][yc][1];
			pm11[xc][yc][1] = 1.0 / 2.0*(pm11[xc][yc][1] + pm13[xc][yc][1]);
			pm13[xc][yc][1] = pm11[xc][yc][1];

			pm16[xc][yc][1] = (1.0 / 2.0)*(pmrho[xc][yc-1][1] - (pm0[xc][yc][1] + pm1[xc][yc][1] + pm2[xc][yc][1] + pm3[xc][yc][1] + pm4[xc][yc][1] + pm5[xc][yc][1] + pm6[xc][yc][1] + pm7[xc][yc][1] + pm8[xc][yc][1] + pm9[xc][yc][1] + pm10[xc][yc][1] + pm11[xc][yc][1] + pm12[xc][yc][1] + pm13[xc][yc][1] + pm14[xc][yc][1] + pm15[xc][yc][1] + pm17[xc][yc][1]));

			pm18[xc][yc][1] = pm16[xc][yc][1];
		}
		#pragma omp section
		{
			//Corner11: A'as shown in the schematic

			A = 0;// 1.0/3.0*rho_out*uxwall;

			yc = 1;
			xc = 1;

			fm1[xc][yc][nz] = fm3[xc][yc][nz] + A;
			fm2[xc][yc][nz] = fm4[xc][yc][nz];
			fm6[xc][yc][nz] = fm5[xc][yc][nz];
			fm7[xc][yc][nz] = fm9[xc][yc][nz];
			fm18[xc][yc][nz] = fm16[xc][yc][nz];
			fm14[xc][yc][nz] = fm12[xc][yc][nz];

			fm10[xc][yc][nz] = fmn10[xc][yc][nz];//undo streaming
			fm8[xc][yc][nz] = fmn8[xc][yc][nz];
			fm11[xc][yc][nz] = fmn11[xc][yc][nz];
			fm13[xc][yc][nz] = fmn13[xc][yc][nz];
			fm15[xc][yc][nz] = fmn15[xc][yc][nz];
			fm17[xc][yc][nz] = fmn17[xc][yc][nz];

			fm10[xc][yc][nz] = 1.0 / 2.0*(fm10[xc][yc][nz] + fm8[xc][yc][nz]);
			fm8[xc][yc][nz] = fm10[xc][yc][nz];
			fm11[xc][yc][nz] = 1.0 / 2.0*(fm11[xc][yc][nz] + fm13[xc][yc][nz]);
			fm13[xc][yc][nz] = fm11[xc][yc][nz];

			fm15[xc][yc][nz] = (1.0 / 2.0)*(rho_out - (fm0[xc][yc][nz] + fm1[xc][yc][nz] + fm2[xc][yc][nz] + fm3[xc][yc][nz] + fm4[xc][yc][nz] + fm5[xc][yc][nz] + fm6[xc][yc][nz] + fm7[xc][yc][nz] + fm8[xc][yc][nz] + fm9[xc][yc][nz] + fm10[xc][yc][nz] + fm11[xc][yc][nz] + fm12[xc][yc][nz] + fm13[xc][yc][nz] + fm14[xc][yc][nz] + fm16[xc][yc][nz] + fm18[xc][yc][nz]));

			fm17[xc][yc][nz] = fm15[xc][yc][nz];
		}
		#pragma omp section
		{
			//Corner11: A'as shown in the schematic

			A = 0;// 1.0/3.0*rho_out*uxwall;

			yc = 1;
			xc = 1;

			gm1[xc][yc][nz] = gm3[xc][yc][nz] + A;
			gm2[xc][yc][nz] = gm4[xc][yc][nz];
			gm6[xc][yc][nz] = gm5[xc][yc][nz];
			gm7[xc][yc][nz] = gm9[xc][yc][nz];
			gm18[xc][yc][nz] = gm16[xc][yc][nz];
			gm14[xc][yc][nz] = gm12[xc][yc][nz];

			gm10[xc][yc][nz] = gmn10[xc][yc][nz];//undo streaming
			gm8[xc][yc][nz] = gmn8[xc][yc][nz];
			gm11[xc][yc][nz] = gmn11[xc][yc][nz];
			gm13[xc][yc][nz] = gmn13[xc][yc][nz];
			gm15[xc][yc][nz] = gmn15[xc][yc][nz];
			gm17[xc][yc][nz] = gmn17[xc][yc][nz];

			gm10[xc][yc][nz] = 1.0 / 2.0*(gm10[xc][yc][nz] + gm8[xc][yc][nz]);
			gm8[xc][yc][nz] = gm10[xc][yc][nz];
			gm11[xc][yc][nz] = 1.0 / 2.0*(gm11[xc][yc][nz] + gm13[xc][yc][nz]);
			gm13[xc][yc][nz] = gm11[xc][yc][nz];

			gm15[xc][yc][nz] = (1.0 / 2.0)*(gmrho[xc][yc-1][nz] - (gm0[xc][yc][nz] + gm1[xc][yc][nz] + gm2[xc][yc][nz] + gm3[xc][yc][nz] + gm4[xc][yc][nz] + gm5[xc][yc][nz] + gm6[xc][yc][nz] + gm7[xc][yc][nz] + gm8[xc][yc][nz] + gm9[xc][yc][nz] + gm10[xc][yc][nz] + gm11[xc][yc][nz] + gm12[xc][yc][nz] + gm13[xc][yc][nz] + gm14[xc][yc][nz] + gm16[xc][yc][nz] + gm18[xc][yc][nz]));

			gm17[xc][yc][nz] = gm15[xc][yc][nz];
		}
		#pragma omp section
		{
			//Corner11: A'as shown in the schematic

			A = 0;// 1.0/3.0*rho_out*uxwall;

			yc = 1;
			xc = 1;

			hm1[xc][yc][nz] = hm3[xc][yc][nz] + A;
			hm2[xc][yc][nz] = hm4[xc][yc][nz];
			hm6[xc][yc][nz] = hm5[xc][yc][nz];
			hm7[xc][yc][nz] = hm9[xc][yc][nz];
			hm18[xc][yc][nz] = hm16[xc][yc][nz];
			hm14[xc][yc][nz] = hm12[xc][yc][nz];

			hm10[xc][yc][nz] = hmn10[xc][yc][nz];//undo streaminh
			hm8[xc][yc][nz] = hmn8[xc][yc][nz];
			hm11[xc][yc][nz] = hmn11[xc][yc][nz];
			hm13[xc][yc][nz] = hmn13[xc][yc][nz];
			hm15[xc][yc][nz] = hmn15[xc][yc][nz];
			hm17[xc][yc][nz] = hmn17[xc][yc][nz];

			hm10[xc][yc][nz] = 1.0 / 2.0*(hm10[xc][yc][nz] + hm8[xc][yc][nz]);
			hm8[xc][yc][nz] = hm10[xc][yc][nz];
			hm11[xc][yc][nz] = 1.0 / 2.0*(hm11[xc][yc][nz] + hm13[xc][yc][nz]);
			hm13[xc][yc][nz] = hm11[xc][yc][nz];

			hm15[xc][yc][nz] = (1.0 / 2.0)*(hmrho[xc][yc-1][nz] - (hm0[xc][yc][nz] + hm1[xc][yc][nz] + hm2[xc][yc][nz] + hm3[xc][yc][nz] + hm4[xc][yc][nz] + hm5[xc][yc][nz] + hm6[xc][yc][nz] + hm7[xc][yc][nz] + hm8[xc][yc][nz] + hm9[xc][yc][nz] + hm10[xc][yc][nz] + hm11[xc][yc][nz] + hm12[xc][yc][nz] + hm13[xc][yc][nz] + hm14[xc][yc][nz] + hm16[xc][yc][nz] + hm18[xc][yc][nz]));

			hm17[xc][yc][nz] = hm15[xc][yc][nz];
		}
		#pragma omp section
		{
			//Corner11: A'as shown in the schematic

			A = 0;// 1.0/3.0*rho_out*uxwall;

			yc = 1;
			xc = 1;

			pm1[xc][yc][nz] = pm3[xc][yc][nz] + A;
			pm2[xc][yc][nz] = pm4[xc][yc][nz];
			pm6[xc][yc][nz] = pm5[xc][yc][nz];
			pm7[xc][yc][nz] = pm9[xc][yc][nz];
			pm18[xc][yc][nz] = pm16[xc][yc][nz];
			pm14[xc][yc][nz] = pm12[xc][yc][nz];

			pm10[xc][yc][nz] = pmn10[xc][yc][nz];//undo streaminp
			pm8[xc][yc][nz] = pmn8[xc][yc][nz];
			pm11[xc][yc][nz] = pmn11[xc][yc][nz];
			pm13[xc][yc][nz] = pmn13[xc][yc][nz];
			pm15[xc][yc][nz] = pmn15[xc][yc][nz];
			pm17[xc][yc][nz] = pmn17[xc][yc][nz];

			pm10[xc][yc][nz] = 1.0 / 2.0*(pm10[xc][yc][nz] + pm8[xc][yc][nz]);
			pm8[xc][yc][nz] = pm10[xc][yc][nz];
			pm11[xc][yc][nz] = 1.0 / 2.0*(pm11[xc][yc][nz] + pm13[xc][yc][nz]);
			pm13[xc][yc][nz] = pm11[xc][yc][nz];

			pm15[xc][yc][nz] = (1.0 / 2.0)*(pmrho[xc][yc-1][nz] - (pm0[xc][yc][nz] + pm1[xc][yc][nz] + pm2[xc][yc][nz] + pm3[xc][yc][nz] + pm4[xc][yc][nz] + pm5[xc][yc][nz] + pm6[xc][yc][nz] + pm7[xc][yc][nz] + pm8[xc][yc][nz] + pm9[xc][yc][nz] + pm10[xc][yc][nz] + pm11[xc][yc][nz] + pm12[xc][yc][nz] + pm13[xc][yc][nz] + pm14[xc][yc][nz] + pm16[xc][yc][nz] + pm18[xc][yc][nz]));

			pm17[xc][yc][nz] = pm15[xc][yc][nz];
		}
		#pragma omp section
		{
			//Corner12: B'as shown in the schematic

			A = 0;// 1.0/3.0*rho_out*uxwall;

			yc = 1;
			xc = mnx;

			fm3[xc][yc][nz] = fm1[xc][yc][nz] - A;
			fm2[xc][yc][nz] = fm4[xc][yc][nz];
			fm6[xc][yc][nz] = fm5[xc][yc][nz];
			fm8[xc][yc][nz] = fm10[xc][yc][nz];
			fm13[xc][yc][nz] = fm11[xc][yc][nz];
			fm18[xc][yc][nz] = fm16[xc][yc][nz];

			fm9[xc][yc][nz] = fmn9[xc][yc][nz];//undo streaming
			fm7[xc][yc][nz] = fmn7[xc][yc][nz];
			fm12[xc][yc][nz] = fmn12[xc][yc][nz];
			fm14[xc][yc][nz] = fmn14[xc][yc][nz];
			fm15[xc][yc][nz] = fmn15[xc][yc][nz];
			fm17[xc][yc][nz] = fmn17[xc][yc][nz];

			fm9[xc][yc][nz] = 1.0 / 2.0*(fm9[xc][yc][nz] + fm7[xc][yc][nz]);
			fm7[xc][yc][nz] = fm9[xc][yc][nz];
			fm12[xc][yc][nz] = 1.0 / 2.0*(fm12[xc][yc][nz] + fm14[xc][yc][nz]);
			fm14[xc][yc][nz] = fm12[xc][yc][nz];

			fm15[xc][yc][nz] = (1.0 / 2.0)*(rho_out - (fm0[xc][yc][nz] + fm1[xc][yc][nz] + fm2[xc][yc][nz] + fm3[xc][yc][nz] + fm4[xc][yc][nz] + fm5[xc][yc][nz] + fm6[xc][yc][nz] + fm7[xc][yc][nz] + fm8[xc][yc][nz] + fm9[xc][yc][nz] + fm10[xc][yc][nz] + fm11[xc][yc][nz] + fm12[xc][yc][nz] + fm13[xc][yc][nz] + fm14[xc][yc][nz] + fm16[xc][yc][nz] + fm18[xc][yc][nz]));

			fm17[xc][yc][nz] = fm15[xc][yc][nz];
		}
		#pragma omp section
		{
			//Corner12: B'as shown in the schematic

			A = 0;// 1.0/3.0*rho_out*uxwall;

			yc = 1;
			xc = mnx;

			gm3[xc][yc][nz] = gm1[xc][yc][nz] - A;
			gm2[xc][yc][nz] = gm4[xc][yc][nz];
			gm6[xc][yc][nz] = gm5[xc][yc][nz];
			gm8[xc][yc][nz] = gm10[xc][yc][nz];
			gm13[xc][yc][nz] = gm11[xc][yc][nz];
			gm18[xc][yc][nz] = gm16[xc][yc][nz];

			gm9[xc][yc][nz] = gmn9[xc][yc][nz];//undo streaming
			gm7[xc][yc][nz] = gmn7[xc][yc][nz];
			gm12[xc][yc][nz] = gmn12[xc][yc][nz];
			gm14[xc][yc][nz] = gmn14[xc][yc][nz];
			gm15[xc][yc][nz] = gmn15[xc][yc][nz];
			gm17[xc][yc][nz] = gmn17[xc][yc][nz];

			gm9[xc][yc][nz] = 1.0 / 2.0*(gm9[xc][yc][nz] + gm7[xc][yc][nz]);
			gm7[xc][yc][nz] = gm9[xc][yc][nz];
			gm12[xc][yc][nz] = 1.0 / 2.0*(gm12[xc][yc][nz] + gm14[xc][yc][nz]);
			gm14[xc][yc][nz] = gm12[xc][yc][nz];

			gm15[xc][yc][nz] = (1.0 / 2.0)*(gmrho[xc][yc-1][nz] - (gm0[xc][yc][nz] + gm1[xc][yc][nz] + gm2[xc][yc][nz] + gm3[xc][yc][nz] + gm4[xc][yc][nz] + gm5[xc][yc][nz] + gm6[xc][yc][nz] + gm7[xc][yc][nz] + gm8[xc][yc][nz] + gm9[xc][yc][nz] + gm10[xc][yc][nz] + gm11[xc][yc][nz] + gm12[xc][yc][nz] + gm13[xc][yc][nz] + gm14[xc][yc][nz] + gm16[xc][yc][nz] + gm18[xc][yc][nz]));

			gm17[xc][yc][nz] = gm15[xc][yc][nz];
		}
		#pragma omp section
		{
			//Corner12: B'as shown in the schematic

			A = 0;// 1.0/3.0*rho_out*uxwall;

			yc = 1;
			xc = mnx;

			hm3[xc][yc][nz] = hm1[xc][yc][nz] - A;
			hm2[xc][yc][nz] = hm4[xc][yc][nz];
			hm6[xc][yc][nz] = hm5[xc][yc][nz];
			hm8[xc][yc][nz] = hm10[xc][yc][nz];
			hm13[xc][yc][nz] = hm11[xc][yc][nz];
			hm18[xc][yc][nz] = hm16[xc][yc][nz];

			hm9[xc][yc][nz] = hmn9[xc][yc][nz];//undo streaminh
			hm7[xc][yc][nz] = hmn7[xc][yc][nz];
			hm12[xc][yc][nz] = hmn12[xc][yc][nz];
			hm14[xc][yc][nz] = hmn14[xc][yc][nz];
			hm15[xc][yc][nz] = hmn15[xc][yc][nz];
			hm17[xc][yc][nz] = hmn17[xc][yc][nz];

			hm9[xc][yc][nz] = 1.0 / 2.0*(hm9[xc][yc][nz] + hm7[xc][yc][nz]);
			hm7[xc][yc][nz] = hm9[xc][yc][nz];
			hm12[xc][yc][nz] = 1.0 / 2.0*(hm12[xc][yc][nz] + hm14[xc][yc][nz]);
			hm14[xc][yc][nz] = hm12[xc][yc][nz];

			hm15[xc][yc][nz] = (1.0 / 2.0)*(hmrho[xc][yc-1][nz] - (hm0[xc][yc][nz] + hm1[xc][yc][nz] + hm2[xc][yc][nz] + hm3[xc][yc][nz] + hm4[xc][yc][nz] + hm5[xc][yc][nz] + hm6[xc][yc][nz] + hm7[xc][yc][nz] + hm8[xc][yc][nz] + hm9[xc][yc][nz] + hm10[xc][yc][nz] + hm11[xc][yc][nz] + hm12[xc][yc][nz] + hm13[xc][yc][nz] + hm14[xc][yc][nz] + hm16[xc][yc][nz] + hm18[xc][yc][nz]));

			hm17[xc][yc][nz] = hm15[xc][yc][nz];
		}
		#pragma omp section
		{
			//Corner12: B'as shown in the schematic

			A = 0;// 1.0/3.0*rho_out*uxwall;

			yc = 1;
			xc = mnx;

			pm3[xc][yc][nz] = pm1[xc][yc][nz] - A;
			pm2[xc][yc][nz] = pm4[xc][yc][nz];
			pm6[xc][yc][nz] = pm5[xc][yc][nz];
			pm8[xc][yc][nz] = pm10[xc][yc][nz];
			pm13[xc][yc][nz] = pm11[xc][yc][nz];
			pm18[xc][yc][nz] = pm16[xc][yc][nz];

			pm9[xc][yc][nz] = pmn9[xc][yc][nz];//undo streaminp
			pm7[xc][yc][nz] = pmn7[xc][yc][nz];
			pm12[xc][yc][nz] = pmn12[xc][yc][nz];
			pm14[xc][yc][nz] = pmn14[xc][yc][nz];
			pm15[xc][yc][nz] = pmn15[xc][yc][nz];
			pm17[xc][yc][nz] = pmn17[xc][yc][nz];

			pm9[xc][yc][nz] = 1.0 / 2.0*(pm9[xc][yc][nz] + pm7[xc][yc][nz]);
			pm7[xc][yc][nz] = pm9[xc][yc][nz];
			pm12[xc][yc][nz] = 1.0 / 2.0*(pm12[xc][yc][nz] + pm14[xc][yc][nz]);
			pm14[xc][yc][nz] = pm12[xc][yc][nz];

			pm15[xc][yc][nz] = (1.0 / 2.0)*(pmrho[xc][yc-1][nz] - (pm0[xc][yc][nz] + pm1[xc][yc][nz] + pm2[xc][yc][nz] + pm3[xc][yc][nz] + pm4[xc][yc][nz] + pm5[xc][yc][nz] + pm6[xc][yc][nz] + pm7[xc][yc][nz] + pm8[xc][yc][nz] + pm9[xc][yc][nz] + pm10[xc][yc][nz] + pm11[xc][yc][nz] + pm12[xc][yc][nz] + pm13[xc][yc][nz] + pm14[xc][yc][nz] + pm16[xc][yc][nz] + pm18[xc][yc][nz]));

			pm17[xc][yc][nz] = pm15[xc][yc][nz];
		}

	}
	#pragma omp parallel sections private(A,xc,yc)
	{
		#pragma omp section
		{
			//Corner13: D' as shown in the schmeatic

			xc = ia;//corners_x0[0];
			yc = 1;//corners_y0[3];

			f6[xc][yc][nz] = f5[xc][yc][nz];
			f7[xc][yc][nz] = f9[xc][yc][nz];
			f13[xc][yc][nz] = f11[xc][yc][nz];
			f14[xc][yc][nz] = f12[xc][yc][nz];
			f17[xc][yc][nz] = f15[xc][yc][nz];
			f18[xc][yc][nz] = f16[xc][yc][nz];

			//Undo Streaming
			f1[xc][yc][nz] = fn1[xc][yc][nz];
			f2[xc][yc][nz] = fn2[xc][yc][nz];
			f3[xc][yc][nz] = fn3[xc][yc][nz];
			f4[xc][yc][nz] = fn4[xc][yc][nz];
			f8[xc][yc][nz] = fn8[xc][yc][nz];
			f10[xc][yc][nz] = fn10[xc][yc][nz];

			f1[xc][yc][nz] = 1.0 / 2.0*(f1[xc][yc][nz] + f3[xc][yc][nz]);
			f3[xc][yc][nz] = f1[xc][yc][nz];

			f4[xc][yc][nz] = 1.0 / 2.0*(f4[xc][yc][nz] + f2[xc][yc][nz]);
			f2[xc][yc][nz] = f4[xc][yc][nz];

			f8[xc][yc][nz] = 1.0 / 2.0*(f8[xc][yc][nz] + f10[xc][yc][nz]);
			f10[xc][yc][nz] = f8[xc][yc][nz];
		}
		#pragma omp section
		{
			//Corner13: D' as shown in the schmeatic

			xc = ia;//corners_x0[0];
			yc = 1;//corners_y0[3];

			g6[xc][yc][nz] = g5[xc][yc][nz];
			g7[xc][yc][nz] = g9[xc][yc][nz];
			g13[xc][yc][nz] = g11[xc][yc][nz];
			g14[xc][yc][nz] = g12[xc][yc][nz];
			g17[xc][yc][nz] = g15[xc][yc][nz];
			g18[xc][yc][nz] = g16[xc][yc][nz];

			//Undo Streaming
			g1[xc][yc][nz] = gn1[xc][yc][nz];
			g2[xc][yc][nz] = gn2[xc][yc][nz];
			g3[xc][yc][nz] = gn3[xc][yc][nz];
			g4[xc][yc][nz] = gn4[xc][yc][nz];
			g8[xc][yc][nz] = gn8[xc][yc][nz];
			g10[xc][yc][nz] = gn10[xc][yc][nz];

			g1[xc][yc][nz] = 1.0 / 2.0*(g1[xc][yc][nz] + g3[xc][yc][nz]);
			g3[xc][yc][nz] = g1[xc][yc][nz];

			g4[xc][yc][nz] = 1.0 / 2.0*(g4[xc][yc][nz] + g2[xc][yc][nz]);
			g2[xc][yc][nz] = g4[xc][yc][nz];

			g8[xc][yc][nz] = 1.0 / 2.0*(g8[xc][yc][nz] + g10[xc][yc][nz]);
			g10[xc][yc][nz] = g8[xc][yc][nz];
		}
		#pragma omp section
		{
			//Corner13: D' as shown in the schmeatic

			xc = ia;//corners_x0[0];
			yc = 1;//corners_y0[3];

			h6[xc][yc][nz] = h5[xc][yc][nz];
			h7[xc][yc][nz] = h9[xc][yc][nz];
			h13[xc][yc][nz] = h11[xc][yc][nz];
			h14[xc][yc][nz] = h12[xc][yc][nz];
			h17[xc][yc][nz] = h15[xc][yc][nz];
			h18[xc][yc][nz] = h16[xc][yc][nz];

			//Undo Streaminh
			h1[xc][yc][nz] = hn1[xc][yc][nz];
			h2[xc][yc][nz] = hn2[xc][yc][nz];
			h3[xc][yc][nz] = hn3[xc][yc][nz];
			h4[xc][yc][nz] = hn4[xc][yc][nz];
			h8[xc][yc][nz] = hn8[xc][yc][nz];
			h10[xc][yc][nz] = hn10[xc][yc][nz];

			h1[xc][yc][nz] = 1.0 / 2.0*(h1[xc][yc][nz] + h3[xc][yc][nz]);
			h3[xc][yc][nz] = h1[xc][yc][nz];

			h4[xc][yc][nz] = 1.0 / 2.0*(h4[xc][yc][nz] + h2[xc][yc][nz]);
			h2[xc][yc][nz] = h4[xc][yc][nz];

			h8[xc][yc][nz] = 1.0 / 2.0*(h8[xc][yc][nz] + h10[xc][yc][nz]);
			h10[xc][yc][nz] = h8[xc][yc][nz];
		}
		#pragma omp section
		{
			//Corner13: D' as shown in the schmeatic

			xc = ia;//corners_x0[0];
			yc = 1;//corners_y0[3];

			p6[xc][yc][nz] = p5[xc][yc][nz];
			p7[xc][yc][nz] = p9[xc][yc][nz];
			p13[xc][yc][nz] = p11[xc][yc][nz];
			p14[xc][yc][nz] = p12[xc][yc][nz];
			p17[xc][yc][nz] = p15[xc][yc][nz];
			p18[xc][yc][nz] = p16[xc][yc][nz];

			//Undo Streaminp
			p1[xc][yc][nz] = pn1[xc][yc][nz];
			p2[xc][yc][nz] = pn2[xc][yc][nz];
			p3[xc][yc][nz] = pn3[xc][yc][nz];
			p4[xc][yc][nz] = pn4[xc][yc][nz];
			p8[xc][yc][nz] = pn8[xc][yc][nz];
			p10[xc][yc][nz] = pn10[xc][yc][nz];

			p1[xc][yc][nz] = 1.0 / 2.0*(p1[xc][yc][nz] + p3[xc][yc][nz]);
			p3[xc][yc][nz] = p1[xc][yc][nz];

			p4[xc][yc][nz] = 1.0 / 2.0*(p4[xc][yc][nz] + p2[xc][yc][nz]);
			p2[xc][yc][nz] = p4[xc][yc][nz];

			p8[xc][yc][nz] = 1.0 / 2.0*(p8[xc][yc][nz] + p10[xc][yc][nz]);
			p10[xc][yc][nz] = p8[xc][yc][nz];
		}
		#pragma omp section
		{
			//Corner14: C' as shown in the schematic

			xc = ib;
			yc = 1;

			f6[xc][yc][nz] = f5[xc][yc][nz];
			f8[xc][yc][nz] = f10[xc][yc][nz];
			f13[xc][yc][nz] = f11[xc][yc][nz];
			f14[xc][yc][nz] = f12[xc][yc][nz];
			f17[xc][yc][nz] = f15[xc][yc][nz];
			f18[xc][yc][nz] = f16[xc][yc][nz];

			// Undo Streaming 

			f1[xc][yc][nz] = fn1[xc][yc][nz];
			f2[xc][yc][nz] = fn2[xc][yc][nz];
			f3[xc][yc][nz] = fn3[xc][yc][nz];
			f4[xc][yc][nz] = fn4[xc][yc][nz];
			f7[xc][yc][nz] = fn7[xc][yc][nz];
			f9[xc][yc][nz] = fn9[xc][yc][nz];

			f9[xc][yc][nz] = 1.0 / 2.0*(f9[xc][yc][nz] + f7[xc][yc][nz]);
			f7[xc][yc][nz] = f9[xc][yc][nz];

			f4[xc][yc][nz] = 1.0 / 2.0*(f4[xc][yc][nz] + f2[xc][yc][nz]);
			f2[xc][yc][nz] = f4[xc][yc][nz];

			f1[xc][yc][nz] = 1.0 / 2.0*(f1[xc][yc][nz] + f3[xc][yc][nz]);
			f3[xc][yc][nz] = f1[xc][yc][nz];
		}
		#pragma omp section
		{
			//Corner14: C' as shown in the schematic

			xc = ib;
			yc = 1;

			g6[xc][yc][nz] = g5[xc][yc][nz];
			g8[xc][yc][nz] = g10[xc][yc][nz];
			g13[xc][yc][nz] = g11[xc][yc][nz];
			g14[xc][yc][nz] = g12[xc][yc][nz];
			g17[xc][yc][nz] = g15[xc][yc][nz];
			g18[xc][yc][nz] = g16[xc][yc][nz];

			// Undo Streaming 

			g1[xc][yc][nz] = gn1[xc][yc][nz];
			g2[xc][yc][nz] = gn2[xc][yc][nz];
			g3[xc][yc][nz] = gn3[xc][yc][nz];
			g4[xc][yc][nz] = gn4[xc][yc][nz];
			g7[xc][yc][nz] = gn7[xc][yc][nz];
			g9[xc][yc][nz] = gn9[xc][yc][nz];

			g9[xc][yc][nz] = 1.0 / 2.0*(g9[xc][yc][nz] + g7[xc][yc][nz]);
			g7[xc][yc][nz] = g9[xc][yc][nz];

			g4[xc][yc][nz] = 1.0 / 2.0*(g4[xc][yc][nz] + g2[xc][yc][nz]);
			g2[xc][yc][nz] = g4[xc][yc][nz];

			g1[xc][yc][nz] = 1.0 / 2.0*(g1[xc][yc][nz] + g3[xc][yc][nz]);
			g3[xc][yc][nz] = g1[xc][yc][nz];
		}
		#pragma omp section
		{
			//Corner14: C' as shown in the schematic

			xc = ib;
			yc = 1;

			h6[xc][yc][nz] = h5[xc][yc][nz];
			h8[xc][yc][nz] = h10[xc][yc][nz];
			h13[xc][yc][nz] = h11[xc][yc][nz];
			h14[xc][yc][nz] = h12[xc][yc][nz];
			h17[xc][yc][nz] = h15[xc][yc][nz];
			h18[xc][yc][nz] = h16[xc][yc][nz];

			// Undo Streaminh 

			h1[xc][yc][nz] = hn1[xc][yc][nz];
			h2[xc][yc][nz] = hn2[xc][yc][nz];
			h3[xc][yc][nz] = hn3[xc][yc][nz];
			h4[xc][yc][nz] = hn4[xc][yc][nz];
			h7[xc][yc][nz] = hn7[xc][yc][nz];
			h9[xc][yc][nz] = hn9[xc][yc][nz];

			h9[xc][yc][nz] = 1.0 / 2.0*(h9[xc][yc][nz] + h7[xc][yc][nz]);
			h7[xc][yc][nz] = h9[xc][yc][nz];

			h4[xc][yc][nz] = 1.0 / 2.0*(h4[xc][yc][nz] + h2[xc][yc][nz]);
			h2[xc][yc][nz] = h4[xc][yc][nz];

			h1[xc][yc][nz] = 1.0 / 2.0*(h1[xc][yc][nz] + h3[xc][yc][nz]);
			h3[xc][yc][nz] = h1[xc][yc][nz];
		}
		#pragma omp section
		{
			//Corner14: C' as shown in the schematic

			xc = ib;
			yc = 1;

			p6[xc][yc][nz] = p5[xc][yc][nz];
			p8[xc][yc][nz] = p10[xc][yc][nz];
			p13[xc][yc][nz] = p11[xc][yc][nz];
			p14[xc][yc][nz] = p12[xc][yc][nz];
			p17[xc][yc][nz] = p15[xc][yc][nz];
			p18[xc][yc][nz] = p16[xc][yc][nz];

			// Undo Streaminp 

			p1[xc][yc][nz] = pn1[xc][yc][nz];
			p2[xc][yc][nz] = pn2[xc][yc][nz];
			p3[xc][yc][nz] = pn3[xc][yc][nz];
			p4[xc][yc][nz] = pn4[xc][yc][nz];
			p7[xc][yc][nz] = pn7[xc][yc][nz];
			p9[xc][yc][nz] = pn9[xc][yc][nz];

			p9[xc][yc][nz] = 1.0 / 2.0*(p9[xc][yc][nz] + p7[xc][yc][nz]);
			p7[xc][yc][nz] = p9[xc][yc][nz];

			p4[xc][yc][nz] = 1.0 / 2.0*(p4[xc][yc][nz] + p2[xc][yc][nz]);
			p2[xc][yc][nz] = p4[xc][yc][nz];

			p1[xc][yc][nz] = 1.0 / 2.0*(p1[xc][yc][nz] + p3[xc][yc][nz]);
			p3[xc][yc][nz] = p1[xc][yc][nz];
		}
		#pragma omp section
		{
			// Corner 15 : C as shown in the schematic

			xc = ib;	yc = 1;

			f5[xc][yc][1] = f6[xc][yc][1];
			f8[xc][yc][1] = f10[xc][yc][1];
			f11[xc][yc][1] = f13[xc][yc][1];
			f12[xc][yc][1] = f14[xc][yc][1];
			f15[xc][yc][1] = f17[xc][yc][1];
			f16[xc][yc][1] = f18[xc][yc][1];

			//Undo Streaming

			f1[xc][yc][1] = fn1[xc][yc][1];
			f2[xc][yc][1] = fn2[xc][yc][1];
			f3[xc][yc][1] = fn3[xc][yc][1];
			f4[xc][yc][1] = fn4[xc][yc][1];
			f7[xc][yc][1] = fn7[xc][yc][1];
			f9[xc][yc][1] = fn9[xc][yc][1];

			f1[xc][yc][1] = 1.0 / 2.0*(f1[xc][yc][1] + f3[xc][yc][1]);
			f3[xc][yc][1] = f1[xc][yc][1];

			f4[xc][yc][1] = 1.0 / 2.0*(f4[xc][yc][1] + f2[xc][yc][1]);
			f2[xc][yc][1] = f4[xc][yc][1];

			f9[xc][yc][1] = 1.0 / 2.0*(f9[xc][yc][1] + f7[xc][yc][1]);
			f7[xc][yc][1] = f9[xc][yc][1];
		}
		#pragma omp section
		{
			// Corner 15 : C as shown in the schematic

			xc = ib;	yc = 1;

			g5[xc][yc][1] = g6[xc][yc][1];
			g8[xc][yc][1] = g10[xc][yc][1];
			g11[xc][yc][1] = g13[xc][yc][1];
			g12[xc][yc][1] = g14[xc][yc][1];
			g15[xc][yc][1] = g17[xc][yc][1];
			g16[xc][yc][1] = g18[xc][yc][1];

			//Undo Streaming

			g1[xc][yc][1] = gn1[xc][yc][1];
			g2[xc][yc][1] = gn2[xc][yc][1];
			g3[xc][yc][1] = gn3[xc][yc][1];
			g4[xc][yc][1] = gn4[xc][yc][1];
			g7[xc][yc][1] = gn7[xc][yc][1];
			g9[xc][yc][1] = gn9[xc][yc][1];

			g1[xc][yc][1] = 1.0 / 2.0*(g1[xc][yc][1] + g3[xc][yc][1]);
			g3[xc][yc][1] = g1[xc][yc][1];

			g4[xc][yc][1] = 1.0 / 2.0*(g4[xc][yc][1] + g2[xc][yc][1]);
			g2[xc][yc][1] = g4[xc][yc][1];

			g9[xc][yc][1] = 1.0 / 2.0*(g9[xc][yc][1] + g7[xc][yc][1]);
			g7[xc][yc][1] = g9[xc][yc][1];
		}
		#pragma omp section
		{
			// Corner 15 : C as shown in the schematic

			xc = ib;	yc = 1;

			h5[xc][yc][1] = h6[xc][yc][1];
			h8[xc][yc][1] = h10[xc][yc][1];
			h11[xc][yc][1] = h13[xc][yc][1];
			h12[xc][yc][1] = h14[xc][yc][1];
			h15[xc][yc][1] = h17[xc][yc][1];
			h16[xc][yc][1] = h18[xc][yc][1];

			//Undo Streaminh

			h1[xc][yc][1] = hn1[xc][yc][1];
			h2[xc][yc][1] = hn2[xc][yc][1];
			h3[xc][yc][1] = hn3[xc][yc][1];
			h4[xc][yc][1] = hn4[xc][yc][1];
			h7[xc][yc][1] = hn7[xc][yc][1];
			h9[xc][yc][1] = hn9[xc][yc][1];

			h1[xc][yc][1] = 1.0 / 2.0*(h1[xc][yc][1] + h3[xc][yc][1]);
			h3[xc][yc][1] = h1[xc][yc][1];

			h4[xc][yc][1] = 1.0 / 2.0*(h4[xc][yc][1] + h2[xc][yc][1]);
			h2[xc][yc][1] = h4[xc][yc][1];

			h9[xc][yc][1] = 1.0 / 2.0*(h9[xc][yc][1] + h7[xc][yc][1]);
			h7[xc][yc][1] = h9[xc][yc][1];
		}
		#pragma omp section
		{
			// Corner 15 : C as shown in the schematic

			xc = ib;	yc = 1;

			p5[xc][yc][1] = p6[xc][yc][1];
			p8[xc][yc][1] = p10[xc][yc][1];
			p11[xc][yc][1] = p13[xc][yc][1];
			p12[xc][yc][1] = p14[xc][yc][1];
			p15[xc][yc][1] = p17[xc][yc][1];
			p16[xc][yc][1] = p18[xc][yc][1];

			//Undo Streaminp

			p1[xc][yc][1] = pn1[xc][yc][1];
			p2[xc][yc][1] = pn2[xc][yc][1];
			p3[xc][yc][1] = pn3[xc][yc][1];
			p4[xc][yc][1] = pn4[xc][yc][1];
			p7[xc][yc][1] = pn7[xc][yc][1];
			p9[xc][yc][1] = pn9[xc][yc][1];

			p1[xc][yc][1] = 1.0 / 2.0*(p1[xc][yc][1] + p3[xc][yc][1]);
			p3[xc][yc][1] = p1[xc][yc][1];

			p4[xc][yc][1] = 1.0 / 2.0*(p4[xc][yc][1] + p2[xc][yc][1]);
			p2[xc][yc][1] = p4[xc][yc][1];

			p9[xc][yc][1] = 1.0 / 2.0*(p9[xc][yc][1] + p7[xc][yc][1]);
			p7[xc][yc][1] = p9[xc][yc][1];
		}
		#pragma omp section
		{
			//Corner 16: D as shown in the schematic

			xc = ia;	yc = 1;

			f5[xc][yc][1] = f6[xc][yc][1];
			f7[xc][yc][1] = f9[xc][yc][1];
			f11[xc][yc][1] = f13[xc][yc][1];
			f12[xc][yc][1] = f14[xc][yc][1];
			f15[xc][yc][1] = f17[xc][yc][1];
			f16[xc][yc][1] = f18[xc][yc][1];

			//Undo Streaming

			f1[xc][yc][1] = fn1[xc][yc][1];
			f2[xc][yc][1] = fn2[xc][yc][1];
			f3[xc][yc][1] = fn3[xc][yc][1];
			f4[xc][yc][1] = fn4[xc][yc][1];
			f8[xc][yc][1] = fn8[xc][yc][1];
			f10[xc][yc][1] = fn10[xc][yc][1];

			f1[xc][yc][1] = 1.0 / 2.0*(f1[xc][yc][1] + f3[xc][yc][1]);
			f3[xc][yc][1] = f1[xc][yc][1];

			f4[xc][yc][1] = 1.0 / 2.0*(f4[xc][yc][1] + f2[xc][yc][1]);
			f2[xc][yc][1] = f4[xc][yc][1];

			f8[xc][yc][1] = 1.0 / 2.0*(f8[xc][yc][1] + f10[xc][yc][1]);
			f10[xc][yc][1] = f8[xc][yc][1];
		}
		#pragma omp section
		{
			//Corner 16: D as shown in the schematic

			xc = ia;	yc = 1;

			g5[xc][yc][1] = g6[xc][yc][1];
			g7[xc][yc][1] = g9[xc][yc][1];
			g11[xc][yc][1] = g13[xc][yc][1];
			g12[xc][yc][1] = g14[xc][yc][1];
			g15[xc][yc][1] = g17[xc][yc][1];
			g16[xc][yc][1] = g18[xc][yc][1];

			//Undo Streaming

			g1[xc][yc][1] = gn1[xc][yc][1];
			g2[xc][yc][1] = gn2[xc][yc][1];
			g3[xc][yc][1] = gn3[xc][yc][1];
			g4[xc][yc][1] = gn4[xc][yc][1];
			g8[xc][yc][1] = gn8[xc][yc][1];
			g10[xc][yc][1] = gn10[xc][yc][1];

			g1[xc][yc][1] = 1.0 / 2.0*(g1[xc][yc][1] + g3[xc][yc][1]);
			g3[xc][yc][1] = g1[xc][yc][1];

			g4[xc][yc][1] = 1.0 / 2.0*(g4[xc][yc][1] + g2[xc][yc][1]);
			g2[xc][yc][1] = g4[xc][yc][1];

			g8[xc][yc][1] = 1.0 / 2.0*(g8[xc][yc][1] + g10[xc][yc][1]);
			g10[xc][yc][1] = g8[xc][yc][1];
		}

		#pragma omp section
		{
			//Corner 16: D as shown in the schematic

			xc = ia;	yc = 1;

			h5[xc][yc][1] = h6[xc][yc][1];
			h7[xc][yc][1] = h9[xc][yc][1];
			h11[xc][yc][1] = h13[xc][yc][1];
			h12[xc][yc][1] = h14[xc][yc][1];
			h15[xc][yc][1] = h17[xc][yc][1];
			h16[xc][yc][1] = h18[xc][yc][1];

			//Undo Streaminh

			h1[xc][yc][1] = hn1[xc][yc][1];
			h2[xc][yc][1] = hn2[xc][yc][1];
			h3[xc][yc][1] = hn3[xc][yc][1];
			h4[xc][yc][1] = hn4[xc][yc][1];
			h8[xc][yc][1] = hn8[xc][yc][1];
			h10[xc][yc][1] = hn10[xc][yc][1];

			h1[xc][yc][1] = 1.0 / 2.0*(h1[xc][yc][1] + h3[xc][yc][1]);
			h3[xc][yc][1] = h1[xc][yc][1];

			h4[xc][yc][1] = 1.0 / 2.0*(h4[xc][yc][1] + h2[xc][yc][1]);
			h2[xc][yc][1] = h4[xc][yc][1];

			h8[xc][yc][1] = 1.0 / 2.0*(h8[xc][yc][1] + h10[xc][yc][1]);
			h10[xc][yc][1] = h8[xc][yc][1];
		}
		#pragma omp section
		{
			//Corner 16: D as shown in the schematic

			xc = ia;	yc = 1;

			p5[xc][yc][1] = p6[xc][yc][1];
			p7[xc][yc][1] = p9[xc][yc][1];
			p11[xc][yc][1] = p13[xc][yc][1];
			p12[xc][yc][1] = p14[xc][yc][1];
			p15[xc][yc][1] = p17[xc][yc][1];
			p16[xc][yc][1] = p18[xc][yc][1];

			//Undo Streaminp

			p1[xc][yc][1] = pn1[xc][yc][1];
			p2[xc][yc][1] = pn2[xc][yc][1];
			p3[xc][yc][1] = pn3[xc][yc][1];
			p4[xc][yc][1] = pn4[xc][yc][1];
			p8[xc][yc][1] = pn8[xc][yc][1];
			p10[xc][yc][1] = pn10[xc][yc][1];

			p1[xc][yc][1] = 1.0 / 2.0*(p1[xc][yc][1] + p3[xc][yc][1]);
			p3[xc][yc][1] = p1[xc][yc][1];

			p4[xc][yc][1] = 1.0 / 2.0*(p4[xc][yc][1] + p2[xc][yc][1]);
			p2[xc][yc][1] = p4[xc][yc][1];

			p8[xc][yc][1] = 1.0 / 2.0*(p8[xc][yc][1] + p10[xc][yc][1]);
			p10[xc][yc][1] = p8[xc][yc][1];
		}
	}

	// Remaining edges

	//Case1 Edge AB as shown in the schematic

	//Known   : rho_out,f0,1,3,4,6,9,10,13,14,17
	//Unknown : f2,5,7,8,11,12,15,16,18
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (i = 2; i <= mnx - 1; i++)
	{
		B = 0;
		C = 0;
		A = (1.0 / 4.0)*(fm1[i][1][1] - fm3[i][1][1]);

		fm2[i][1][1] = fm4[i][1][1];
		fm5[i][1][1] = fm6[i][1][1];
		fm7[i][1][1] = fm9[i][1][1] - (A + B);
		fm8[i][1][1] = fm10[i][1][1] - (-A + B);
		fm11[i][1][1] = fm13[i][1][1] - (A + C);
		fm12[i][1][1] = fm14[i][1][1] - (-A + C);
		fm15[i][1][1] = fm17[i][1][1] - (B + C);

		fm18[i][1][1] = (1.0 / 2.0)*(rho_out - (fm0[i][1][1] + fm1[i][1][1] + fm2[i][1][1] + fm3[i][1][1] + fm4[i][1][1] + fm5[i][1][1] + fm6[i][1][1] + fm7[i][1][1] + fm8[i][1][1] + fm9[i][1][1] + fm10[i][1][1] + fm11[i][1][1] + fm12[i][1][1] + fm13[i][1][1] + fm14[i][1][1] + fm15[i][1][1] + fm17[i][1][1]));

		fm16[i][1][1] = fm18[i][1][1];
	}
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (i = 2; i <= mnx - 1; i++)
	{
		B = 0;
		C = 0;
		A = (1.0 / 4.0)*(gm1[i][1][1] - gm3[i][1][1]);

		gm2[i][1][1] = gm4[i][1][1];
		gm5[i][1][1] = gm6[i][1][1];
		gm7[i][1][1] = gm9[i][1][1] - (A + B);
		gm8[i][1][1] = gm10[i][1][1] - (-A + B);
		gm11[i][1][1] = gm13[i][1][1] - (A + C);
		gm12[i][1][1] = gm14[i][1][1] - (-A + C);
		gm15[i][1][1] = gm17[i][1][1] - (B + C);

		gm18[i][1][1] = (1.0 / 2.0)*(gmrho[i][2][1] - (gm0[i][1][1] + gm1[i][1][1] + gm2[i][1][1] + gm3[i][1][1] + gm4[i][1][1] + gm5[i][1][1] + gm6[i][1][1] + gm7[i][1][1] + gm8[i][1][1] + gm9[i][1][1] + gm10[i][1][1] + gm11[i][1][1] + gm12[i][1][1] + gm13[i][1][1] + gm14[i][1][1] + gm15[i][1][1] + gm17[i][1][1]));

		gm16[i][1][1] = gm18[i][1][1];
	}
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (i = 2; i <= mnx - 1; i++)
	{
		B = 0;
		C = 0;
		A = (1.0 / 4.0)*(hm1[i][1][1] - hm3[i][1][1]);

		hm2[i][1][1] = hm4[i][1][1];
		hm5[i][1][1] = hm6[i][1][1];
		hm7[i][1][1] = hm9[i][1][1] - (A + B);
		hm8[i][1][1] = hm10[i][1][1] - (-A + B);
		hm11[i][1][1] = hm13[i][1][1] - (A + C);
		hm12[i][1][1] = hm14[i][1][1] - (-A + C);
		hm15[i][1][1] = hm17[i][1][1] - (B + C);

		hm18[i][1][1] = (1.0 / 2.0)*(hmrho[i][2][1] - (hm0[i][1][1] + hm1[i][1][1] + hm2[i][1][1] + hm3[i][1][1] + hm4[i][1][1] + hm5[i][1][1] + hm6[i][1][1] + hm7[i][1][1] + hm8[i][1][1] + hm9[i][1][1] + hm10[i][1][1] + hm11[i][1][1] + hm12[i][1][1] + hm13[i][1][1] + hm14[i][1][1] + hm15[i][1][1] + hm17[i][1][1]));

		hm16[i][1][1] = hm18[i][1][1];
	}
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (i = 2; i <= mnx - 1; i++)
	{
		B = 0;
		C = 0;
		A = (1.0 / 4.0)*(pm1[i][1][1] - pm3[i][1][1]);

		pm2[i][1][1] = pm4[i][1][1];
		pm5[i][1][1] = pm6[i][1][1];
		pm7[i][1][1] = pm9[i][1][1] - (A + B);
		pm8[i][1][1] = pm10[i][1][1] - (-A + B);
		pm11[i][1][1] = pm13[i][1][1] - (A + C);
		pm12[i][1][1] = pm14[i][1][1] - (-A + C);
		pm15[i][1][1] = pm17[i][1][1] - (B + C);

		pm18[i][1][1] = (1.0 / 2.0)*(pmrho[i][2][1] - (pm0[i][1][1] + pm1[i][1][1] + pm2[i][1][1] + pm3[i][1][1] + pm4[i][1][1] + pm5[i][1][1] + pm6[i][1][1] + pm7[i][1][1] + pm8[i][1][1] + pm9[i][1][1] + pm10[i][1][1] + pm11[i][1][1] + pm12[i][1][1] + pm13[i][1][1] + pm14[i][1][1] + pm15[i][1][1] + pm17[i][1][1]));

		pm16[i][1][1] = pm18[i][1][1];
	}


	//Case2 Edge A'B' as shown in the schematic

	//Known   : rho_out,f0,1,3,4,5,9,10,11,12,16
	//Unknown : f2,6,7,8,13,14,15,17,18
#pragma omp parallel for collapse(1) private (A,B,C)
	for (i = 2; i <= mnx - 1; i++)
	{
		B = 0;
		C = 0;
		A = (1.0 / 4.0)*(fm1[i][1][nz] - fm3[i][1][nz]);

		fm2[i][1][nz] = fm4[i][1][nz];
		fm6[i][1][nz] = fm5[i][1][nz];
		fm7[i][1][nz] = fm9[i][1][nz] - (A + B);
		fm8[i][1][nz] = fm10[i][1][nz] - (-A + B);
		fm13[i][1][nz] = fm11[i][1][nz] - (-A - C);
		fm14[i][1][nz] = fm12[i][1][nz] - (A - C);
		fm18[i][1][nz] = fm16[i][1][nz] - (B - C);

		fm15[i][1][nz] = (1.0 / 2.0)*(rho_out - (fm0[i][1][nz] + fm1[i][1][nz] + fm2[i][1][nz] + fm3[i][1][nz] + fm4[i][1][nz] + fm5[i][1][nz] + fm6[i][1][nz] + fm7[i][1][nz] + fm8[i][1][nz] + fm9[i][1][nz] + fm10[i][1][nz] + fm11[i][1][nz] + fm12[i][1][nz] + fm13[i][1][nz] + fm14[i][1][nz] + fm16[i][1][nz] + fm18[i][1][nz]));

		fm17[i][1][nz] = fm15[i][1][nz];
	}
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (i = 2; i <= mnx - 1; i++)
	{
		B = 0;
		C = 0;
		A = (1.0 / 4.0)*(gm1[i][1][nz] - gm3[i][1][nz]);

		gm2[i][1][nz] = gm4[i][1][nz];
		gm6[i][1][nz] = gm5[i][1][nz];
		gm7[i][1][nz] = gm9[i][1][nz] - (A + B);
		gm8[i][1][nz] = gm10[i][1][nz] - (-A + B);
		gm13[i][1][nz] = gm11[i][1][nz] - (-A - C);
		gm14[i][1][nz] = gm12[i][1][nz] - (A - C);
		gm18[i][1][nz] = gm16[i][1][nz] - (B - C);

		gm15[i][1][nz] = (1.0 / 2.0)*(gmrho[i][2][nz] - (gm0[i][1][nz] + gm1[i][1][nz] + gm2[i][1][nz] + gm3[i][1][nz] + gm4[i][1][nz] + gm5[i][1][nz] + gm6[i][1][nz] + gm7[i][1][nz] + gm8[i][1][nz] + gm9[i][1][nz] + gm10[i][1][nz] + gm11[i][1][nz] + gm12[i][1][nz] + gm13[i][1][nz] + gm14[i][1][nz] + gm16[i][1][nz] + gm18[i][1][nz]));

		gm17[i][1][nz] = gm15[i][1][nz];
	}
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (i = 2; i <= mnx - 1; i++)
	{
		B = 0;
		C = 0;
		A = (1.0 / 4.0)*(hm1[i][1][nz] - hm3[i][1][nz]);

		hm2[i][1][nz] = hm4[i][1][nz];
		hm6[i][1][nz] = hm5[i][1][nz];
		hm7[i][1][nz] = hm9[i][1][nz] - (A + B);
		hm8[i][1][nz] = hm10[i][1][nz] - (-A + B);
		hm13[i][1][nz] = hm11[i][1][nz] - (-A - C);
		hm14[i][1][nz] = hm12[i][1][nz] - (A - C);
		hm18[i][1][nz] = hm16[i][1][nz] - (B - C);

		hm15[i][1][nz] = (1.0 / 2.0)*(hmrho[i][2][nz] - (hm0[i][1][nz] + hm1[i][1][nz] + hm2[i][1][nz] + hm3[i][1][nz] + hm4[i][1][nz] + hm5[i][1][nz] + hm6[i][1][nz] + hm7[i][1][nz] + hm8[i][1][nz] + hm9[i][1][nz] + hm10[i][1][nz] + hm11[i][1][nz] + hm12[i][1][nz] + hm13[i][1][nz] + hm14[i][1][nz] + hm16[i][1][nz] + hm18[i][1][nz]));

		hm17[i][1][nz] = hm15[i][1][nz];
	}
	#pragma omp parallel for collapse(1) private (A,B,C)
	for (i = 2; i <= mnx - 1; i++)
	{
		B = 0;
		C = 0;
		A = (1.0 / 4.0)*(pm1[i][1][nz] - pm3[i][1][nz]);

		pm2[i][1][nz] = pm4[i][1][nz];
		pm6[i][1][nz] = pm5[i][1][nz];
		pm7[i][1][nz] = pm9[i][1][nz] - (A + B);
		pm8[i][1][nz] = pm10[i][1][nz] - (-A + B);
		pm13[i][1][nz] = pm11[i][1][nz] - (-A - C);
		pm14[i][1][nz] = pm12[i][1][nz] - (A - C);
		pm18[i][1][nz] = pm16[i][1][nz] - (B - C);

		pm15[i][1][nz] = (1.0 / 2.0)*(pmrho[i][2][nz] - (pm0[i][1][nz] + pm1[i][1][nz] + pm2[i][1][nz] + pm3[i][1][nz] + pm4[i][1][nz] + pm5[i][1][nz] + pm6[i][1][nz] + pm7[i][1][nz] + pm8[i][1][nz] + pm9[i][1][nz] + pm10[i][1][nz] + pm11[i][1][nz] + pm12[i][1][nz] + pm13[i][1][nz] + pm14[i][1][nz] + pm16[i][1][nz] + pm18[i][1][nz]));

		pm17[i][1][nz] = pm15[i][1][nz];
	}

	// Case3 Edge AA' as shown in the schematic
	// Excluding its nodes
	// Known : ux=uy=uz=0;rho_out,f0,3,4,5,6,9,12,13,16,17
	// Unknown : f1,2,7,8,10,11,14,15,18
	#pragma omp parallel for collapse(1) private (xc,yc,A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		xc = 1;
		yc = 1;

		A = 0;
		B = 0;
		C = (1.0 / 4.0)*(fm5[xc][yc][p] - fm6[xc][yc][p]);

		fm1[xc][yc][p] = fm3[xc][yc][p];
		fm2[xc][yc][p] = fm4[xc][yc][p];
		fm7[xc][yc][p] = fm9[xc][yc][p] - (A + B);
		fm11[xc][yc][p] = fm13[xc][yc][p] - (A + C);
		fm14[xc][yc][p] = fm12[xc][yc][p] - (A - C);
		fm15[xc][yc][p] = fm17[xc][yc][p] - (B + C);
		fm18[xc][yc][p] = fm16[xc][yc][p] - (B - C);

		fm8[xc][yc][p] = (1.0 / 2.0)*(rho_out - (fm0[xc][yc][p] + fm1[xc][yc][p] + fm2[xc][yc][p] + fm3[xc][yc][p] + fm4[xc][yc][p] + fm5[xc][yc][p] + fm6[xc][yc][p] + fm7[xc][yc][p] + fm9[xc][yc][p] + fm11[xc][yc][p] + fm12[xc][yc][p] + fm13[xc][yc][p] + fm14[xc][yc][p] + fm15[xc][yc][p] + fm16[xc][yc][p] + fm17[xc][yc][p] + fm18[xc][yc][p]));

		fm10[xc][yc][p] = fm8[xc][yc][p];
	}
	#pragma omp parallel for collapse(1) private (xc,yc,A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		xc = 1;
		yc = 1;

		A = 0;
		B = 0;
		C = (1.0 / 4.0)*(gm5[xc][yc][p] - gm6[xc][yc][p]);

		gm1[xc][yc][p] = gm3[xc][yc][p];
		gm2[xc][yc][p] = gm4[xc][yc][p];
		gm7[xc][yc][p] = gm9[xc][yc][p] - (A + B);
		gm11[xc][yc][p] = gm13[xc][yc][p] - (A + C);
		gm14[xc][yc][p] = gm12[xc][yc][p] - (A - C);
		gm15[xc][yc][p] = gm17[xc][yc][p] - (B + C);
		gm18[xc][yc][p] = gm16[xc][yc][p] - (B - C);

		gm8[xc][yc][p] = (1.0 / 2.0)*(gmrho[xc][yc+1][p] - (gm0[xc][yc][p] + gm1[xc][yc][p] + gm2[xc][yc][p] + gm3[xc][yc][p] + gm4[xc][yc][p] + gm5[xc][yc][p] + gm6[xc][yc][p] + gm7[xc][yc][p] + gm9[xc][yc][p] + gm11[xc][yc][p] + gm12[xc][yc][p] + gm13[xc][yc][p] + gm14[xc][yc][p] + gm15[xc][yc][p] + gm16[xc][yc][p] + gm17[xc][yc][p] + gm18[xc][yc][p]));

		gm10[xc][yc][p] = gm8[xc][yc][p];
	}
	#pragma omp parallel for collapse(1) private (xc,yc,A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		xc = 1;
		yc = 1;

		A = 0;
		B = 0;
		C = (1.0 / 4.0)*(hm5[xc][yc][p] - hm6[xc][yc][p]);

		hm1[xc][yc][p] = hm3[xc][yc][p];
		hm2[xc][yc][p] = hm4[xc][yc][p];
		hm7[xc][yc][p] = hm9[xc][yc][p] - (A + B);
		hm11[xc][yc][p] = hm13[xc][yc][p] - (A + C);
		hm14[xc][yc][p] = hm12[xc][yc][p] - (A - C);
		hm15[xc][yc][p] = hm17[xc][yc][p] - (B + C);
		hm18[xc][yc][p] = hm16[xc][yc][p] - (B - C);

		hm8[xc][yc][p] = (1.0 / 2.0)*(hmrho[xc][yc+1][p] - (hm0[xc][yc][p] + hm1[xc][yc][p] + hm2[xc][yc][p] + hm3[xc][yc][p] + hm4[xc][yc][p] + hm5[xc][yc][p] + hm6[xc][yc][p] + hm7[xc][yc][p] + hm9[xc][yc][p] + hm11[xc][yc][p] + hm12[xc][yc][p] + hm13[xc][yc][p] + hm14[xc][yc][p] + hm15[xc][yc][p] + hm16[xc][yc][p] + hm17[xc][yc][p] + hm18[xc][yc][p]));

		hm10[xc][yc][p] = hm8[xc][yc][p];
	}
	#pragma omp parallel for collapse(1) private (xc,yc,A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		xc = 1;
		yc = 1;

		A = 0;
		B = 0;
		C = (1.0 / 4.0)*(pm5[xc][yc][p] - pm6[xc][yc][p]);

		pm1[xc][yc][p] = pm3[xc][yc][p];
		pm2[xc][yc][p] = pm4[xc][yc][p];
		pm7[xc][yc][p] = pm9[xc][yc][p] - (A + B);
		pm11[xc][yc][p] = pm13[xc][yc][p] - (A + C);
		pm14[xc][yc][p] = pm12[xc][yc][p] - (A - C);
		pm15[xc][yc][p] = pm17[xc][yc][p] - (B + C);
		pm18[xc][yc][p] = pm16[xc][yc][p] - (B - C);

		pm8[xc][yc][p] = (1.0 / 2.0)*(pmrho[xc][yc+1][p] - (pm0[xc][yc][p] + pm1[xc][yc][p] + pm2[xc][yc][p] + pm3[xc][yc][p] + pm4[xc][yc][p] + pm5[xc][yc][p] + pm6[xc][yc][p] + pm7[xc][yc][p] + pm9[xc][yc][p] + pm11[xc][yc][p] + pm12[xc][yc][p] + pm13[xc][yc][p] + pm14[xc][yc][p] + pm15[xc][yc][p] + pm16[xc][yc][p] + pm17[xc][yc][p] + pm18[xc][yc][p]));

		pm10[xc][yc][p] = pm8[xc][yc][p];
	}

	// Case4 Edge BB' as shown in the schematic
	// Excluding its nodes
	// Known : ux=uy=uz=0;rho_out,f0,1,4,5,6,10,11,14,16,17
	// Unknown : f2,3,7,8,9,12,13,15,18
	#pragma omp parallel for collapse(1) private (xc,yc,A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		xc = mnx;
		yc = 1;

		A = 0;
		B = 0;
		C = (1.0 / 4.0)*(fm5[xc][yc][p] - fm6[xc][yc][p]);

		fm3[xc][yc][p] = fm1[xc][yc][p];
		fm2[xc][yc][p] = fm4[xc][yc][p];
		fm8[xc][yc][p] = fm10[xc][yc][p] - (-A + B);

		fm12[xc][yc][p] = fm14[xc][yc][p] - (-A + C);
		fm13[xc][yc][p] = fm11[xc][yc][p] - (-C - A);
		fm15[xc][yc][p] = fm17[xc][yc][p] - (B + C);
		fm18[xc][yc][p] = fm16[xc][yc][p] - (B - C);

		fm7[xc][yc][p] = (1.0 / 2.0)*(rho_out - (fm0[xc][yc][p] + fm1[xc][yc][p] + fm2[xc][yc][p] + fm3[xc][yc][p] + fm4[xc][yc][p] + fm5[xc][yc][p] + fm6[xc][yc][p] + fm8[xc][yc][p] + fm10[xc][yc][p] + fm11[xc][yc][p] + fm12[xc][yc][p] + fm13[xc][yc][p] + fm14[xc][yc][p] + fm15[xc][yc][p] + fm16[xc][yc][p] + fm17[xc][yc][p] + fm18[xc][yc][p]));
		fm9[xc][yc][p] = fm7[xc][yc][p];
	}
	#pragma omp parallel for collapse(1) private (xc,yc,A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		xc = mnx;
		yc = 1;

		A = 0;
		B = 0;
		C = (1.0 / 4.0)*(gm5[xc][yc][p] - gm6[xc][yc][p]);

		gm3[xc][yc][p] = gm1[xc][yc][p];
		gm2[xc][yc][p] = gm4[xc][yc][p];
		gm8[xc][yc][p] = gm10[xc][yc][p] - (-A + B);

		gm12[xc][yc][p] = gm14[xc][yc][p] - (-A + C);
		gm13[xc][yc][p] = gm11[xc][yc][p] - (-C - A);
		gm15[xc][yc][p] = gm17[xc][yc][p] - (B + C);
		gm18[xc][yc][p] = gm16[xc][yc][p] - (B - C);

		gm7[xc][yc][p] = (1.0 / 2.0)*(gmrho[xc][yc+1][p] - (gm0[xc][yc][p] + gm1[xc][yc][p] + gm2[xc][yc][p] + gm3[xc][yc][p] + gm4[xc][yc][p] + gm5[xc][yc][p] + gm6[xc][yc][p] + gm8[xc][yc][p] + gm10[xc][yc][p] + gm11[xc][yc][p] + gm12[xc][yc][p] + gm13[xc][yc][p] + gm14[xc][yc][p] + gm15[xc][yc][p] + gm16[xc][yc][p] + gm17[xc][yc][p] + gm18[xc][yc][p]));
		gm9[xc][yc][p] = gm7[xc][yc][p];
	}
	#pragma omp parallel for collapse(1) private (xc,yc,A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		xc = mnx;
		yc = 1;

		A = 0;
		B = 0;
		C = (1.0 / 4.0)*(hm5[xc][yc][p] - hm6[xc][yc][p]);

		hm3[xc][yc][p] = hm1[xc][yc][p];
		hm2[xc][yc][p] = hm4[xc][yc][p];
		hm8[xc][yc][p] = hm10[xc][yc][p] - (-A + B);

		hm12[xc][yc][p] = hm14[xc][yc][p] - (-A + C);
		hm13[xc][yc][p] = hm11[xc][yc][p] - (-C - A);
		hm15[xc][yc][p] = hm17[xc][yc][p] - (B + C);
		hm18[xc][yc][p] = hm16[xc][yc][p] - (B - C);

		hm7[xc][yc][p] = (1.0 / 2.0)*(hmrho[xc][yc+1][p] - (hm0[xc][yc][p] + hm1[xc][yc][p] + hm2[xc][yc][p] + hm3[xc][yc][p] + hm4[xc][yc][p] + hm5[xc][yc][p] + hm6[xc][yc][p] + hm8[xc][yc][p] + hm10[xc][yc][p] + hm11[xc][yc][p] + hm12[xc][yc][p] + hm13[xc][yc][p] + hm14[xc][yc][p] + hm15[xc][yc][p] + hm16[xc][yc][p] + hm17[xc][yc][p] + hm18[xc][yc][p]));
		hm9[xc][yc][p] = hm7[xc][yc][p];
	}
	#pragma omp parallel for collapse(1) private (xc,yc,A,B,C)
	for (p = 2; p <= nz - 1; p++)
	{
		xc = mnx;
		yc = 1;

		A = 0;
		B = 0;
		C = (1.0 / 4.0)*(pm5[xc][yc][p] - pm6[xc][yc][p]);

		pm3[xc][yc][p] = pm1[xc][yc][p];
		pm2[xc][yc][p] = pm4[xc][yc][p];
		pm8[xc][yc][p] = pm10[xc][yc][p] - (-A + B);

		pm12[xc][yc][p] = pm14[xc][yc][p] - (-A + C);
		pm13[xc][yc][p] = pm11[xc][yc][p] - (-C - A);
		pm15[xc][yc][p] = pm17[xc][yc][p] - (B + C);
		pm18[xc][yc][p] = pm16[xc][yc][p] - (B - C);

		pm7[xc][yc][p] = (1.0 / 2.0)*(pmrho[xc][yc+1][p] - (pm0[xc][yc][p] + pm1[xc][yc][p] + pm2[xc][yc][p] + pm3[xc][yc][p] + pm4[xc][yc][p] + pm5[xc][yc][p] + pm6[xc][yc][p] + pm8[xc][yc][p] + pm10[xc][yc][p] + pm11[xc][yc][p] + pm12[xc][yc][p] + pm13[xc][yc][p] + pm14[xc][yc][p] + pm15[xc][yc][p] + pm16[xc][yc][p] + pm17[xc][yc][p] + pm18[xc][yc][p]));
		pm9[xc][yc][p] = pm7[xc][yc][p];
	}

	// Case5 Edge AD as shown in the schematic
	// Excluding its nodes
	// Known : ux=uy=uz=0;f0,2,3,4,6,8,9,12,13,17,18
	// Unknown : rho,f1,5,7,10,11,12,14,15,16
	#pragma omp parallel for collapse(1) private (xc,A,B,C)
	for (j = 2; j <= mny; j++)
	{
		xc = 1;

		A = 0.0;
		C = 0.0;
		B = (1.0 / 4.0)*(fm2[xc][j][1] - fm4[xc][j][1]);

		fm1[xc][j][1] = fm3[xc][j][1];
		fm5[xc][j][1] = fm6[xc][j][1];
		fm7[xc][j][1] = fm9[xc][j][1] - (A + B);
		fm10[xc][j][1] = fm8[xc][j][1] - (A - B);
		fm11[xc][j][1] = fm13[xc][j][1] - (A + C);
		fm15[xc][j][1] = fm17[xc][j][1] - (B + C);
		fm16[xc][j][1] = fm18[xc][j][1] - (C - B);

		//Undo Streaming 

		fm12[xc][j][1] = fmn12[xc][j][1];
		fm14[xc][j][1] = fmn14[xc][j][1];

		fm14[xc][j][1] = (1.0 / 2.0)*(fm12[xc][j][1] + fm14[xc][j][1]);
		fm12[xc][j][1] = fm14[xc][j][1];

	}
	#pragma omp parallel for collapse(1) private (xc,A,B,C)
	for (j = 2; j <= mny; j++)
	{
		xc = 1;

		A = 0.0;
		C = 0.0;
		B = (1.0 / 4.0)*(gm2[xc][j][1] - gm4[xc][j][1]);

		gm1[xc][j][1] = gm3[xc][j][1];
		gm5[xc][j][1] = gm6[xc][j][1];
		gm7[xc][j][1] = gm9[xc][j][1] - (A + B);
		gm10[xc][j][1] = gm8[xc][j][1] - (A - B);
		gm11[xc][j][1] = gm13[xc][j][1] - (A + C);
		gm15[xc][j][1] = gm17[xc][j][1] - (B + C);
		gm16[xc][j][1] = gm18[xc][j][1] - (C - B);

		//Undo Streaming 

		gm12[xc][j][1] = gmn12[xc][j][1];
		gm14[xc][j][1] = gmn14[xc][j][1];

		gm14[xc][j][1] = (1.0 / 2.0)*(gm12[xc][j][1] + gm14[xc][j][1]);
		gm12[xc][j][1] = gm14[xc][j][1];

	}
	#pragma omp parallel for collapse(1) private (xc,A,B,C)
	for (j = 2; j <= mny; j++)
	{
		xc = 1;

		A = 0.0;
		C = 0.0;
		B = (1.0 / 4.0)*(hm2[xc][j][1] - hm4[xc][j][1]);

		hm1[xc][j][1] = hm3[xc][j][1];
		hm5[xc][j][1] = hm6[xc][j][1];
		hm7[xc][j][1] = hm9[xc][j][1] - (A + B);
		hm10[xc][j][1] = hm8[xc][j][1] - (A - B);
		hm11[xc][j][1] = hm13[xc][j][1] - (A + C);
		hm15[xc][j][1] = hm17[xc][j][1] - (B + C);
		hm16[xc][j][1] = hm18[xc][j][1] - (C - B);

		//Undo Streaminh 

		hm12[xc][j][1] = hmn12[xc][j][1];
		hm14[xc][j][1] = hmn14[xc][j][1];

		hm14[xc][j][1] = (1.0 / 2.0)*(hm12[xc][j][1] + hm14[xc][j][1]);
		hm12[xc][j][1] = hm14[xc][j][1];

	}
	#pragma omp parallel for collapse(1) private (xc,A,B,C)
	for (j = 2; j <= mny; j++)
	{
		xc = 1;

		A = 0.0;
		C = 0.0;
		B = (1.0 / 4.0)*(pm2[xc][j][1] - pm4[xc][j][1]);

		pm1[xc][j][1] = pm3[xc][j][1];
		pm5[xc][j][1] = pm6[xc][j][1];
		pm7[xc][j][1] = pm9[xc][j][1] - (A + B);
		pm10[xc][j][1] = pm8[xc][j][1] - (A - B);
		pm11[xc][j][1] = pm13[xc][j][1] - (A + C);
		pm15[xc][j][1] = pm17[xc][j][1] - (B + C);
		pm16[xc][j][1] = pm18[xc][j][1] - (C - B);

		//Undo Streaminp 

		pm12[xc][j][1] = pmn12[xc][j][1];
		pm14[xc][j][1] = pmn14[xc][j][1];

		pm14[xc][j][1] = (1.0 / 2.0)*(pm12[xc][j][1] + pm14[xc][j][1]);
		pm12[xc][j][1] = pm14[xc][j][1];

	}
	// Case6 Edge A'D' as shown in the schematic
	// Excluding its nodes
	// Known : ux=uy=uz=0;f0,2,3,4,5,8,9,12,15,16
	// Unknown : rho,f1,6,7,10,(11,13),14,17,18
#pragma omp parallel for collapse(1) private (xc,A,B,C)
	for (j = 2; j <= mny; j++)
	{
		xc = 1;
		A = 0;
		B = (1.0 / 4.0)*(fm2[xc][j][nz] - fm4[xc][j][nz]);
		C = 0;

		fm1[xc][j][nz] = fm3[xc][j][nz];
		fm6[xc][j][nz] = fm5[xc][j][nz];
		fm7[xc][j][nz] = fm9[xc][j][nz] - (A + B);
		fm10[xc][j][nz] = fm8[xc][j][nz] - (A - B);
		fm14[xc][j][nz] = fm12[xc][j][nz] - (A - C);
		fm17[xc][j][nz] = fm15[xc][j][nz] - (-B - C);
		fm18[xc][j][nz] = fm16[xc][j][nz] - (B - C);

		// Undo Streaming.

		fm11[xc][j][nz] = fmn11[xc][j][nz];
		fm13[xc][j][nz] = fmn13[xc][j][nz];

		fm13[xc][j][nz] = (1.0 / 2.0)*(fm11[xc][j][nz] + fm13[xc][j][nz]);
		fm11[xc][j][nz] = fm13[xc][j][nz];
	}
	#pragma omp parallel for collapse(1) private (xc,A,B,C)
	for (j = 2; j <= mny; j++)
	{
		xc = 1;
		A = 0;
		B = (1.0 / 4.0)*(gm2[xc][j][nz] - gm4[xc][j][nz]);
		C = 0;

		gm1[xc][j][nz] = gm3[xc][j][nz];
		gm6[xc][j][nz] = gm5[xc][j][nz];
		gm7[xc][j][nz] = gm9[xc][j][nz] - (A + B);
		gm10[xc][j][nz] = gm8[xc][j][nz] - (A - B);
		gm14[xc][j][nz] = gm12[xc][j][nz] - (A - C);
		gm17[xc][j][nz] = gm15[xc][j][nz] - (-B - C);
		gm18[xc][j][nz] = gm16[xc][j][nz] - (B - C);

		// Undo Streaming.

		gm11[xc][j][nz] = gmn11[xc][j][nz];
		gm13[xc][j][nz] = gmn13[xc][j][nz];

		gm13[xc][j][nz] = (1.0 / 2.0)*(gm11[xc][j][nz] + gm13[xc][j][nz]);
		gm11[xc][j][nz] = gm13[xc][j][nz];
	}
	#pragma omp parallel for collapse(1) private (xc,A,B,C)
	for (j = 2; j <= mny; j++)
	{
		xc = 1;
		A = 0;
		B = (1.0 / 4.0)*(hm2[xc][j][nz] - hm4[xc][j][nz]);
		C = 0;

		hm1[xc][j][nz] = hm3[xc][j][nz];
		hm6[xc][j][nz] = hm5[xc][j][nz];
		hm7[xc][j][nz] = hm9[xc][j][nz] - (A + B);
		hm10[xc][j][nz] = hm8[xc][j][nz] - (A - B);
		hm14[xc][j][nz] = hm12[xc][j][nz] - (A - C);
		hm17[xc][j][nz] = hm15[xc][j][nz] - (-B - C);
		hm18[xc][j][nz] = hm16[xc][j][nz] - (B - C);

		// Undo Streaminh.

		hm11[xc][j][nz] = hmn11[xc][j][nz];
		hm13[xc][j][nz] = hmn13[xc][j][nz];

		hm13[xc][j][nz] = (1.0 / 2.0)*(hm11[xc][j][nz] + hm13[xc][j][nz]);
		hm11[xc][j][nz] = hm13[xc][j][nz];
	}
	#pragma omp parallel for collapse(1) private (xc,A,B,C)
	for (j = 2; j <= mny; j++)
	{
		xc = 1;
		A = 0;
		B = (1.0 / 4.0)*(pm2[xc][j][nz] - pm4[xc][j][nz]);
		C = 0;

		pm1[xc][j][nz] = pm3[xc][j][nz];
		pm6[xc][j][nz] = pm5[xc][j][nz];
		pm7[xc][j][nz] = pm9[xc][j][nz] - (A + B);
		pm10[xc][j][nz] = pm8[xc][j][nz] - (A - B);
		pm14[xc][j][nz] = pm12[xc][j][nz] - (A - C);
		pm17[xc][j][nz] = pm15[xc][j][nz] - (-B - C);
		pm18[xc][j][nz] = pm16[xc][j][nz] - (B - C);

		// Undo Streaminp.

		pm11[xc][j][nz] = pmn11[xc][j][nz];
		pm13[xc][j][nz] = pmn13[xc][j][nz];

		pm13[xc][j][nz] = (1.0 / 2.0)*(pm11[xc][j][nz] + pm13[xc][j][nz]);
		pm11[xc][j][nz] = pm13[xc][j][nz];
	}
	// Case7 Edge BC as shown in the schematic
	// Excluding its nodes
	// Known : ux=uy=uz=0;f0,1,2,4,6,7,10,14,17,f18
	// Unknown : rho,f3,5,8,9,(11,13),12,15,16
	#pragma omp parallel for collapse(1) private (xc,A,B,C)
	for (j = 2; j <= mny; j++)
	{
		xc = mnx;
		A = 0;
		C = 0;
		B = (1.0 / 4.0)*(fm2[xc][j][1] - fm4[xc][j][1]);

		fm3[xc][j][1] = fm1[xc][j][1];
		fm5[xc][j][1] = fm6[xc][j][1];
		fm8[xc][j][1] = fm10[xc][j][1] - (-A + B);
		fm9[xc][j][1] = fm7[xc][j][1] - (-A - B);
		fm12[xc][j][1] = fm14[xc][j][1] - (-A + C);
		fm15[xc][j][1] = fm17[xc][j][1] - (B + C);
		fm16[xc][j][1] = fm18[xc][j][1] - (-B + C);

		// Undo Streaming.

		fm11[xc][j][1] = fmn11[xc][j][1];
		fm13[xc][j][1] = fmn13[xc][j][1];

		fm13[xc][j][1] = (1.0 / 2.0)*(fm11[xc][j][1] + fm13[xc][j][1]);
		fm11[xc][j][1] = fm13[xc][j][1];
	}
	#pragma omp parallel for collapse(1) private (xc,A,B,C)
	for (j = 2; j <= mny; j++)
	{
		xc = mnx;
		A = 0;
		C = 0;
		B = (1.0 / 4.0)*(gm2[xc][j][1] - gm4[xc][j][1]);

		gm3[xc][j][1] = gm1[xc][j][1];
		gm5[xc][j][1] = gm6[xc][j][1];
		gm8[xc][j][1] = gm10[xc][j][1] - (-A + B);
		gm9[xc][j][1] = gm7[xc][j][1] - (-A - B);
		gm12[xc][j][1] = gm14[xc][j][1] - (-A + C);
		gm15[xc][j][1] = gm17[xc][j][1] - (B + C);
		gm16[xc][j][1] = gm18[xc][j][1] - (-B + C);

		// Undo Streaming.

		gm11[xc][j][1] = gmn11[xc][j][1];
		gm13[xc][j][1] = gmn13[xc][j][1];

		gm13[xc][j][1] = (1.0 / 2.0)*(gm11[xc][j][1] + gm13[xc][j][1]);
		gm11[xc][j][1] = gm13[xc][j][1];
	}
	#pragma omp parallel for collapse(1) private (xc,A,B,C)
	for (j = 2; j <= mny; j++)
	{
		xc = mnx;
		A = 0;
		C = 0;
		B = (1.0 / 4.0)*(hm2[xc][j][1] - hm4[xc][j][1]);

		hm3[xc][j][1] = hm1[xc][j][1];
		hm5[xc][j][1] = hm6[xc][j][1];
		hm8[xc][j][1] = hm10[xc][j][1] - (-A + B);
		hm9[xc][j][1] = hm7[xc][j][1] - (-A - B);
		hm12[xc][j][1] = hm14[xc][j][1] - (-A + C);
		hm15[xc][j][1] = hm17[xc][j][1] - (B + C);
		hm16[xc][j][1] = hm18[xc][j][1] - (-B + C);

		// Undo Streaminh.

		hm11[xc][j][1] = hmn11[xc][j][1];
		hm13[xc][j][1] = hmn13[xc][j][1];

		hm13[xc][j][1] = (1.0 / 2.0)*(hm11[xc][j][1] + hm13[xc][j][1]);
		hm11[xc][j][1] = hm13[xc][j][1];
	}
	#pragma omp parallel for collapse(1) private (xc,A,B,C)
	for (j = 2; j <= mny; j++)
	{
		xc = mnx;
		A = 0;
		C = 0;
		B = (1.0 / 4.0)*(pm2[xc][j][1] - pm4[xc][j][1]);

		pm3[xc][j][1] = pm1[xc][j][1];
		pm5[xc][j][1] = pm6[xc][j][1];
		pm8[xc][j][1] = pm10[xc][j][1] - (-A + B);
		pm9[xc][j][1] = pm7[xc][j][1] - (-A - B);
		pm12[xc][j][1] = pm14[xc][j][1] - (-A + C);
		pm15[xc][j][1] = pm17[xc][j][1] - (B + C);
		pm16[xc][j][1] = pm18[xc][j][1] - (-B + C);

		// Undo Streaminp.

		pm11[xc][j][1] = pmn11[xc][j][1];
		pm13[xc][j][1] = pmn13[xc][j][1];

		pm13[xc][j][1] = (1.0 / 2.0)*(pm11[xc][j][1] + pm13[xc][j][1]);
		pm11[xc][j][1] = pm13[xc][j][1];
	}

	// Case8 Edge B'C' as shown in the schematic
	// Excluding its nodes
	// Known : ux=uy=uz=0;f0,1,2,4,5,7,10,11,15,16
	// Unknown : rho,f3,6,8,9,12,13,14,17,18
	#pragma omp parallel for collapse(1) private (xc,A,B,C)
	for (j = 2; j <= mny; j++)
	{
		xc = mnx;
		A = 0;
		B = (1.0 / 4.0)*(fm2[xc][j][nz] - fm4[xc][j][nz]);
		C = 0;


		fm3[xc][j][nz] = fm1[xc][j][nz];
		fm6[xc][j][nz] = fm5[xc][j][nz];
		fm9[xc][j][nz] = fm7[xc][j][nz] - (-A - B);
		fm8[xc][j][nz] = fm10[xc][j][nz] - (-A + B);
		fm13[xc][j][nz] = fm11[xc][j][nz] - (-A - C);
		fm17[xc][j][nz] = fm15[xc][j][nz] - (-B - C);
		fm18[xc][j][nz] = fm16[xc][j][nz] - (B - C);

		// Undo Streaming.

		fm12[xc][j][nz] = fmn12[xc][j][nz];
		fm14[xc][j][nz] = fmn14[xc][j][nz];

		fm14[xc][j][nz] = (1.0 / 2.0)*(fm12[xc][j][nz] + fm14[xc][j][nz]);
		fm12[xc][j][nz] = fm14[xc][j][nz];
	}
	#pragma omp parallel for collapse(1) private (xc,A,B,C)
	for (j = 2; j <= mny; j++)
	{
		xc = mnx;
		A = 0;
		B = (1.0 / 4.0)*(gm2[xc][j][nz] - gm4[xc][j][nz]);
		C = 0;


		gm3[xc][j][nz] = gm1[xc][j][nz];
		gm6[xc][j][nz] = gm5[xc][j][nz];
		gm9[xc][j][nz] = gm7[xc][j][nz] - (-A - B);
		gm8[xc][j][nz] = gm10[xc][j][nz] - (-A + B);
		gm13[xc][j][nz] = gm11[xc][j][nz] - (-A - C);
		gm17[xc][j][nz] = gm15[xc][j][nz] - (-B - C);
		gm18[xc][j][nz] = gm16[xc][j][nz] - (B - C);

		// Undo Streaming.

		gm12[xc][j][nz] = gmn12[xc][j][nz];
		gm14[xc][j][nz] = gmn14[xc][j][nz];

		gm14[xc][j][nz] = (1.0 / 2.0)*(gm12[xc][j][nz] + gm14[xc][j][nz]);
		gm12[xc][j][nz] = gm14[xc][j][nz];
	}
	#pragma omp parallel for collapse(1) private (xc,A,B,C)
	for (j = 2; j <= mny; j++)
	{
		xc = mnx;
		A = 0;
		B = (1.0 / 4.0)*(hm2[xc][j][nz] - hm4[xc][j][nz]);
		C = 0;


		hm3[xc][j][nz] = hm1[xc][j][nz];
		hm6[xc][j][nz] = hm5[xc][j][nz];
		hm9[xc][j][nz] = hm7[xc][j][nz] - (-A - B);
		hm8[xc][j][nz] = hm10[xc][j][nz] - (-A + B);
		hm13[xc][j][nz] = hm11[xc][j][nz] - (-A - C);
		hm17[xc][j][nz] = hm15[xc][j][nz] - (-B - C);
		hm18[xc][j][nz] = hm16[xc][j][nz] - (B - C);

		// Undo Streaminh.

		hm12[xc][j][nz] = hmn12[xc][j][nz];
		hm14[xc][j][nz] = hmn14[xc][j][nz];

		hm14[xc][j][nz] = (1.0 / 2.0)*(hm12[xc][j][nz] + hm14[xc][j][nz]);
		hm12[xc][j][nz] = hm14[xc][j][nz];
	}
	#pragma omp parallel for collapse(1) private (xc,A,B,C)
	for (j = 2; j <= mny; j++)
	{
		xc = mnx;
		A = 0;
		B = (1.0 / 4.0)*(pm2[xc][j][nz] - pm4[xc][j][nz]);
		C = 0;


		pm3[xc][j][nz] = pm1[xc][j][nz];
		pm6[xc][j][nz] = pm5[xc][j][nz];
		pm9[xc][j][nz] = pm7[xc][j][nz] - (-A - B);
		pm8[xc][j][nz] = pm10[xc][j][nz] - (-A + B);
		pm13[xc][j][nz] = pm11[xc][j][nz] - (-A - C);
		pm17[xc][j][nz] = pm15[xc][j][nz] - (-B - C);
		pm18[xc][j][nz] = pm16[xc][j][nz] - (B - C);

		// Undo Streaminp.

		pm12[xc][j][nz] = pmn12[xc][j][nz];
		pm14[xc][j][nz] = pmn14[xc][j][nz];

		pm14[xc][j][nz] = (1.0 / 2.0)*(pm12[xc][j][nz] + pm14[xc][j][nz]);
		pm12[xc][j][nz] = pm14[xc][j][nz];
	}

	// Case Edge DD' as shown in the schematic
	// Excluding its nodes
	// Known : ux=uy=uz=0;f0-6,8-18 from streaming
	// Unknown : rho,f7
	#pragma omp parallel for collapse(1) private (xc,yc)
	for (p = 2; p <= nz - 1; p++)
	{
		xc = ia;
		yc = 1;

		//Undo Streaming

		f1[xc][yc][p] = fn1[xc][yc][p];
		f2[xc][yc][p] = fn2[xc][yc][p];
		f3[xc][yc][p] = fn3[xc][yc][p];
		f4[xc][yc][p] = fn4[xc][yc][p];
		f5[xc][yc][p] = fn5[xc][yc][p];
		f6[xc][yc][p] = fn6[xc][yc][p];
		f8[xc][yc][p] = fn8[xc][yc][p];

		f9[xc][yc][p] = fn9[xc][yc][p];
		f10[xc][yc][p] = fn10[xc][yc][p];
		f11[xc][yc][p] = fn11[xc][yc][p];
		f12[xc][yc][p] = fn12[xc][yc][p];
		f13[xc][yc][p] = fn13[xc][yc][p];
		f14[xc][yc][p] = fn14[xc][yc][p];
		f15[xc][yc][p] = fn15[xc][yc][p];
		f16[xc][yc][p] = fn16[xc][yc][p];
		f17[xc][yc][p] = fn17[xc][yc][p];
		f18[xc][yc][p] = fn18[xc][yc][p];

		f3[xc][yc][p] = (1.0 / 2.0)*(f3[xc][yc][p] + f1[xc][yc][p]);
		f1[xc][yc][p] = f3[xc][yc][p];

		f4[xc][yc][p] = (1.0 / 2.0)*(f4[xc][yc][p] + f2[xc][yc][p]);
		f2[xc][yc][p] = f4[xc][yc][p];

		f6[xc][yc][p] = (1.0 / 2.0)*(f5[xc][yc][p] + f6[xc][yc][p]);
		f5[xc][yc][p] = f6[xc][yc][p];


		//f7[xc][yc][p] = f9[xc][yc][p];

		f10[xc][yc][p] = (1.0 / 2.0)*(f8[xc][yc][p] + f10[xc][yc][p]);
		f8[xc][yc][p] = f10[xc][yc][p];

		f13[xc][yc][p] = (1.0 / 2.0)*(f11[xc][yc][p] + f13[xc][yc][p]);
		f11[xc][yc][p] = f13[xc][yc][p];

		f14[xc][yc][p] = (1.0 / 2.0)*(f12[xc][yc][p] + f14[xc][yc][p]);
		f12[xc][yc][p] = f14[xc][yc][p];

		f17[xc][yc][p] = (1.0 / 2.0)*(f15[xc][yc][p] + f17[xc][yc][p]);
		f15[xc][yc][p] = f17[xc][yc][p];

		f18[xc][yc][p] = (1.0 / 2.0)*(f16[xc][yc][p] + f18[xc][yc][p]);
		f16[xc][yc][p] = f18[xc][yc][p];

		f7[xc][yc][p] = (1.0 / 2.0)*(rho[xc][yc + 1][p] - (f0[xc][yc][p] + f1[xc][yc][p] + f2[xc][yc][p] + f3[xc][yc][p] + f4[xc][yc][p] + f5[xc][yc][p] + f6[xc][yc][p] + f8[xc][yc][p] + f10[xc][yc][p] + f11[xc][yc][p] + f12[xc][yc][p] + f13[xc][yc][p] + f14[xc][yc][p] + f15[xc][yc][p] + f16[xc][yc][p] + f17[xc][yc][p] + f18[xc][yc][p]));
		f9[xc][yc][p] = f7[xc][yc][p];
	}
	#pragma omp parallel for collapse(1) private (xc,yc)
	for (p = 2; p <= nz - 1; p++)
	{
		xc = ia;
		yc = 1;

		//Undo Streaming

		g1[xc][yc][p] = gn1[xc][yc][p];
		g2[xc][yc][p] = gn2[xc][yc][p];
		g3[xc][yc][p] = gn3[xc][yc][p];
		g4[xc][yc][p] = gn4[xc][yc][p];
		g5[xc][yc][p] = gn5[xc][yc][p];
		g6[xc][yc][p] = gn6[xc][yc][p];
		g8[xc][yc][p] = gn8[xc][yc][p];

		g9[xc][yc][p] = gn9[xc][yc][p];
		g10[xc][yc][p] = gn10[xc][yc][p];
		g11[xc][yc][p] = gn11[xc][yc][p];
		g12[xc][yc][p] = gn12[xc][yc][p];
		g13[xc][yc][p] = gn13[xc][yc][p];
		g14[xc][yc][p] = gn14[xc][yc][p];
		g15[xc][yc][p] = gn15[xc][yc][p];
		g16[xc][yc][p] = gn16[xc][yc][p];
		g17[xc][yc][p] = gn17[xc][yc][p];
		g18[xc][yc][p] = gn18[xc][yc][p];

		g3[xc][yc][p] = (1.0 / 2.0)*(g3[xc][yc][p] + g1[xc][yc][p]);
		g1[xc][yc][p] = g3[xc][yc][p];

		g4[xc][yc][p] = (1.0 / 2.0)*(g4[xc][yc][p] + g2[xc][yc][p]);
		g2[xc][yc][p] = g4[xc][yc][p];

		g6[xc][yc][p] = (1.0 / 2.0)*(g5[xc][yc][p] + g6[xc][yc][p]);
		g5[xc][yc][p] = g6[xc][yc][p];


		//g7[xc][yc][p] = g9[xc][yc][p];

		g10[xc][yc][p] = (1.0 / 2.0)*(g8[xc][yc][p] + g10[xc][yc][p]);
		g8[xc][yc][p] = g10[xc][yc][p];

		g13[xc][yc][p] = (1.0 / 2.0)*(g11[xc][yc][p] + g13[xc][yc][p]);
		g11[xc][yc][p] = g13[xc][yc][p];

		g14[xc][yc][p] = (1.0 / 2.0)*(g12[xc][yc][p] + g14[xc][yc][p]);
		g12[xc][yc][p] = g14[xc][yc][p];

		g17[xc][yc][p] = (1.0 / 2.0)*(g15[xc][yc][p] + g17[xc][yc][p]);
		g15[xc][yc][p] = g17[xc][yc][p];

		g18[xc][yc][p] = (1.0 / 2.0)*(g16[xc][yc][p] + g18[xc][yc][p]);
		g16[xc][yc][p] = g18[xc][yc][p];

		g7[xc][yc][p] = (1.0 / 2.0)*(grho[xc][yc + 1][p] - (g0[xc][yc][p] + g1[xc][yc][p] + g2[xc][yc][p] + g3[xc][yc][p] + g4[xc][yc][p] + g5[xc][yc][p] + g6[xc][yc][p] + g8[xc][yc][p] + g10[xc][yc][p] + g11[xc][yc][p] + g12[xc][yc][p] + g13[xc][yc][p] + g14[xc][yc][p] + g15[xc][yc][p] + g16[xc][yc][p] + g17[xc][yc][p] + g18[xc][yc][p]));
		g9[xc][yc][p] = g7[xc][yc][p];


	}
	#pragma omp parallel for collapse(1) private (xc,yc)
	for (p = 2; p <= nz - 1; p++)
	{
		xc = ia;
		yc = 1;

		//Undo Streaming

		h1[xc][yc][p] = hn1[xc][yc][p];
		h2[xc][yc][p] = hn2[xc][yc][p];
		h3[xc][yc][p] = hn3[xc][yc][p];
		h4[xc][yc][p] = hn4[xc][yc][p];
		h5[xc][yc][p] = hn5[xc][yc][p];
		h6[xc][yc][p] = hn6[xc][yc][p];
		h8[xc][yc][p] = hn8[xc][yc][p];

		h9[xc][yc][p] = hn9[xc][yc][p];
		h10[xc][yc][p] = hn10[xc][yc][p];
		h11[xc][yc][p] = hn11[xc][yc][p];
		h12[xc][yc][p] = hn12[xc][yc][p];
		h13[xc][yc][p] = hn13[xc][yc][p];
		h14[xc][yc][p] = hn14[xc][yc][p];
		h15[xc][yc][p] = hn15[xc][yc][p];
		h16[xc][yc][p] = hn16[xc][yc][p];
		h17[xc][yc][p] = hn17[xc][yc][p];
		h18[xc][yc][p] = hn18[xc][yc][p];

		h3[xc][yc][p] = (1.0 / 2.0)*(h3[xc][yc][p] + h1[xc][yc][p]);
		h1[xc][yc][p] = h3[xc][yc][p];

		h4[xc][yc][p] = (1.0 / 2.0)*(h4[xc][yc][p] + h2[xc][yc][p]);
		h2[xc][yc][p] = h4[xc][yc][p];

		h6[xc][yc][p] = (1.0 / 2.0)*(h5[xc][yc][p] + h6[xc][yc][p]);
		h5[xc][yc][p] = h6[xc][yc][p];


		//h7[xc][yc][p] = h9[xc][yc][p];

		h10[xc][yc][p] = (1.0 / 2.0)*(h8[xc][yc][p] + h10[xc][yc][p]);
		h8[xc][yc][p] = h10[xc][yc][p];

		h13[xc][yc][p] = (1.0 / 2.0)*(h11[xc][yc][p] + h13[xc][yc][p]);
		h11[xc][yc][p] = h13[xc][yc][p];

		h14[xc][yc][p] = (1.0 / 2.0)*(h12[xc][yc][p] + h14[xc][yc][p]);
		h12[xc][yc][p] = h14[xc][yc][p];

		h17[xc][yc][p] = (1.0 / 2.0)*(h15[xc][yc][p] + h17[xc][yc][p]);
		h15[xc][yc][p] = h17[xc][yc][p];

		h18[xc][yc][p] = (1.0 / 2.0)*(h16[xc][yc][p] + h18[xc][yc][p]);
		h16[xc][yc][p] = h18[xc][yc][p];

		h7[xc][yc][p] = (1.0 / 2.0)*(hrho[xc][yc + 1][p] - (h0[xc][yc][p] + h1[xc][yc][p] + h2[xc][yc][p] + h3[xc][yc][p] + h4[xc][yc][p] + h5[xc][yc][p] + h6[xc][yc][p] + h8[xc][yc][p] + h10[xc][yc][p] + h11[xc][yc][p] + h12[xc][yc][p] + h13[xc][yc][p] + h14[xc][yc][p] + h15[xc][yc][p] + h16[xc][yc][p] + h17[xc][yc][p] + h18[xc][yc][p]));
		h9[xc][yc][p] = h7[xc][yc][p];


	}
	#pragma omp parallel for collapse(1) private (xc,yc)
	for (p = 2; p <= nz - 1; p++)
	{
		xc = ia;
		yc = 1;

		//Undo Streaming

		p1[xc][yc][p] = pn1[xc][yc][p];
		p2[xc][yc][p] = pn2[xc][yc][p];
		p3[xc][yc][p] = pn3[xc][yc][p];
		p4[xc][yc][p] = pn4[xc][yc][p];
		p5[xc][yc][p] = pn5[xc][yc][p];
		p6[xc][yc][p] = pn6[xc][yc][p];
		p8[xc][yc][p] = pn8[xc][yc][p];

		p9[xc][yc][p] = pn9[xc][yc][p];
		p10[xc][yc][p] = pn10[xc][yc][p];
		p11[xc][yc][p] = pn11[xc][yc][p];
		p12[xc][yc][p] = pn12[xc][yc][p];
		p13[xc][yc][p] = pn13[xc][yc][p];
		p14[xc][yc][p] = pn14[xc][yc][p];
		p15[xc][yc][p] = pn15[xc][yc][p];
		p16[xc][yc][p] = pn16[xc][yc][p];
		p17[xc][yc][p] = pn17[xc][yc][p];
		p18[xc][yc][p] = pn18[xc][yc][p];

		p3[xc][yc][p] = (1.0 / 2.0)*(p3[xc][yc][p] + p1[xc][yc][p]);
		p1[xc][yc][p] = p3[xc][yc][p];

		p4[xc][yc][p] = (1.0 / 2.0)*(p4[xc][yc][p] + p2[xc][yc][p]);
		p2[xc][yc][p] = p4[xc][yc][p];

		p6[xc][yc][p] = (1.0 / 2.0)*(p5[xc][yc][p] + p6[xc][yc][p]);
		p5[xc][yc][p] = p6[xc][yc][p];


		//p7[xc][yc][p] = p9[xc][yc][p];

		p10[xc][yc][p] = (1.0 / 2.0)*(p8[xc][yc][p] + p10[xc][yc][p]);
		p8[xc][yc][p] = p10[xc][yc][p];

		p13[xc][yc][p] = (1.0 / 2.0)*(p11[xc][yc][p] + p13[xc][yc][p]);
		p11[xc][yc][p] = p13[xc][yc][p];

		p14[xc][yc][p] = (1.0 / 2.0)*(p12[xc][yc][p] + p14[xc][yc][p]);
		p12[xc][yc][p] = p14[xc][yc][p];

		p17[xc][yc][p] = (1.0 / 2.0)*(p15[xc][yc][p] + p17[xc][yc][p]);
		p15[xc][yc][p] = p17[xc][yc][p];

		p18[xc][yc][p] = (1.0 / 2.0)*(p16[xc][yc][p] + p18[xc][yc][p]);
		p16[xc][yc][p] = p18[xc][yc][p];

		p7[xc][yc][p] = (1.0 / 2.0)*(prho[xc][yc + 1][p] - (p0[xc][yc][p] + p1[xc][yc][p] + p2[xc][yc][p] + p3[xc][yc][p] + p4[xc][yc][p] + p5[xc][yc][p] + p6[xc][yc][p] + p8[xc][yc][p] + p10[xc][yc][p] + p11[xc][yc][p] + p12[xc][yc][p] + p13[xc][yc][p] + p14[xc][yc][p] + p15[xc][yc][p] + p16[xc][yc][p] + p17[xc][yc][p] + p18[xc][yc][p]));
		p9[xc][yc][p] = p7[xc][yc][p];


	}




	// Case Edge CC' as shown in the schematic
	// Excluding its nodes
	// Known : ux=uy=uz=0;f0-7,9-18 from streaming
	// Unknown : rho,f8
	#pragma omp parallel for collapse(1) private (xc,yc)
	for (p = 2; p <= nz - 1; p++)
	{
		xc = ib;
		yc = 1;

		//Undo Streaming

		f1[xc][yc][p] = fn1[xc][yc][p];
		f2[xc][yc][p] = fn2[xc][yc][p];
		f3[xc][yc][p] = fn3[xc][yc][p];
		f4[xc][yc][p] = fn4[xc][yc][p];
		f5[xc][yc][p] = fn5[xc][yc][p];
		f6[xc][yc][p] = fn6[xc][yc][p];
		f7[xc][yc][p] = fn7[xc][yc][p];

		f9[xc][yc][p] = fn9[xc][yc][p];
		f10[xc][yc][p] = fn10[xc][yc][p];
		f11[xc][yc][p] = fn11[xc][yc][p];
		f12[xc][yc][p] = fn12[xc][yc][p];
		f13[xc][yc][p] = fn13[xc][yc][p];
		f14[xc][yc][p] = fn14[xc][yc][p];
		f15[xc][yc][p] = fn15[xc][yc][p];
		f16[xc][yc][p] = fn16[xc][yc][p];
		f17[xc][yc][p] = fn17[xc][yc][p];
		f18[xc][yc][p] = fn18[xc][yc][p];

		f3[xc][yc][p] = (1.0 / 2.0)*(f3[xc][yc][p] + f1[xc][yc][p]);
		f1[xc][yc][p] = f3[xc][yc][p];

		f4[xc][yc][p] = (1.0 / 2.0)*(f4[xc][yc][p] + f2[xc][yc][p]);
		f2[xc][yc][p] = f4[xc][yc][p];

		f6[xc][yc][p] = (1.0 / 2.0)*(f5[xc][yc][p] + f6[xc][yc][p]);
		f5[xc][yc][p] = f6[xc][yc][p];

		f9[xc][yc][p] = (1.0 / 2.0)*(f7[xc][yc][p] + f9[xc][yc][p]);
		f7[xc][yc][p] = f9[xc][yc][p];


		//f8[xc][yc][p] = f10[xc][yc][p];

		f13[xc][yc][p] = (1.0 / 2.0)*(f11[xc][yc][p] + f13[xc][yc][p]);
		f11[xc][yc][p] = f13[xc][yc][p];

		f14[xc][yc][p] = (1.0 / 2.0)*(f12[xc][yc][p] + f14[xc][yc][p]);
		f12[xc][yc][p] = f14[xc][yc][p];

		f17[xc][yc][p] = (1.0 / 2.0)*(f15[xc][yc][p] + f17[xc][yc][p]);
		f15[xc][yc][p] = f17[xc][yc][p];

		f18[xc][yc][p] = (1.0 / 2.0)*(f16[xc][yc][p] + f18[xc][yc][p]);
		f16[xc][yc][p] = f18[xc][yc][p];

		f8[xc][yc][p] = (1.0 / 2.0)*(rho[xc][yc + 1][p] - (f0[xc][yc][p] + f1[xc][yc][p] + f2[xc][yc][p] + f3[xc][yc][p] + f4[xc][yc][p] + f5[xc][yc][p] + f6[xc][yc][p] + f7[xc][yc][p] + f9[xc][yc][p] + f11[xc][yc][p] + f12[xc][yc][p] + f13[xc][yc][p] + f14[xc][yc][p] + f15[xc][yc][p] + f16[xc][yc][p] + f17[xc][yc][p] + f18[xc][yc][p]));
		f10[xc][yc][p] = f8[xc][yc][p];

	}
	#pragma omp parallel for collapse(1) private (xc,yc)
	for (p = 2; p <= nz - 1; p++)
	{
		xc = ib;
		yc = 1;

		//Undo Streaming

		g1[xc][yc][p] = gn1[xc][yc][p];
		g2[xc][yc][p] = gn2[xc][yc][p];
		g3[xc][yc][p] = gn3[xc][yc][p];
		g4[xc][yc][p] = gn4[xc][yc][p];
		g5[xc][yc][p] = gn5[xc][yc][p];
		g6[xc][yc][p] = gn6[xc][yc][p];
		g7[xc][yc][p] = gn7[xc][yc][p];

		g9[xc][yc][p] = gn9[xc][yc][p];
		g10[xc][yc][p] = gn10[xc][yc][p];
		g11[xc][yc][p] = gn11[xc][yc][p];
		g12[xc][yc][p] = gn12[xc][yc][p];
		g13[xc][yc][p] = gn13[xc][yc][p];
		g14[xc][yc][p] = gn14[xc][yc][p];
		g15[xc][yc][p] = gn15[xc][yc][p];
		g16[xc][yc][p] = gn16[xc][yc][p];
		g17[xc][yc][p] = gn17[xc][yc][p];
		g18[xc][yc][p] = gn18[xc][yc][p];

		g3[xc][yc][p] = (1.0 / 2.0)*(g3[xc][yc][p] + g1[xc][yc][p]);
		g1[xc][yc][p] = g3[xc][yc][p];

		g4[xc][yc][p] = (1.0 / 2.0)*(g4[xc][yc][p] + g2[xc][yc][p]);
		g2[xc][yc][p] = g4[xc][yc][p];

		g6[xc][yc][p] = (1.0 / 2.0)*(g5[xc][yc][p] + g6[xc][yc][p]);
		g5[xc][yc][p] = g6[xc][yc][p];

		g9[xc][yc][p] = (1.0 / 2.0)*(g7[xc][yc][p] + g9[xc][yc][p]);
		g7[xc][yc][p] = g9[xc][yc][p];


		//g8[xc][yc][p] = g10[xc][yc][p];

		g13[xc][yc][p] = (1.0 / 2.0)*(g11[xc][yc][p] + g13[xc][yc][p]);
		g11[xc][yc][p] = g13[xc][yc][p];

		g14[xc][yc][p] = (1.0 / 2.0)*(g12[xc][yc][p] + g14[xc][yc][p]);
		g12[xc][yc][p] = g14[xc][yc][p];

		g17[xc][yc][p] = (1.0 / 2.0)*(g15[xc][yc][p] + g17[xc][yc][p]);
		g15[xc][yc][p] = g17[xc][yc][p];

		g18[xc][yc][p] = (1.0 / 2.0)*(g16[xc][yc][p] + g18[xc][yc][p]);
		g16[xc][yc][p] = g18[xc][yc][p];

		g8[xc][yc][p] = (1.0 / 2.0)*(grho[xc][yc + 1][p] - (g0[xc][yc][p] + g1[xc][yc][p] + g2[xc][yc][p] + g3[xc][yc][p] + g4[xc][yc][p] + g5[xc][yc][p] + g6[xc][yc][p] + g7[xc][yc][p] + g9[xc][yc][p] + g11[xc][yc][p] + g12[xc][yc][p] + g13[xc][yc][p] + g14[xc][yc][p] + g15[xc][yc][p] + g16[xc][yc][p] + g17[xc][yc][p] + g18[xc][yc][p]));
		g10[xc][yc][p] = g8[xc][yc][p];

	}
	#pragma omp parallel for collapse(1) private (xc,yc)
	for (p = 2; p <= nz - 1; p++)
	{
		xc = ib;
		yc = 1;

		//Undo Streaming

		h1[xc][yc][p] = hn1[xc][yc][p];
		h2[xc][yc][p] = hn2[xc][yc][p];
		h3[xc][yc][p] = hn3[xc][yc][p];
		h4[xc][yc][p] = hn4[xc][yc][p];
		h5[xc][yc][p] = hn5[xc][yc][p];
		h6[xc][yc][p] = hn6[xc][yc][p];
		h7[xc][yc][p] = hn7[xc][yc][p];

		h9[xc][yc][p] = hn9[xc][yc][p];
		h10[xc][yc][p] = hn10[xc][yc][p];
		h11[xc][yc][p] = hn11[xc][yc][p];
		h12[xc][yc][p] = hn12[xc][yc][p];
		h13[xc][yc][p] = hn13[xc][yc][p];
		h14[xc][yc][p] = hn14[xc][yc][p];
		h15[xc][yc][p] = hn15[xc][yc][p];
		h16[xc][yc][p] = hn16[xc][yc][p];
		h17[xc][yc][p] = hn17[xc][yc][p];
		h18[xc][yc][p] = hn18[xc][yc][p];

		h3[xc][yc][p] = (1.0 / 2.0)*(h3[xc][yc][p] + h1[xc][yc][p]);
		h1[xc][yc][p] = h3[xc][yc][p];

		h4[xc][yc][p] = (1.0 / 2.0)*(h4[xc][yc][p] + h2[xc][yc][p]);
		h2[xc][yc][p] = h4[xc][yc][p];

		h6[xc][yc][p] = (1.0 / 2.0)*(h5[xc][yc][p] + h6[xc][yc][p]);
		h5[xc][yc][p] = h6[xc][yc][p];

		h9[xc][yc][p] = (1.0 / 2.0)*(h7[xc][yc][p] + h9[xc][yc][p]);
		h7[xc][yc][p] = h9[xc][yc][p];


		//h8[xc][yc][p] = h10[xc][yc][p];

		h13[xc][yc][p] = (1.0 / 2.0)*(h11[xc][yc][p] + h13[xc][yc][p]);
		h11[xc][yc][p] = h13[xc][yc][p];

		h14[xc][yc][p] = (1.0 / 2.0)*(h12[xc][yc][p] + h14[xc][yc][p]);
		h12[xc][yc][p] = h14[xc][yc][p];

		h17[xc][yc][p] = (1.0 / 2.0)*(h15[xc][yc][p] + h17[xc][yc][p]);
		h15[xc][yc][p] = h17[xc][yc][p];

		h18[xc][yc][p] = (1.0 / 2.0)*(h16[xc][yc][p] + h18[xc][yc][p]);
		h16[xc][yc][p] = h18[xc][yc][p];

		h8[xc][yc][p] = (1.0 / 2.0)*(hrho[xc][yc + 1][p] - (h0[xc][yc][p] + h1[xc][yc][p] + h2[xc][yc][p] + h3[xc][yc][p] + h4[xc][yc][p] + h5[xc][yc][p] + h6[xc][yc][p] + h7[xc][yc][p] + h9[xc][yc][p] + h11[xc][yc][p] + h12[xc][yc][p] + h13[xc][yc][p] + h14[xc][yc][p] + h15[xc][yc][p] + h16[xc][yc][p] + h17[xc][yc][p] + h18[xc][yc][p]));
		h10[xc][yc][p] = h8[xc][yc][p];

	}
	#pragma omp parallel for collapse(1) private (xc,yc)
	for (p = 2; p <= nz - 1; p++)
	{
		xc = ib;
		yc = 1;

		//Undo Streaming

		p1[xc][yc][p] = pn1[xc][yc][p];
		p2[xc][yc][p] = pn2[xc][yc][p];
		p3[xc][yc][p] = pn3[xc][yc][p];
		p4[xc][yc][p] = pn4[xc][yc][p];
		p5[xc][yc][p] = pn5[xc][yc][p];
		p6[xc][yc][p] = pn6[xc][yc][p];
		p7[xc][yc][p] = pn7[xc][yc][p];

		p9[xc][yc][p] = pn9[xc][yc][p];
		p10[xc][yc][p] = pn10[xc][yc][p];
		p11[xc][yc][p] = pn11[xc][yc][p];
		p12[xc][yc][p] = pn12[xc][yc][p];
		p13[xc][yc][p] = pn13[xc][yc][p];
		p14[xc][yc][p] = pn14[xc][yc][p];
		p15[xc][yc][p] = pn15[xc][yc][p];
		p16[xc][yc][p] = pn16[xc][yc][p];
		p17[xc][yc][p] = pn17[xc][yc][p];
		p18[xc][yc][p] = pn18[xc][yc][p];

		p3[xc][yc][p] = (1.0 / 2.0)*(p3[xc][yc][p] + p1[xc][yc][p]);
		p1[xc][yc][p] = p3[xc][yc][p];

		p4[xc][yc][p] = (1.0 / 2.0)*(p4[xc][yc][p] + p2[xc][yc][p]);
		p2[xc][yc][p] = p4[xc][yc][p];

		p6[xc][yc][p] = (1.0 / 2.0)*(p5[xc][yc][p] + p6[xc][yc][p]);
		p5[xc][yc][p] = p6[xc][yc][p];

		p9[xc][yc][p] = (1.0 / 2.0)*(p7[xc][yc][p] + p9[xc][yc][p]);
		p7[xc][yc][p] = p9[xc][yc][p];


		//p8[xc][yc][p] = p10[xc][yc][p];

		p13[xc][yc][p] = (1.0 / 2.0)*(p11[xc][yc][p] + p13[xc][yc][p]);
		p11[xc][yc][p] = p13[xc][yc][p];

		p14[xc][yc][p] = (1.0 / 2.0)*(p12[xc][yc][p] + p14[xc][yc][p]);
		p12[xc][yc][p] = p14[xc][yc][p];

		p17[xc][yc][p] = (1.0 / 2.0)*(p15[xc][yc][p] + p17[xc][yc][p]);
		p15[xc][yc][p] = p17[xc][yc][p];

		p18[xc][yc][p] = (1.0 / 2.0)*(p16[xc][yc][p] + p18[xc][yc][p]);
		p16[xc][yc][p] = p18[xc][yc][p];

		p8[xc][yc][p] = (1.0 / 2.0)*(prho[xc][yc + 1][p] - (p0[xc][yc][p] + p1[xc][yc][p] + p2[xc][yc][p] + p3[xc][yc][p] + p4[xc][yc][p] + p5[xc][yc][p] + p6[xc][yc][p] + p7[xc][yc][p] + p9[xc][yc][p] + p11[xc][yc][p] + p12[xc][yc][p] + p13[xc][yc][p] + p14[xc][yc][p] + p15[xc][yc][p] + p16[xc][yc][p] + p17[xc][yc][p] + p18[xc][yc][p]));
		p10[xc][yc][p] = p8[xc][yc][p];

	}




}

/* --------------------------------------------------------------------------------------------------
Streaming of particles
f(r+dr,t+dt)=f(r,t) ;  For all points lying inside channel including boundaries.
-----------------------------------------------------------------------------------------------------*/
void streaming()
{

	int i, j, k;
	#pragma omp parallel for collapse(3)
	for (i = 1; i <= inx; i++)
	{
		for (j = 1; j <= iny; j++)
		{
			for (k = 1; k <= nz; k++)
			{
				f0[i][j][k] = fn0[i][j][k];
				f1[i][j][k] = fn1[i - 1][j][k];

				f2[i][j][k] = fn2[i][j - 1][k];// These pdf's at the interface should be from the Mixing channel

				f3[i][j][k] = fn3[i + 1][j][k];
				f4[i][j][k] = fn4[i][j + 1][k];
				f5[i][j][k] = fn5[i][j][k - 1];
				f6[i][j][k] = fn6[i][j][k + 1];

				f7[i][j][k] = fn7[i - 1][j - 1][k];// These pdf's at the interface should be from the Mixing channel
				f8[i][j][k] = fn8[i + 1][j - 1][k];// These pdf's at the interface should be from the Mixing channel

				f9[i][j][k] = fn9[i + 1][j + 1][k];
				f10[i][j][k] = fn10[i - 1][j + 1][k];
				f11[i][j][k] = fn11[i - 1][j][k - 1];
				f12[i][j][k] = fn12[i + 1][j][k - 1];
				f13[i][j][k] = fn13[i + 1][j][k + 1];
				f14[i][j][k] = fn14[i - 1][j][k + 1];

				f15[i][j][k] = fn15[i][j - 1][k - 1];// These pdf's at the interface should be from the Mixing channel

				f16[i][j][k] = fn16[i][j + 1][k - 1];
				f17[i][j][k] = fn17[i][j + 1][k + 1];

				f18[i][j][k] = fn18[i][j - 1][k + 1];// These pdf's at the interface should be from the Mixing channel

				if ((j == 1) && (i >= ia && i <= ib))
				{
					f2[i][j][k] = fmn2[(i - ia + 1)][mny][k];
					f7[i][j][k] = fmn7[(i - ia + 1) - 1][mny][k];
					f8[i][j][k] = fmn8[(i - ia + 1) + 1][mny][k];
					f15[i][j][k] = fmn15[(i - ia + 1)][mny][k - 1];
					f18[i][j][k] = fmn18[(i - ia + 1)][mny][k + 1];

				}

				g0[i][j][k] = gn0[i][j][k];
				g1[i][j][k] = gn1[i - 1][j][k];

				g2[i][j][k] = gn2[i][j - 1][k];// These pdg's at the intergace should be grom the Mixing channel

				g3[i][j][k] = gn3[i + 1][j][k];
				g4[i][j][k] = gn4[i][j + 1][k];
				g5[i][j][k] = gn5[i][j][k - 1];
				g6[i][j][k] = gn6[i][j][k + 1];

				g7[i][j][k] = gn7[i - 1][j - 1][k];// These pdg's at the intergace should be grom the Mixing channel
				g8[i][j][k] = gn8[i + 1][j - 1][k];// These pdg's at the intergace should be grom the Mixing channel

				g9[i][j][k] = gn9[i + 1][j + 1][k];
				g10[i][j][k] = gn10[i - 1][j + 1][k];
				g11[i][j][k] = gn11[i - 1][j][k - 1];
				g12[i][j][k] = gn12[i + 1][j][k - 1];
				g13[i][j][k] = gn13[i + 1][j][k + 1];
				g14[i][j][k] = gn14[i - 1][j][k + 1];

				g15[i][j][k] = gn15[i][j - 1][k - 1];// These pdg's at the intergace should be grom the Mixing channel

				g16[i][j][k] = gn16[i][j + 1][k - 1];
				g17[i][j][k] = gn17[i][j + 1][k + 1];

				g18[i][j][k] = gn18[i][j - 1][k + 1];// These pdg's at the intergace should be grom the Mixing channel

				if ((j == 1) && (i >= ia && i <= ib))
				{
					g2[i][j][k] = gmn2[(i - ia + 1)][mny][k];
					g7[i][j][k] = gmn7[(i - ia + 1) - 1][mny][k];
					g8[i][j][k] = gmn8[(i - ia + 1) + 1][mny][k];
					g15[i][j][k] = gmn15[(i - ia + 1)][mny][k - 1];
					g18[i][j][k] = gmn18[(i - ia + 1)][mny][k + 1];
				}

				h0[i][j][k] = hn0[i][j][k];
				h1[i][j][k] = hn1[i - 1][j][k];

				h2[i][j][k] = hn2[i][j - 1][k];// These pdh's at the interhace should be hrom the Mixinh channel

				h3[i][j][k] = hn3[i + 1][j][k];
				h4[i][j][k] = hn4[i][j + 1][k];
				h5[i][j][k] = hn5[i][j][k - 1];
				h6[i][j][k] = hn6[i][j][k + 1];

				h7[i][j][k] = hn7[i - 1][j - 1][k];// These pdh's at the interhace should be hrom the Mixinh channel
				h8[i][j][k] = hn8[i + 1][j - 1][k];// These pdh's at the interhace should be hrom the Mixinh channel

				h9[i][j][k] = hn9[i + 1][j + 1][k];
				h10[i][j][k] = hn10[i - 1][j + 1][k];
				h11[i][j][k] = hn11[i - 1][j][k - 1];
				h12[i][j][k] = hn12[i + 1][j][k - 1];
				h13[i][j][k] = hn13[i + 1][j][k + 1];
				h14[i][j][k] = hn14[i - 1][j][k + 1];

				h15[i][j][k] = hn15[i][j - 1][k - 1];// These pdh's at the interhace should be hrom the Mixinh channel

				h16[i][j][k] = hn16[i][j + 1][k - 1];
				h17[i][j][k] = hn17[i][j + 1][k + 1];

				h18[i][j][k] = hn18[i][j - 1][k + 1];// These pdh's at the interhace should be hrom the Mixinh channel

				if ((j == 1) && (i >= ia && i <= ib))
				{
					h2[i][j][k] = hmn2[(i - ia + 1)][mny][k];
					h7[i][j][k] = hmn7[(i - ia + 1) - 1][mny][k];
					h8[i][j][k] = hmn8[(i - ia + 1) + 1][mny][k];
					h15[i][j][k] = hmn15[(i - ia + 1)][mny][k - 1];
					h18[i][j][k] = hmn18[(i - ia + 1)][mny][k + 1];
				}

				p0[i][j][k] = pn0[i][j][k];
				p1[i][j][k] = pn1[i - 1][j][k];

				p2[i][j][k] = pn2[i][j - 1][k];// Tpese pdp's at tpe interpace spould be prom tpe Mixinp cpannel

				p3[i][j][k] = pn3[i + 1][j][k];
				p4[i][j][k] = pn4[i][j + 1][k];
				p5[i][j][k] = pn5[i][j][k - 1];
				p6[i][j][k] = pn6[i][j][k + 1];

				p7[i][j][k] = pn7[i - 1][j - 1][k];// Tpese pdp's at tpe interpace spould be prom tpe Mixinp cpannel
				p8[i][j][k] = pn8[i + 1][j - 1][k];// Tpese pdp's at tpe interpace spould be prom tpe Mixinp cpannel

				p9[i][j][k] = pn9[i + 1][j + 1][k];
				p10[i][j][k] = pn10[i - 1][j + 1][k];
				p11[i][j][k] = pn11[i - 1][j][k - 1];
				p12[i][j][k] = pn12[i + 1][j][k - 1];
				p13[i][j][k] = pn13[i + 1][j][k + 1];
				p14[i][j][k] = pn14[i - 1][j][k + 1];

				p15[i][j][k] = pn15[i][j - 1][k - 1];// Tpese pdp's at tpe interpace spould be prom tpe Mixinp cpannel

				p16[i][j][k] = pn16[i][j + 1][k - 1];
				p17[i][j][k] = pn17[i][j + 1][k + 1];

				p18[i][j][k] = pn18[i][j - 1][k + 1];// Tpese pdp's at tpe interpace spould be prom tpe Mixinp cpannel

				if ((j == 1) && (i >= ia && i <= ib))
				{
					p2[i][j][k] = pmn2[(i - ia + 1)][mny][k];
					p7[i][j][k] = pmn7[(i - ia + 1) - 1][mny][k];
					p8[i][j][k] = pmn8[(i - ia + 1) + 1][mny][k];
					p15[i][j][k] = pmn15[(i - ia + 1)][mny][k - 1];
					p18[i][j][k] = pmn18[(i - ia + 1)][mny][k + 1];
				}
			}
		}
	}
	#pragma omp parallel for collapse(3)
	for (i = 1; i <= mnx; i++)
	{
		for (j = 1; j <= mny; j++)
		{
			for (k = 1; k <= nz; k++)
			{
				fm0[i][j][k] = fmn0[i][j][k];
				fm1[i][j][k] = fmn1[i - 1][j][k];
				fm2[i][j][k] = fmn2[i][j - 1][k];
				fm3[i][j][k] = fmn3[i + 1][j][k];

				fm4[i][j][k] = fmn4[i][j + 1][k];// These pdf's at the interface should be from the Inlet Channel

				fm5[i][j][k] = fmn5[i][j][k - 1];
				fm6[i][j][k] = fmn6[i][j][k + 1];
				fm7[i][j][k] = fmn7[i - 1][j - 1][k];
				fm8[i][j][k] = fmn8[i + 1][j - 1][k];

				fm9[i][j][k] = fmn9[i + 1][j + 1][k];// These pdf's at the interface should be from the Inlet Channel
				fm10[i][j][k] = fmn10[i - 1][j + 1][k];// These pdf's at the interface should be from the Inlet Channel

				fm11[i][j][k] = fmn11[i - 1][j][k - 1];
				fm12[i][j][k] = fmn12[i + 1][j][k - 1];
				fm13[i][j][k] = fmn13[i + 1][j][k + 1];
				fm14[i][j][k] = fmn14[i - 1][j][k + 1];
				fm15[i][j][k] = fmn15[i][j - 1][k - 1];

				fm16[i][j][k] = fmn16[i][j + 1][k - 1];// These pdf's at the interface should be from the Inlet Channel
				fm17[i][j][k] = fmn17[i][j + 1][k + 1];// These pdf's at the interface should be from the Inlet Channel

				fm18[i][j][k] = fmn18[i][j - 1][k + 1];

				if (j == mny)
				{
					fm4[i][j][k] = fn4[(i + ia - 1)][1][k];
					fm9[i][j][k] = fn9[(i + ia - 1) + 1][1][k];
					fm10[i][j][k] = fn10[(i + ia - 1) - 1][1][k];
					fm16[i][j][k] = fn16[(i + ia - 1)][1][k - 1];
					fm17[i][j][k] = fn17[(i + ia - 1)][1][k + 1];
				}


				gm0[i][j][k] = gmn0[i][j][k];
				gm1[i][j][k] = gmn1[i - 1][j][k];
				gm2[i][j][k] = gmn2[i][j - 1][k];
				gm3[i][j][k] = gmn3[i + 1][j][k];

				gm4[i][j][k] = gmn4[i][j + 1][k];// These pdg's at the intergace should be grom the Inlet Channel

				gm5[i][j][k] = gmn5[i][j][k - 1];
				gm6[i][j][k] = gmn6[i][j][k + 1];
				gm7[i][j][k] = gmn7[i - 1][j - 1][k];
				gm8[i][j][k] = gmn8[i + 1][j - 1][k];

				gm9[i][j][k] = gmn9[i + 1][j + 1][k];// These pdg's at the intergace should be grom the Inlet Channel
				gm10[i][j][k] = gmn10[i - 1][j + 1][k];// These pdg's at the intergace should be grom the Inlet Channel

				gm11[i][j][k] = gmn11[i - 1][j][k - 1];
				gm12[i][j][k] = gmn12[i + 1][j][k - 1];
				gm13[i][j][k] = gmn13[i + 1][j][k + 1];
				gm14[i][j][k] = gmn14[i - 1][j][k + 1];
				gm15[i][j][k] = gmn15[i][j - 1][k - 1];

				gm16[i][j][k] = gmn16[i][j + 1][k - 1];// These pdg's at the intergace should be grom the Inlet Channel
				gm17[i][j][k] = gmn17[i][j + 1][k + 1];// These pdg's at the intergace should be grom the Inlet Channel

				gm18[i][j][k] = gmn18[i][j - 1][k + 1];

				if (j == mny)
				{
					gm4[i][j][k] = gn4[(i + ia - 1)][1][k];
					gm9[i][j][k] = gn9[(i + ia - 1) + 1][1][k];
					gm10[i][j][k] = gn10[(i + ia - 1) - 1][1][k];
					gm16[i][j][k] = gn16[(i + ia - 1)][1][k - 1];
					gm17[i][j][k] = gn17[(i + ia - 1)][1][k + 1];
				}

				hm0[i][j][k] = hmn0[i][j][k];
				hm1[i][j][k] = hmn1[i - 1][j][k];
				hm2[i][j][k] = hmn2[i][j - 1][k];
				hm3[i][j][k] = hmn3[i + 1][j][k];

				hm4[i][j][k] = hmn4[i][j + 1][k];// These pdh's at the interhace should be hrom the Inlet Channel

				hm5[i][j][k] = hmn5[i][j][k - 1];
				hm6[i][j][k] = hmn6[i][j][k + 1];
				hm7[i][j][k] = hmn7[i - 1][j - 1][k];
				hm8[i][j][k] = hmn8[i + 1][j - 1][k];

				hm9[i][j][k] = hmn9[i + 1][j + 1][k];// These pdh's at the interhace should be hrom the Inlet Channel
				hm10[i][j][k] = hmn10[i - 1][j + 1][k];// These pdh's at the interhace should be hrom the Inlet Channel

				hm11[i][j][k] = hmn11[i - 1][j][k - 1];
				hm12[i][j][k] = hmn12[i + 1][j][k - 1];
				hm13[i][j][k] = hmn13[i + 1][j][k + 1];
				hm14[i][j][k] = hmn14[i - 1][j][k + 1];
				hm15[i][j][k] = hmn15[i][j - 1][k - 1];

				hm16[i][j][k] = hmn16[i][j + 1][k - 1];// These pdh's at the interhace should be hrom the Inlet Channel
				hm17[i][j][k] = hmn17[i][j + 1][k + 1];// These pdh's at the interhace should be hrom the Inlet Channel

				hm18[i][j][k] = hmn18[i][j - 1][k + 1];

				if (j == mny)
				{
					hm4[i][j][k] = hn4[(i + ia - 1)][1][k];
					hm9[i][j][k] = hn9[(i + ia - 1) + 1][1][k];
					hm10[i][j][k] = hn10[(i + ia - 1) - 1][1][k];
					hm16[i][j][k] = hn16[(i + ia - 1)][1][k - 1];
					hm17[i][j][k] = hn17[(i + ia - 1)][1][k + 1];
				}

				pm0[i][j][k] = pmn0[i][j][k];
				pm1[i][j][k] = pmn1[i - 1][j][k];
				pm2[i][j][k] = pmn2[i][j - 1][k];
				pm3[i][j][k] = pmn3[i + 1][j][k];

				pm4[i][j][k] = pmn4[i][j + 1][k];// Tpese pdp's at tpe interpace spould be prom tpe Inlet Cpannel

				pm5[i][j][k] = pmn5[i][j][k - 1];
				pm6[i][j][k] = pmn6[i][j][k + 1];
				pm7[i][j][k] = pmn7[i - 1][j - 1][k];
				pm8[i][j][k] = pmn8[i + 1][j - 1][k];

				pm9[i][j][k] = pmn9[i + 1][j + 1][k];// Tpese pdp's at tpe interpace spould be prom tpe Inlet Cpannel
				pm10[i][j][k] = pmn10[i - 1][j + 1][k];// Tpese pdp's at tpe interpace spould be prom tpe Inlet Cpannel

				pm11[i][j][k] = pmn11[i - 1][j][k - 1];
				pm12[i][j][k] = pmn12[i + 1][j][k - 1];
				pm13[i][j][k] = pmn13[i + 1][j][k + 1];
				pm14[i][j][k] = pmn14[i - 1][j][k + 1];
				pm15[i][j][k] = pmn15[i][j - 1][k - 1];

				pm16[i][j][k] = pmn16[i][j + 1][k - 1];// Tpese pdp's at tpe interpace spould be prom tpe Inlet Cpannel
				pm17[i][j][k] = pmn17[i][j + 1][k + 1];// Tpese pdp's at tpe interpace spould be prom tpe Inlet Cpannel

				pm18[i][j][k] = pmn18[i][j - 1][k + 1];

				if (j == mny)
				{
					pm4[i][j][k] = pn4[(i + ia - 1)][1][k];
					pm9[i][j][k] = pn9[(i + ia - 1) + 1][1][k];
					pm10[i][j][k] = pn10[(i + ia - 1) - 1][1][k];
					pm16[i][j][k] = pn16[(i + ia - 1)][1][k - 1];
					pm17[i][j][k] = pn17[(i + ia - 1)][1][k + 1];
				}
			}
		}
	}
}

/*-------------------------------------------------------------------------------------------------
It calculates velocity and density by updated distribution function i.e at t=t+dt
---------------------------------------------------------------------------------------------------*/
void calculate_velocity_density()
{
	int i, j, p;
	double sumux, sumuy, sumuz;
	// CALCULATION OF DENSITY
	/*	Initializing the rho with zero values*/
	//Inlet Channel
	#pragma omp parallel for collapse(3)
	for (i = 0; i <= inx + 1; i++)
	{
		for (j = 0; j <= iny + 1; j++)
		{
			for (p = 0; p <= nz + 1; p++)
			{
				rho[i][j][p] = 0.0;
				grho[i][j][p] = 0.0;
				hrho[i][j][p] = 0.0;
				prho[i][j][p] = 0.0;

			}
		}
	}
	//Mixing Channel
	#pragma omp parallel for collapse(3)
	for (i = 0; i <= mnx + 1; i++)
	{
		for (j = 0; j <= mny + 1; j++)
		{
			for (p = 0; p <= nz + 1; p++)
			{
				mrho[i][j][p] = 0.0;
				gmrho[i][j][p] = 0.0;
				hmrho[i][j][p] = 0.0;
				pmrho[i][j][p] = 0.0;
			}
		}
	}

	/* Calculation of density */
#pragma omp parallel for collapse(3)
	for (i = 1; i <= inx; i++) // Horizontal Section
	{
		for (j = 1; j <= iny; j++)
		{
			for (p = 1; p <= nz; p++)
			{

				rho[i][j][p] =    f0[i][j][p]
								+ f1[i][j][p]
								+ f2[i][j][p]
								+ f3[i][j][p]
								+ f4[i][j][p]
								+ f5[i][j][p]
								+ f6[i][j][p]
								+ f7[i][j][p]
								+ f8[i][j][p]
								+ f9[i][j][p]
								+ f10[i][j][p]
								+ f11[i][j][p]
								+ f12[i][j][p]
								+ f13[i][j][p]
								+ f14[i][j][p]
								+ f15[i][j][p]
								+ f16[i][j][p]
								+ f17[i][j][p]
								+ f18[i][j][p];
				grho[i][j][p] =    g0[i][j][p]
								+ g1[i][j][p]
								+ g2[i][j][p]
								+ g3[i][j][p]
								+ g4[i][j][p]
								+ g5[i][j][p]
								+ g6[i][j][p]
								+ g7[i][j][p]
								+ g8[i][j][p]
								+ g9[i][j][p]
								+ g10[i][j][p]
								+ g11[i][j][p]
								+ g12[i][j][p]
								+ g13[i][j][p]
								+ g14[i][j][p]
								+ g15[i][j][p]
								+ g16[i][j][p]
								+ g17[i][j][p]
								+ g18[i][j][p];
				hrho[i][j][p] =    h0[i][j][p]
								+ h1[i][j][p]
								+ h2[i][j][p]
								+ h3[i][j][p]
								+ h4[i][j][p]
								+ h5[i][j][p]
								+ h6[i][j][p]
								+ h7[i][j][p]
								+ h8[i][j][p]
								+ h9[i][j][p]
								+ h10[i][j][p]
								+ h11[i][j][p]
								+ h12[i][j][p]
								+ h13[i][j][p]
								+ h14[i][j][p]
								+ h15[i][j][p]
								+ h16[i][j][p]
								+ h17[i][j][p]
								+ h18[i][j][p];
				prho[i][j][p] =    p0[i][j][p]
								+ p1[i][j][p]
								+ p2[i][j][p]
								+ p3[i][j][p]
								+ p4[i][j][p]
								+ p5[i][j][p]
								+ p6[i][j][p]
								+ p7[i][j][p]
								+ p8[i][j][p]
								+ p9[i][j][p]
								+ p10[i][j][p]
								+ p11[i][j][p]
								+ p12[i][j][p]
								+ p13[i][j][p]
								+ p14[i][j][p]
								+ p15[i][j][p]
								+ p16[i][j][p]
								+ p17[i][j][p]
								+ p18[i][j][p];
			}
		}
	}
#pragma omp parallel for collapse(3)
	for (i = 1; i <= mnx; i++) //Vertical leg section
	{
		for (j = 1; j <= mny; j++)
		{
			for (p = 1; p <= nz; p++)
			{
	mrho[i][j][p] = fm0[i][j][p]
					+ fm1[i][j][p]
					+ fm2[i][j][p]
					+ fm3[i][j][p]
					+ fm4[i][j][p]
					+ fm5[i][j][p]
					+ fm6[i][j][p]
					+ fm7[i][j][p]
					+ fm8[i][j][p]
					+ fm9[i][j][p]
					+ fm10[i][j][p]
					+ fm11[i][j][p]
					+ fm12[i][j][p]
					+ fm13[i][j][p]
					+ fm14[i][j][p]
					+ fm15[i][j][p]
					+ fm16[i][j][p]
					+ fm17[i][j][p]
					+ fm18[i][j][p];
	gmrho[i][j][p] = gm0[i][j][p]
					+ gm1[i][j][p]
					+ gm2[i][j][p]
					+ gm3[i][j][p]
					+ gm4[i][j][p]
					+ gm5[i][j][p]
					+ gm6[i][j][p]
					+ gm7[i][j][p]
					+ gm8[i][j][p]
					+ gm9[i][j][p]
					+ gm10[i][j][p]
					+ gm11[i][j][p]
					+ gm12[i][j][p]
					+ gm13[i][j][p]
					+ gm14[i][j][p]
					+ gm15[i][j][p]
					+ gm16[i][j][p]
					+ gm17[i][j][p]
					+ gm18[i][j][p];

	hmrho[i][j][p] = hm0[i][j][p]
					+ hm1[i][j][p]
					+ hm2[i][j][p]
					+ hm3[i][j][p]
					+ hm4[i][j][p]
					+ hm5[i][j][p]
					+ hm6[i][j][p]
					+ hm7[i][j][p]
					+ hm8[i][j][p]
					+ hm9[i][j][p]
					+ hm10[i][j][p]
					+ hm11[i][j][p]
					+ hm12[i][j][p]
					+ hm13[i][j][p]
					+ hm14[i][j][p]
					+ hm15[i][j][p]
					+ hm16[i][j][p]
					+ hm17[i][j][p]
					+ hm18[i][j][p];
	pmrho[i][j][p] = pm0[i][j][p]
					+ pm1[i][j][p]
					+ pm2[i][j][p]
					+ pm3[i][j][p]
					+ pm4[i][j][p]
					+ pm5[i][j][p]
					+ pm6[i][j][p]
					+ pm7[i][j][p]
					+ pm8[i][j][p]
					+ pm9[i][j][p]
					+ pm10[i][j][p]
					+ pm11[i][j][p]
					+ pm12[i][j][p]
					+ pm13[i][j][p]
					+ pm14[i][j][p]
					+ pm15[i][j][p]
					+ pm16[i][j][p]
					+ pm17[i][j][p]
					+ pm18[i][j][p];
			}
		}
	}
	//Forcing Inlet Velocities [Y-Z PLANE X=1,X=nx]
	#pragma omp parallel for collapse(2)
	for (j = 1; j <= iny; j++)
	{
		for (p = 1; p <= nz; p++)
		{
			ux[1][j][p] = Uin1[j][p];
			ux[inx][j][p] = Uin2[j][p];
		}
	}
	//Outlet
	#pragma omp parallel for collapse(2)
	for (i = 1; i <= mnx; i++)
	{
		for (p = 1; p <= nz; p++)
		{
			mrho[i][1][p] = rho_out;
		}
	}

	// CALCULATION OF VELOCITIES
	#pragma omp parallel for collapse(3) private(sumux,sumuy,sumuz,i,j,p)
	for (i = 1; i <= inx; i++) // Horizontal Section
	{
		for (j = 1; j <= iny; j++)
		{
			for (p = 1; p <= nz; p++)
			{
				sumux = 0.0;
				sumuy = 0.0;
				sumuz = 0.0;

				sumux = ex[0] * f0[i][j][p] + ex[1] * f1[i][j][p] + ex[2] * f2[i][j][p] + ex[3] * f3[i][j][p] + ex[4] * f4[i][j][p]
					+ ex[5] * f5[i][j][p] + ex[6] * f6[i][j][p] + ex[7] * f7[i][j][p] + ex[8] * f8[i][j][p] + ex[9] * f9[i][j][p]
					+ ex[10] * f10[i][j][p] + ex[11] * f11[i][j][p] + ex[12] * f12[i][j][p] + ex[13] * f13[i][j][p] + ex[14] * f14[i][j][p]
					+ ex[15] * f15[i][j][p] + ex[16] * f16[i][j][p] + ex[17] * f17[i][j][p] + ex[18] * f18[i][j][p];

				sumuy = ey[0] * f0[i][j][p] + ey[1] * f1[i][j][p] + ey[2] * f2[i][j][p] + ey[3] * f3[i][j][p] + ey[4] * f4[i][j][p]
					+ ey[5] * f5[i][j][p] + ey[6] * f6[i][j][p] + ey[7] * f7[i][j][p] + ey[8] * f8[i][j][p] + ey[9] * f9[i][j][p]
					+ ey[10] * f10[i][j][p] + ey[11] * f11[i][j][p] + ey[12] * f12[i][j][p] + ey[13] * f13[i][j][p] + ey[14] * f14[i][j][p]
					+ ey[15] * f15[i][j][p] + ey[16] * f16[i][j][p] + ey[17] * f17[i][j][p] + ey[18] * f18[i][j][p];

				sumuz = ez[0] * f0[i][j][p] + ez[1] * f1[i][j][p] + ez[2] * f2[i][j][p] + ez[3] * f3[i][j][p] + ez[4] * f4[i][j][p]
					+ ez[5] * f5[i][j][p] + ez[6] * f6[i][j][p] + ez[7] * f7[i][j][p] + ez[8] * f8[i][j][p] + ez[9] * f9[i][j][p]
					+ ez[10] * f10[i][j][p] + ez[11] * f11[i][j][p] + ez[12] * f12[i][j][p] + ez[13] * f13[i][j][p] + ez[14] * f14[i][j][p]
					+ ez[15] * f15[i][j][p] + ez[16] * f16[i][j][p] + ez[17] * f17[i][j][p] + ez[18] * f18[i][j][p];

				ux[i][j][p] = sumux / rho[i][j][p];
				uy[i][j][p] = sumuy / rho[i][j][p];
				uz[i][j][p] = sumuz / rho[i][j][p];
			}
		}
	}
    #pragma omp parallel for collapse(3)  private(sumux,sumuy,sumuz,i,j,p)
	for (i = 1; i <= mnx; i++) //Vertical leg section
	{
		for (j = 1; j <= mny; j++)
		{
			for (p = 1; p <= nz; p++)
			{
				sumux = 0.0;
				sumuy = 0.0;
				sumuz = 0.0;

				sumux = ex[0] * fm0[i][j][p] + ex[1] * fm1[i][j][p] + ex[2] * fm2[i][j][p] + ex[3] * fm3[i][j][p] + ex[4] * fm4[i][j][p]
					+ ex[5] * fm5[i][j][p] + ex[6] * fm6[i][j][p] + ex[7] * fm7[i][j][p] + ex[8] * fm8[i][j][p] + ex[9] * fm9[i][j][p]
					+ ex[10] * fm10[i][j][p] + ex[11] * fm11[i][j][p] + ex[12] * fm12[i][j][p] + ex[13] * fm13[i][j][p]
					+ ex[14] * fm14[i][j][p] + ex[15] * fm15[i][j][p] + ex[16] * fm16[i][j][p] + ex[17] * fm17[i][j][p]
					+ ex[18] * fm18[i][j][p];

				sumuy = ey[0] * fm0[i][j][p] + ey[1] * fm1[i][j][p] + ey[2] * fm2[i][j][p] + ey[3] * fm3[i][j][p] + ey[4] * fm4[i][j][p]
					+ ey[5] * fm5[i][j][p] + ey[6] * fm6[i][j][p] + ey[7] * fm7[i][j][p] + ey[8] * fm8[i][j][p] + ey[9] * fm9[i][j][p]
					+ ey[10] * fm10[i][j][p] + ey[11] * fm11[i][j][p] + ey[12] * fm12[i][j][p] + ey[13] * fm13[i][j][p]
					+ ey[14] * fm14[i][j][p] + ey[15] * fm15[i][j][p] + ey[16] * fm16[i][j][p] + ey[17] * fm17[i][j][p]
					+ ey[18] * fm18[i][j][p];

				sumuz = ez[0] * fm0[i][j][p] + ez[1] * fm1[i][j][p] + ez[2] * fm2[i][j][p] + ez[3] * fm3[i][j][p] + ez[4] * fm4[i][j][p]
					+ ez[5] * fm5[i][j][p] + ez[6] * fm6[i][j][p] + ez[7] * fm7[i][j][p] + ez[8] * fm8[i][j][p] + ez[9] * fm9[i][j][p]
					+ ez[10] * fm10[i][j][p] + ez[11] * fm11[i][j][p] + ez[12] * fm12[i][j][p] + ez[13] * fm13[i][j][p]
					+ ez[14] * fm14[i][j][p] + ez[15] * fm15[i][j][p] + ez[16] * fm16[i][j][p] + ez[17] * fm17[i][j][p]
					+ ez[18] * fm18[i][j][p];

				mux[i][j][p] = sumux / mrho[i][j][p]; // Changed from rho,ux,uy,uz to mrho,mux,muy,muz in mixing channel
				muy[i][j][p] = sumuy / mrho[i][j][p];
				muz[i][j][p] = sumuz / mrho[i][j][p];
			}
		}
	}



}

/*-----------------------------------------------------------------------------
Calculation of distribution function from boltzmann transport eqn
------------------------------------------------------------------------------*/
void collision()
{
	//f'(r,t)=f(r,t)-1/tau(f(r,t)-feq(r,t)) this is due to collision only
	//f(r+dr,t+dt)=f(r,t) Particle undergoes streaming just after collision
	int i, j, k;
	
	double Ma = 1.0,Mb = 1.0,Mp = 2.0; // Molecular weight of the species 
	// Horizontal section along the inlet channels GFHE(GFHE)'
	#pragma omp parallel for collapse(3)
	for (i = 0; i <= inx + 1; i++)
	{
		for (j = 0; j <= iny + 1; j++)
		{
			for (k = 0; k <= nz + 1; k++)
			{
				
				fn0[i][j][k] = f0[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq0[i][j][k];
				fn1[i][j][k] = f1[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq1[i][j][k];
				fn2[i][j][k] = f2[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq2[i][j][k];
				fn3[i][j][k] = f3[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq3[i][j][k];
				fn4[i][j][k] = f4[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq4[i][j][k];
				fn5[i][j][k] = f5[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq5[i][j][k];
				fn6[i][j][k] = f6[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq6[i][j][k];
				fn7[i][j][k] = f7[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq7[i][j][k];
				fn8[i][j][k] = f8[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq8[i][j][k];
				fn9[i][j][k] = f9[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq9[i][j][k];
				fn10[i][j][k] = f10[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq10[i][j][k];
				fn11[i][j][k] = f11[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq11[i][j][k];
				fn12[i][j][k] = f12[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq12[i][j][k];
				fn13[i][j][k] = f13[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq13[i][j][k];
				fn14[i][j][k] = f14[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq14[i][j][k];
				fn15[i][j][k] = f15[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq15[i][j][k];
				fn16[i][j][k] = f16[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq16[i][j][k];
				fn17[i][j][k] = f17[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq17[i][j][k];
				fn18[i][j][k] = f18[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * feq18[i][j][k];

			
				double Ca = grho[i][j][k]/Ma;
				double Cb = hrho[i][j][k]/Mb;				
				double reactionterm = rateconst*Ca*Cb;
				

				gn0[i][j][k] = g0[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq0[i][j][k] -weights[0]*reactionterm*Ma;// /rho0;
				gn1[i][j][k] = g1[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq1[i][j][k] -weights[1]*reactionterm*Ma;// /rho0;
				gn2[i][j][k] = g2[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq2[i][j][k] -weights[2]*reactionterm*Ma;// /rho0;
				gn3[i][j][k] = g3[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq3[i][j][k] -weights[3]*reactionterm*Ma;// /rho0;
				gn4[i][j][k] = g4[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq4[i][j][k] -weights[4]*reactionterm*Ma;// /rho0;
				gn5[i][j][k] = g5[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq5[i][j][k] -weights[5]*reactionterm*Ma;// /rho0;
				gn6[i][j][k] = g6[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq6[i][j][k] -weights[6]*reactionterm*Ma;// /rho0;
				gn7[i][j][k] = g7[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq7[i][j][k] -weights[7]*reactionterm*Ma;// /rho0;
				gn8[i][j][k] = g8[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq8[i][j][k] -weights[8]*reactionterm*Ma;// /rho0;
				gn9[i][j][k] = g9[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq9[i][j][k] -weights[9]*reactionterm*Ma;// /rho0;
				gn10[i][j][k] = g10[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq10[i][j][k] -weights[10]*reactionterm*Ma;// /rho0;
				gn11[i][j][k] = g11[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq11[i][j][k] -weights[11]*reactionterm*Ma;// /rho0;
				gn12[i][j][k] = g12[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq12[i][j][k] -weights[12]*reactionterm*Ma;// /rho0;
				gn13[i][j][k] = g13[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq13[i][j][k] -weights[13]*reactionterm*Ma;// /rho0;
				gn14[i][j][k] = g14[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq14[i][j][k] -weights[14]*reactionterm*Ma;// /rho0;
				gn15[i][j][k] = g15[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq15[i][j][k] -weights[15]*reactionterm*Ma;// /rho0;
				gn16[i][j][k] = g16[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq16[i][j][k] -weights[16]*reactionterm*Ma;// /rho0;
				gn17[i][j][k] = g17[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq17[i][j][k] -weights[17]*reactionterm*Ma;// /rho0;
				gn18[i][j][k] = g18[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * geq18[i][j][k] -weights[18]*reactionterm*Ma;// /rho0;

				hn0[i][j][k] = h0[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq0[i][j][k]  -weights[0]*reactionterm*Mb;// /rho0;
				hn1[i][j][k] = h1[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq1[i][j][k]  -weights[1]*reactionterm*Mb;// /rho0;
				hn2[i][j][k] = h2[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq2[i][j][k]  -weights[2]*reactionterm*Mb;// /rho0;
				hn3[i][j][k] = h3[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq3[i][j][k]  -weights[3]*reactionterm*Mb;// /rho0;
				hn4[i][j][k] = h4[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq4[i][j][k]  -weights[4]*reactionterm*Mb;// /rho0;
				hn5[i][j][k] = h5[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq5[i][j][k]  -weights[5]*reactionterm*Mb;// /rho0;
				hn6[i][j][k] = h6[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq6[i][j][k]  -weights[6]*reactionterm*Mb;// /rho0;
				hn7[i][j][k] = h7[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq7[i][j][k]  -weights[7]*reactionterm*Mb;// /rho0;
				hn8[i][j][k] = h8[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq8[i][j][k]  -weights[8]*reactionterm*Mb;// /rho0;
				hn9[i][j][k] = h9[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq9[i][j][k]  -weights[9]*reactionterm*Mb;// /rho0;
				hn10[i][j][k] = h10[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq10[i][j][k] -weights[10]*reactionterm*Mb;// /rho0;
				hn11[i][j][k] = h11[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq11[i][j][k] -weights[11]*reactionterm*Mb;// /rho0;
				hn12[i][j][k] = h12[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq12[i][j][k] -weights[12]*reactionterm*Mb;// /rho0;
				hn13[i][j][k] = h13[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq13[i][j][k] -weights[13]*reactionterm*Mb;// /rho0;
				hn14[i][j][k] = h14[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq14[i][j][k] -weights[14]*reactionterm*Mb;// /rho0;
				hn15[i][j][k] = h15[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq15[i][j][k] -weights[15]*reactionterm*Mb;// /rho0;
				hn16[i][j][k] = h16[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq16[i][j][k] -weights[16]*reactionterm*Mb;// /rho0;
				hn17[i][j][k] = h17[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq17[i][j][k] -weights[17]*reactionterm*Mb;// /rho0;
				hn18[i][j][k] = h18[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * heq18[i][j][k] -weights[18]*reactionterm*Mb;// /rho0;

				pn0[i][j][k] = p0[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq0[i][j][k]  +weights[0]*reactionterm*Mp;// /rho0;
				pn1[i][j][k] = p1[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq1[i][j][k]  +weights[1]*reactionterm*Mp;// /rho0;
				pn2[i][j][k] = p2[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq2[i][j][k]  +weights[2]*reactionterm*Mp;// /rho0;
				pn3[i][j][k] = p3[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq3[i][j][k]  +weights[3]*reactionterm*Mp;// /rho0;
				pn4[i][j][k] = p4[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq4[i][j][k]  +weights[4]*reactionterm*Mp;// /rho0;
				pn5[i][j][k] = p5[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq5[i][j][k]  +weights[5]*reactionterm*Mp;// /rho0;
				pn6[i][j][k] = p6[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq6[i][j][k]  +weights[6]*reactionterm*Mp;// /rho0;
				pn7[i][j][k] = p7[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq7[i][j][k]  +weights[7]*reactionterm*Mp;// /rho0;
				pn8[i][j][k] = p8[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq8[i][j][k]  +weights[8]*reactionterm*Mp;// /rho0;
				pn9[i][j][k] = p9[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq9[i][j][k]  +weights[9]*reactionterm*Mp;// /rho0;
				pn10[i][j][k] = p10[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq10[i][j][k] +weights[10]*reactionterm*Mp;// /rho0;
				pn11[i][j][k] = p11[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq11[i][j][k] +weights[11]*reactionterm*Mp;// /rho0;
				pn12[i][j][k] = p12[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq12[i][j][k] +weights[12]*reactionterm*Mp;// /rho0;
				pn13[i][j][k] = p13[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq13[i][j][k] +weights[13]*reactionterm*Mp;// /rho0;
				pn14[i][j][k] = p14[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq14[i][j][k] +weights[14]*reactionterm*Mp;// /rho0;
				pn15[i][j][k] = p15[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq15[i][j][k] +weights[15]*reactionterm*Mp;// /rho0;
				pn16[i][j][k] = p16[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq16[i][j][k] +weights[16]*reactionterm*Mp;// /rho0;
				pn17[i][j][k] = p17[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq17[i][j][k] +weights[17]*reactionterm*Mp;// /rho0;
				pn18[i][j][k] = p18[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup) * peq18[i][j][k] +weights[18]*reactionterm*Mp;// /rho0;
			}
		}
	}
	// Vertical section along the mixing channels ABCD(ABCD)'
	#pragma omp parallel for collapse(3)
	for (i = 0; i <= mnx + 1; i++)
	{
		for (j = 0; j <= mny + 1; j++)
		{
			for (k = 0; k <= nz + 1; k++)
			{
				fmn0[i][j][k] = fm0[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq0[i][j][k] ;
				fmn1[i][j][k] = fm1[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq1[i][j][k] ;
				fmn2[i][j][k] = fm2[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq2[i][j][k] ;
				fmn3[i][j][k] = fm3[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq3[i][j][k] ;
				fmn4[i][j][k] = fm4[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq4[i][j][k] ;
				fmn5[i][j][k] = fm5[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq5[i][j][k] ;
				fmn6[i][j][k] = fm6[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq6[i][j][k] ;
				fmn7[i][j][k] = fm7[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq7[i][j][k] ;
				fmn8[i][j][k] = fm8[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq8[i][j][k] ;
				fmn9[i][j][k] = fm9[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq9[i][j][k] ;
				fmn10[i][j][k] = fm10[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq10[i][j][k] ;
				fmn11[i][j][k] = fm11[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq11[i][j][k] ;
				fmn12[i][j][k] = fm12[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq12[i][j][k] ;
				fmn13[i][j][k] = fm13[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq13[i][j][k] ;
				fmn14[i][j][k] = fm14[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq14[i][j][k] ;
				fmn15[i][j][k] = fm15[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq15[i][j][k] ;
				fmn16[i][j][k] = fm16[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq16[i][j][k] ;
				fmn17[i][j][k] = fm17[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq17[i][j][k] ;
				fmn18[i][j][k] = fm18[i][j][k] * (1.0 - 1.0 / tau) + (1.0 / tau) * fmeq18[i][j][k] ;

				
				double Ca = gmrho[i][j][k]/Ma;
				double Cb = hmrho[i][j][k]/Mb;				
				double reactionterm = rateconst*Ca*Cb;
				
				gmn0[i][j][k]  = gm0[i][j][k]  * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq0[i][j][k]  - weights[0]*reactionterm*Ma;// /rho0;//delt is 1
				gmn1[i][j][k]  = gm1[i][j][k]  * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq1[i][j][k]  - weights[1]*reactionterm*Ma;// /rho0;//delt is 1
				gmn2[i][j][k]  = gm2[i][j][k]  * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq2[i][j][k]  - weights[2]*reactionterm*Ma;// /rho0;//delt is 1
				gmn3[i][j][k]  = gm3[i][j][k]  * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq3[i][j][k]  - weights[3]*reactionterm*Ma;// /rho0;//delt is 1
				gmn4[i][j][k]  = gm4[i][j][k]  * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq4[i][j][k]  - weights[4]*reactionterm*Ma;// /rho0;//delt is 1
				gmn5[i][j][k]  = gm5[i][j][k]  * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq5[i][j][k]  - weights[5]*reactionterm*Ma;// /rho0;//delt is 1
				gmn6[i][j][k]  = gm6[i][j][k]  * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq6[i][j][k]  - weights[6]*reactionterm*Ma;// /rho0;//delt is 1
				gmn7[i][j][k]  = gm7[i][j][k]  * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq7[i][j][k]  - weights[7]*reactionterm*Ma;// /rho0;//delt is 1
				gmn8[i][j][k]  = gm8[i][j][k]  * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq8[i][j][k]  - weights[8]*reactionterm*Ma;// /rho0;//delt is 1
				gmn9[i][j][k]  = gm9[i][j][k]  * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq9[i][j][k]  - weights[9]*reactionterm*Ma;// /rho0;//delt is 1
				gmn10[i][j][k] = gm10[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq10[i][j][k] - weights[10]*reactionterm*Ma;// /rho0;//delt is 1
				gmn11[i][j][k] = gm11[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq11[i][j][k] - weights[11]*reactionterm*Ma;// /rho0;//delt is 1
				gmn12[i][j][k] = gm12[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq12[i][j][k] - weights[12]*reactionterm*Ma;// /rho0;//delt is 1
				gmn13[i][j][k] = gm13[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq13[i][j][k] - weights[13]*reactionterm*Ma;// /rho0;//delt is 1
				gmn14[i][j][k] = gm14[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq14[i][j][k] - weights[14]*reactionterm*Ma;// /rho0;//delt is 1
				gmn15[i][j][k] = gm15[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq15[i][j][k] - weights[15]*reactionterm*Ma;// /rho0;//delt is 1
				gmn16[i][j][k] = gm16[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq16[i][j][k] - weights[16]*reactionterm*Ma;// /rho0;//delt is 1
				gmn17[i][j][k] = gm17[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq17[i][j][k] - weights[17]*reactionterm*Ma;// /rho0;//delt is 1
				gmn18[i][j][k] = gm18[i][j][k] * (1.0 - 1.0 / taua) + (1.0 / taua) * gmeq18[i][j][k] - weights[18]*reactionterm*Ma;// /rho0;//delt is 1
 
				hmn0[i][j][k]  = hm0[i][j][k]  * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq0[i][j][k]   - weights[0]*reactionterm*Mb;// /rho0;//delt is 1
				hmn1[i][j][k]  = hm1[i][j][k]  * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq1[i][j][k]   - weights[1]*reactionterm*Mb;// /rho0;//delt is 1
				hmn2[i][j][k]  = hm2[i][j][k]  * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq2[i][j][k]   - weights[2]*reactionterm*Mb;// /rho0;//delt is 1
				hmn3[i][j][k]  = hm3[i][j][k]  * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq3[i][j][k]   - weights[3]*reactionterm*Mb;// /rho0;//delt is 1
				hmn4[i][j][k]  = hm4[i][j][k]  * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq4[i][j][k]   - weights[4]*reactionterm*Mb;// /rho0;//delt is 1
				hmn5[i][j][k]  = hm5[i][j][k]  * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq5[i][j][k]   - weights[5]*reactionterm*Mb;// /rho0;//delt is 1
				hmn6[i][j][k]  = hm6[i][j][k]  * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq6[i][j][k]   - weights[6]*reactionterm*Mb;// /rho0;//delt is 1
				hmn7[i][j][k]  = hm7[i][j][k]  * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq7[i][j][k]   - weights[7]*reactionterm*Mb;// /rho0;
				hmn8[i][j][k]  = hm8[i][j][k]  * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq8[i][j][k]   - weights[8]*reactionterm*Mb;// /rho0;
				hmn9[i][j][k]  = hm9[i][j][k]  * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq9[i][j][k]   - weights[9]*reactionterm*Mb;// /rho0;
				hmn10[i][j][k] = hm10[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq10[i][j][k]  - weights[10]*reactionterm*Mb;// /rho0;
				hmn11[i][j][k] = hm11[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq11[i][j][k]  - weights[11]*reactionterm*Mb;// /rho0;
				hmn12[i][j][k] = hm12[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq12[i][j][k]  - weights[12]*reactionterm*Mb;// /rho0;
				hmn13[i][j][k] = hm13[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq13[i][j][k]  - weights[13]*reactionterm*Mb;// /rho0;
				hmn14[i][j][k] = hm14[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq14[i][j][k]  - weights[14]*reactionterm*Mb;// /rho0;
				hmn15[i][j][k] = hm15[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq15[i][j][k]  - weights[15]*reactionterm*Mb;// /rho0;
				hmn16[i][j][k] = hm16[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq16[i][j][k]  - weights[16]*reactionterm*Mb;// /rho0;
				hmn17[i][j][k] = hm17[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq17[i][j][k]  - weights[17]*reactionterm*Mb;// /rho0;
				hmn18[i][j][k] = hm18[i][j][k] * (1.0 - 1.0 / taub) + (1.0 / taub) * hmeq18[i][j][k]  - weights[18]*reactionterm*Mb;// /rho0;
 
				pmn0[i][j][k]  =pm0[i][j][k]  * (1.0 - 1.0 / taup) + (1.0 / taup) * pmeq0[i][j][k] +weights[0]*reactionterm*Mp;// /rho0;
				pmn1[i][j][k]  =pm1[i][j][k]  * (1.0 - 1.0 / taup) + (1.0 / taup) * pmeq1[i][j][k] +weights[1]*reactionterm*Mp;// /rho0;
				pmn2[i][j][k]  =pm2[i][j][k]  * (1.0 - 1.0 / taup) + (1.0 / taup) * pmeq2[i][j][k] +weights[2]*reactionterm*Mp;// /rho0;
				pmn3[i][j][k]  =pm3[i][j][k]  * (1.0 - 1.0 / taup) + (1.0 / taup) * pmeq3[i][j][k] +weights[3]*reactionterm*Mp;// /rho0;
				pmn4[i][j][k]  =pm4[i][j][k]  * (1.0 - 1.0 / taup) + (1.0 / taup) * pmeq4[i][j][k] +weights[4]*reactionterm*Mp;// /rho0;
				pmn5[i][j][k]  =pm5[i][j][k]  * (1.0 - 1.0 / taup) + (1.0 / taup) * pmeq5[i][j][k] +weights[5]*reactionterm*Mp;// /rho0;
				pmn6[i][j][k]  =pm6[i][j][k]  * (1.0 - 1.0 / taup) + (1.0 / taup) * pmeq6[i][j][k] +weights[6]*reactionterm*Mp;// /rho0;
				pmn7[i][j][k]  =pm7[i][j][k]  * (1.0 - 1.0 / taup) + (1.0 / taup) * pmeq7[i][j][k] +weights[7]*reactionterm*Mp;// /rho0;
				pmn8[i][j][k]  =pm8[i][j][k]  * (1.0 - 1.0 / taup) + (1.0 / taup) * pmeq8[i][j][k] +weights[8]*reactionterm*Mp;// /rho0;
				pmn9[i][j][k]  =pm9[i][j][k]  * (1.0 - 1.0 / taup) + (1.0 / taup) * pmeq9[i][j][k] +weights[9]*reactionterm*Mp;// /rho0;
				pmn10[i][j][k] =pm10[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup)* pmeq10[i][j][k] +weights[10]*reactionterm*Mp;// /rho0;
				pmn11[i][j][k] =pm11[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup)* pmeq11[i][j][k] +weights[11]*reactionterm*Mp;// /rho0;
				pmn12[i][j][k] =pm12[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup)* pmeq12[i][j][k] +weights[12]*reactionterm*Mp;// /rho0;
				pmn13[i][j][k] =pm13[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup)* pmeq13[i][j][k] +weights[13]*reactionterm*Mp;// /rho0;
				pmn14[i][j][k] =pm14[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup)* pmeq14[i][j][k] +weights[14]*reactionterm*Mp;// /rho0;
				pmn15[i][j][k] =pm15[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup)* pmeq15[i][j][k] +weights[15]*reactionterm*Mp;// /rho0;
				pmn16[i][j][k] =pm16[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup)* pmeq16[i][j][k] +weights[16]*reactionterm*Mp;// /rho0;
				pmn17[i][j][k] =pm17[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup)* pmeq17[i][j][k] +weights[17]*reactionterm*Mp;// /rho0;
				pmn18[i][j][k] =pm18[i][j][k] * (1.0 - 1.0 / taup) + (1.0 / taup)* pmeq18[i][j][k] +weights[18]*reactionterm*Mp;// /rho0;
			}
		}
	}
}
/********************************************************************************

   sub routines to calculate pressure,power,mixing quality and write data files

**********************************************************************************/

void write_Pressure_Power(int itr)
{
	// Writing Pressure along the Center line in the mixing channel in horizantal and vertical section
	int i = 0, j = 0, k = 0;
	double Ptop;
	FILE *fp11;  // File Handler
	char buffer[25]; // Dynamic File name holder

	double sumx, sumy, sumz;
	
	// double sumconcx, sumconcy, sumconcz,avgconc; // AVG of A
	// double sumconcx_b, sumconcy_b, sumconcz_b,avgconc_b; // AVG of B
	// double sumconcx_c, sumconcy_c, sumconcz_c,avgconc_c; // AVG of C

	double sumtotalx, sumtotalz;
	double delux, delvy, delwz;
	double delvx, deluy;
	double deluz, delwx;
	double delvz, delwy;

	double vis = ((2.0*tau - 1.0) / 6.0)*(esp*esp*1.0);

	double temp, temptotal;

	sprintf(buffer, "Pressure_%d.dat", itr);
	fp11 = fopen(buffer, "w");

	fprintf(fp11, "ZONE T=\"PressureAlongMixingChannel\" \r\n");

	sumx = 0;
	for (i = 1; i <= mnx; i++)
	{
		sumz = 0;
		for (k = 1; k <= nz; k++)
		{
			sumz += rho[(i + ia - 1)][iny][k];
		}
		sumx += sumz;
	}
	Ptop = (sumx / (1.0*nz*mnx)) / 3.0;


	for (j = 1; j <= mny; j++)
	{
		sumx = 0;
		// sumconcx = 0;
		// sumconcx_b = 0;
		// sumconcx_c = 0;

		for (i = 1; i <= mnx; i++)
		{
			sumz = 0;
			// sumconcz = 0;
			// sumconcz_b = 0;
			// sumconcz_c = 0;
			for (k = 1; k <= nz; k++)
			{
				sumz += mrho[i][j][k];
				// sumconcz += gmrho[i][j][k];
				// sumconcz_b += hmrho[i][j][k];
				// sumconcz_c += pmrho[i][j][k];
			}
			sumx += sumz;
			// sumconcx += sumconcz;
			// sumconcx_b += sumconcz_b;
			// sumconcx_c += sumconcz_c;
		}
		// avgconc = (sumconcx/ (1.0*nz*mnx));
		// avgconc_b = (sumconcx_b/ (1.0*nz*mnx));
		// avgconc_c = (sumconcx_c/ (1.0*nz*mnx));
		temp = (sumx / (1.0*nz*mnx)) / 3.0;

		fprintf(fp11, "%d\t%g\t%g\t%g\n", j, temp, temp*P0, (temp - Ptop)*P0);
	}
	for (j = 1; j <= iny; j++)
	{
		sumx = 0;
		// sumconcx = 0;
		
		for (i = ia; i <= ib; i++)
		{
			sumz = 0;
			// sumconcz = 0;
			for (k = 1; k <= nz; k++)
			{
				sumz += rho[i][j][k];
				//sumconcz += grho[i][j][k];
			}
			sumx += sumz;
			//sumconcx += sumconcz;
		}
		//avgconc = (sumconcx/ (1.0*nz*mnx));
		temp = (sumx / (1.0*nz*mnx)) / 3.0;

		fprintf(fp11, "%d\t%g\t%g\t%g\n", j + mny, temp, temp*P0, (temp - Ptop)*P0);
	}
	fprintf(fp11, "ZONE T=\"HorizontalSectionPressure\" \r\n");
	for (i = 1; i <= inx; i++)
	{
		sumy = 0;
		// sumconcy = 0;
		for (j = 1; j <= iny; j++)
		{
			sumz = 0;
			// sumconcz = 0;
			for (k = 1; k <= nz; k++)
			{
				sumz += rho[i][j][k];
				// sumconcz += grho[i][j][k];
			}
			// sumconcy += sumconcz;
			sumy += sumz;
		}
		// avgconc = (sumconcy / (1.0*nz*iny));
		temp = (sumy / (1.0*nz*iny)) / 3.0;
		fprintf(fp11, "%d\t%g\t%g\t%g\n", i, temp, temp*P0,0.0);
	}

	fprintf(fp11, "ZONE T=\"SpecificPower\"\n");
	for (j = 2; j < mny; j = j + 1)
	{
		sumtotalz = 0;
		sumz = 0;
		for (k = 2; k < nz; k++)
		{
			sumtotalx = 0;
			sumx = 0;
			for (i = 2; i< mnx; i++)
			{
				delux = (mux[i + 1][j][k] - mux[i - 1][j][k]) / 2.0;
				delvy = (muy[i][j + 1][k] - muy[i][j - 1][k]) / 2.0;
				delwz = (muz[i][j][k + 1] - muz[i][j][k - 1]) / 2.0;

				deluy = (mux[i][j + 1][k] - mux[i][j - 1][k]) / 2.0;
				delvx = (muy[i + 1][j][k] - muy[i - 1][j][k]) / 2.0;

				deluz = (mux[i][j][k + 1] - mux[i][j][k - 1]) / 2.0;
				delwx = (muz[i + 1][j][k] - muz[i - 1][j][k]) / 2.0;

				delvz = (muy[i][j][k + 1] - muy[i][j][k - 1]) / 2.0;
				delwy = (muz[i][j + 1][k] - muz[i][j - 1][k]) / 2.0;

				sumx = pow(delux, 2.0) + pow(delwz, 2.0) + 0.5*pow((deluz + delwx), 2.0); // Crossection deformation rate

				sumtotalx = pow(delux, 2.0)
					+ pow(delvy, 2.0)
					+ pow(delwz, 2.0)
					+ 0.5*pow((delvx + deluy), 2.0)
					+ 0.5*pow((deluz + delwx), 2.0)
					+ 0.5*pow((delvz + delwy), 2.0); // total dissipation rate

			}
			sumz += sumx;
			sumtotalz += sumtotalx;
		}
		temp = 2.0*vis*(sumz / (1.0*(nz - 2)*(mnx - 2)));
		temptotal = 2.0*vis*(sumtotalz / (1.0*(nz - 2)*(mnx - 2)));
		fprintf(fp11, "%d\t%g\t%g\t%g\n", j, temp, temp*Pw0, temptotal*Pw0);
	}
	fclose(fp11);
}
void write_Domain_Data(int itr)
{
	int i, j, k;
	FILE *fp11;
	char buffer[25];

	sprintf(buffer, "Domain_Data_%d.dat", itr);
	fp11 = fopen(buffer, "w");
	fprintf(fp11, "VARIABLES = X\tY\tZ\tUx\tUy\tUz\trho\tgrho\thrho\tprho\r\n");
	fprintf(fp11, "ZONE T= \"Horizontal Channel\"  I=%d,J=%d,K=%d  \r\n", (inx), (iny), (nz));
	for (i = 1; i <= inx; i = i + 1)
	{
		for (j = 1; j <= iny; j = j + 1)
		{
			for (k = 1; k <= nz; k = k + 1)
			{
				fprintf(fp11, "%d\t%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", i, j + mny, k, ux[i][j][k], uy[i][j][k], uz[i][j][k], rho[i][j][k],grho[i][j][k],hrho[i][j][k],prho[i][j][k]);
			}
			fprintf(fp11, "\n");
		}
		fprintf(fp11, "\n");
	}
	fprintf(fp11, "ZONE T= \"Mixing Channel \" I=%d,J=%d,K=%d  \r\n", (inx), (iny), (nz));
	for (i = 1; i <= mnx; i = i + 1)
	{
		for (j = 1; j <= mny; j = j + 1)
		{
			for (k = 1; k <= nz; k = k + 1)
			{
				fprintf(fp11, "%d\t%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", i, j, k, mux[i][j][k], muy[i][j][k], muz[i][j][k], mrho[i][j][k],gmrho[i][j][k],hmrho[i][j][k],pmrho[i][j][k]);
			}
			fprintf(fp11, "\n");
		}
		fprintf(fp11, "\n");
	}
	fclose(fp11);
}
void write_Data_P_V_C(int itr)
{	
	int zc;
	int i,j,k;
	FILE *fp11;
	char buffer[25];

	sprintf(buffer, "Data_P_V_C%d.dat", itr);
	fp11 = fopen(buffer, "w");
	/**Header for tecplot data files**/
	fprintf(fp11, "VARIABLES = 'X','Y','Ux','Uy','rho','Ca','Cb','Cc'\r\n");
	fprintf(fp11, "ZONE T=\"Inlet Channel\" \r\n");

	/****************/
	zc = (nz + 1) / 2; 		// Input Z plane Here
	/****************/

	for (i = 1; i <= inx; i++)
	{
		for (j = mny + 1; j <= iny + mny; j++)
		{
				fprintf(fp11, "%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", i, j, ux[i][j - mny][zc], uy[i][j - mny][zc], rho[i][j - mny][zc], grho[i][j - mny][zc], hrho[i][j - mny][zc], prho[i][j - mny][zc]);
		}
		fprintf(fp11, "\n");
	}
	fprintf(fp11, "ZONE T=\"Mixing Channel\"\r\n");

	for (i = ia; i <= ib; i++)
	{
		for (j = 1; j <= mny + 1; j++)
		{
			if (j == mny + 1)
			{
				fprintf(fp11, "%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", i, j, ux[i][j - mny][zc], uy[i][j - mny][zc], rho[i][j - mny][zc],grho[i][j - mny][zc],hrho[i][j - mny][zc],prho[i][j - mny][zc]);
			}
			else
			{
				fprintf(fp11, "%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", i, j, mux[i - ia + 1][j][zc], muy[i - ia + 1][j][zc], mrho[i - ia + 1][j][zc], gmrho[i - ia + 1][j][zc], hmrho[i - ia + 1][j][zc], pmrho[i - ia + 1][j][zc]);
			}

		}
		fprintf(fp11, "\n");
	}

	fprintf(fp11, "ZONE T=\"InletCrossSectionY_Z(Z_%d,Y_%d)\"\r\n", (nz), (iny));
	for (j = 1; j <= iny; j++)
	{
		for (k = 1; k <= nz; k++)
		{
			fprintf(fp11, "%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", j, k, ux[1][j][k], uz[1][j][k], rho[1][j][k], grho[1][j][k], hrho[1][j][k], prho[1][j][k]);
		}
		fprintf(fp11, "\n");
	}
	
	zc = 2;//Outlet
	fprintf(fp11, "ZONE T=\"Outlet_Z_X%d(Z_%d,X_%d)\"\r\n", zc, (nz), (mnx));
	for (i = 1; i <= nz; i++)
	{
		for (j = 1; j <= mnx; j++)
		{
			fprintf(fp11, "%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", i, j, mux[j][zc][i], muz[j][zc][i], mrho[j][zc][i], gmrho[j][zc][i], hmrho[j][zc][i], pmrho[j][zc][i]);
		}
		fprintf(fp11, "\n");
	}
	zc = 600;//Mixing channel x-section
	fprintf(fp11, "ZONE T=\"MixingZ_X_%d(Z_%d,X_%d)\"\r\n",zc, (nz), (mnx));
	for( i = 1; i <= nz;i++)
	{
		for(j = 1; j <=mnx; j++)
		{
			fprintf(fp11, "%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", i,j,muz[j][zc][i],mux[j][zc][i],mrho[j][zc][i],gmrho[j][zc][i],hmrho[j][zc][i],pmrho[j][zc][i]);	
		}
		fprintf(fp11, "\n");
	}
	zc = mny-1;//Mixing channel x-section2
	fprintf(fp11, "ZONE T=\"MixingZ_X_%d(Z_%d,X_%d)\"\r\n",zc, (nz), (mnx));
	for( i = 1; i <= nz;i++)
	{
		for(j = 1; j <=mnx; j++)
		{
			fprintf(fp11, "%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", i,j,muz[j][zc][i],mux[j][zc][i],mrho[j][zc][i],gmrho[j][zc][i],hmrho[j][zc][i],pmrho[j][zc][i]);	
		}
		fprintf(fp11, "\n");
	}
	zc = 775-mny;//Mixing channel x-section1
	fprintf(fp11, "ZONE T=\"MixingZ_X_%d(Z_%d,X_%d)\"\r\n",zc+mny, (nz), (mnx));
	for( k = 1; k <= nz;k++)
	{
		for(i = 1; i <=inx; i++)
		{
			fprintf(fp11, "%d\t%d\t%lf\t%lf\t%lf\t%lf\t%lf\t%lf\n", k,i,uz[i][zc][k],ux[i][zc][k],rho[i][zc][k],grho[i][zc][k],hrho[i][zc][k],prho[i][zc][k]);	
		}
		fprintf(fp11, "\n");
	}
	fclose(fp11);
}
void write_MixingQuality(int itr)
{
	// Calculating mixing quality in the mixing channel
	int i = 0, j = 0, k = 0;
	
	//double Ptop;
	
	FILE *fp11;  // File Handler
	char buffer[25]; // Dynamic File name holder

	double amax,bmax,cmax;
	
	double sumx,  sumz,  sumy; // terms to calculate average in mean and variance A
	double sumx_b,sumz_b,sumy_b; // terms to calculate average in mean and variance B
	double sumx_c,sumz_c,sumy_c; // terms to calculate average in mean and variance C
	
	double avg,  variance;//,  cmax; // average and mean of concentration data
	double avg_b,variance_b;//,cmax_b; // average and mean of concentration data
	double avg_c,variance_c;//,cmax_c; // average and mean of concentration data
	
	double maxvariance,maxvariance_b,maxvariance_c;
	/**Calculation of Maximum variance at the entrace**/
	for (j = 1; j <= iny; j++)
	{
		sumz   = 0;
		sumz_b = 0;
		sumz_c = 0;
		for (k = 1; k <= nz; k++)
		{
			sumz   += grho[1][j][k];
			sumz_b += hrho[inx][j][k];
			sumz_c += prho[1][j][k];
		}
		sumy   += sumz;
		sumy_b += sumz_b;
		sumy_c += sumz_c;
	}
	
	avg   = (sumy   / (1.0*nz*iny));
	avg_b = (sumy_b / (1.0*nz*iny));
	avg_c = (sumy_c / (1.0*nz*iny));

	for (j = 1; j <= iny; j++)
	{
		sumz   = 0;
		sumz_b = 0;
		sumz_c = 0;

		for (k = 1; k <= nz; k++)
		{
			sumz += pow((grho[1][j][k]-avg),2.0);
			sumz_b += pow((hrho[inx][j][k]-avg_b),2.0);
			sumz_c += pow((prho[1][j][k]-avg_c),2.0);
		}
		sumy += sumz;
		sumy_b += sumz_b;
		sumy_c += sumz_c;
	}
	
	maxvariance   = (1.0/(1.0*nz*iny))*sumy; 
	maxvariance_b = (1.0/(1.0*nz*iny))*sumy_b; 
	maxvariance_c = (1.0/(1.0*nz*iny))*sumy_c; 
    /**Completed calculation of Maximum variance at the entrace**/

    //printf("%lf,%lf,%lf\n",maxvariance,maxvariance_b,maxvariance_c );

	sprintf(buffer, "MixingQuality_%d.dat", itr);
	fp11 = fopen(buffer, "w");
	fprintf(fp11, "ZONE T=\"Mixing Quality\" \r\n");
	fprintf(fp11, "VARIABLES= \"Y\"\"A\"\"B\"\"C\"\"IA\"\"IB\"\"IC\"\r\n");

	sumx   = 0;
	sumx_b = 0;
	sumx_c = 0;

	cmax = 0;
	
	for (j = 1; j <= mny; j++)
	{
		sumx   = 0;
		sumx_b = 0;
		sumx_c = 0;

		for (i = 1; i <= mnx; i++)
		{
			sumz   = 0;
			sumz_b = 0;
			sumz_c = 0;

			for (k = 1; k <= nz; k++)
			{
				sumz   += gmrho[i][j][k];
				sumz_b += hmrho[i][j][k];
				sumz_c += pmrho[i][j][k];

				if(amax < gmrho[i][j][k])
				{
					amax = gmrho[i][j][k];
				}
				if(bmax < hmrho[i][j][k])
				{
					bmax = hmrho[i][j][k];
				}
				if(cmax < pmrho[i][j][k])
				{
					cmax = pmrho[i][j][k];
				}


			}
			sumx   += sumz;
			sumx_b += sumz_b;
			sumx_c += sumz_c;
		}
		avg = (sumx / (1.0*nz*mnx));
		avg_b = (sumx_b / (1.0*nz*mnx));
		avg_c = (sumx_c / (1.0*nz*mnx)); // cbar from the bothe.et.al eq.20

		sumx = 0;
		sumx_b = 0;
		sumx_c = 0;
		for (i = 1; i <= mnx; i++)
		{
			sumz = 0;
			sumz_c = 0;
			sumz_b = 0;
			for (k = 1; k <= nz; k++)
			{
				sumz   += pow((gmrho[i][j][k]-avg),2.0);
				sumz_b += pow((hmrho[i][j][k]-avg_b),2.0);
				sumz_c += pow((pmrho[i][j][k]-avg_c),2.0);
			}
			sumx   += sumz;
			sumx_b += sumz_b;
			sumx_c += sumz_c;
		}
		variance = (1.0/(1.0*nz*mnx))*sumx; // sigma2 from the bothe.et.al eq 20
		variance_b = (1.0/(1.0*nz*mnx))*sumx_b; // sigma2 from the bothe.et.al eq 20
		variance_c = (1.0/(1.0*nz*mnx))*sumx_c; // sigma2 from the bothe.et.al eq 20

		if(avg > 0 && avg_b > 0 && avg_c > 0)
		{
			// //double Is = variance/(avg*(Ca_in1-avg));
			// double Is = variance/maxvariance;
			// double Is_b = variance_b/maxvariance_b;
			// //double Is_c = variance_c/maxvariance;
			// double Is_c = variance_c/(avg_c*(cmax - avg_c));

			double Is = variance/(avg*(amax- avg));
			double Is_b = variance_b/(avg_b*(bmax- avg_b));
			double Is_c = variance_c/(avg_c*(cmax- avg_c));

			double Im = 1-sqrt(Is);
			double Im_b = 1-sqrt(Is_b);
			double Im_c = 1-sqrt(Is_c);

			fprintf(fp11, "%d\t%g\t%g\t%g\t%g\t%g\t%g\n",j,avg,avg_b,avg_c,Im,Im_b,Im_c);	
		}
		else
		{
			fprintf(fp11, "%d\t%g\t%g\t%g\t%g\t%g\t%g\n",j,avg,avg_b,avg_c,0.0,0.0,0.0);	
		}
		
	}
	//cmax = 0;
	for (j = 1; j <= iny; j++)
	{
		sumx = 0;
		sumx_b = 0;
		sumx_c = 0;
		for (i = ia; i <= ib; i++)
		{
			sumz = 0;
			sumz_b = 0;
			sumz_c = 0;
			for (k = 1; k <= nz; k++)
			{
				sumz += grho[i][j][k];
				sumz_b += hrho[i][j][k];
				sumz_c += prho[i][j][k];

				if(amax < grho[i][j][k])
				{
					amax = grho[i][j][k];
				}
				if(bmax < hrho[i][j][k])
				{
					bmax = hrho[i][j][k];
				}
				if(cmax < prho[i][j][k])
				{
					cmax = prho[i][j][k];
				}
			}
			sumx   += sumz;
			sumx_b += sumz_b;
			sumx_c += sumz_c;

		}
		avg   = (sumx   / (1.0*nz*mnx)); // cbar from the bothe.et.al eq.20
		avg_b = (sumx_b / (1.0*nz*mnx)); // cbar from the bothe.et.al eq.20
		avg_c = (sumx_c / (1.0*nz*mnx)); // cbar from the bothe.et.al eq.20
		
		sumx   = 0;
		sumx_b = 0;
		sumx_c = 0;

		for (i = ia; i <= ib; i++)
		{
			sumz   = 0;
			sumz_b = 0;
			sumz_c = 0;

			for (k = 1; k <= nz; k++)
			{
				sumz   += pow((grho[i][j][k]-avg),2.0);
				sumz_b += pow((hrho[i][j][k]-avg_b),2.0);
				sumz_c += pow((prho[i][j][k]-avg_c),2.0);
			}
			sumx   += sumz;
			sumx_b += sumz_b;
			sumx_c += sumz_c;
		}
		variance = (1.0/(1.0*nz*mnx))*sumx; // sigma2 from the bothe.et.al eq 20
		variance_b = (1.0/(1.0*nz*mnx))*sumx_b; // sigma2 from the bothe.et.al eq 20
		variance_c = (1.0/(1.0*nz*mnx))*sumx_c; // sigma2 from the bothe.et.al eq 20
		if(avg > 0)
		{
			//double Is = variance/(avg*(Ca_in1-avg));
			//double Is   = variance/maxvariance;
			//double Is_b = variance_b/maxvariance_b;
			//double Is_c = variance_c/maxvariance_c;
			double Is = variance/(avg*(amax- avg));
			double Is_b = variance_b/(avg_b*(bmax- avg_b));
			double Is_c = variance_c/(avg_c*(cmax- avg_c));

			double Im = 1.0-sqrt(Is);
			double Im_b = 1.0-sqrt(Is_b);
			double Im_c = 1.0-sqrt(Is_c);

			fprintf(fp11, "%d\t%g\t%g\t%g\t%g\t%g\t%g\n",j+mny,avg,avg_b,avg_c,Im,Im_b,Im_c);	
		}
		else
		{
			fprintf(fp11, "%d\t%g\t%g\t%g\t%g\t%g\t%g\n",j+mny,avg,avg_b,avg_c,0.0,0.0,0.0);	
		}
	}
	fclose(fp11);
}


/*-------------------------------------------------------------------MAIN FUNCTION----------------------------------------------------------------------------------------------------*/
int main()
{
	int itr; // To keep track of the no of iterations
	double dneu, delx, dely, delz, delt, Rei = 0, Reo = 0;

	/*Memoy allocation for the distribution functions*/
	allocateMemory();

	printf("nx:%d,ny:%d,nz:%d,ia:%d,ib:%d\n\n", inx, iny + mny, nz, ia, ib);
	printf("Umean:%lf,tau:%lf,taua:%lf,cw:%d\n\n", Umean, tau,taua, mnx - 1);
	printf("taub:%lf\ntaup:%lf",taub,taup);
	printf("rho0:%lf,rho_out:%lf\n\n", rho0, rho_out);

	GenerateInletVelocity();
	print_ConversionFactors_LBM_Real();	

	
	//--------------Calculation of delx,dely,delt,dneu,---------------------------------------------------	
	dely = width / (mny + iny);
	delx = dely;
	delz = dely;
	delt = delx / esp;

	//Kinematic Viscosity in lattice units
	dneu = ((2.0*tau - 1.0) / 6.0)*(esp*esp*delt);     

	printf("The value of delx:%lf\tdely:%lf\tdelt:%lf\tdelz:%lf\tdneu:%lf\n", delx, dely, delt, delz, dneu);

	printf("the value of rate const is %lf\n",rateconst );
	
	//Calculation of Hydraulic radius of the inlet and outlet & Reynolds Number

	double cw = (mnx - 1);
	double Ai = (iny - 1)*(nz - 1);
	double Pi = 2.0*((iny - 1) + (nz - 1));
	double Dhi = (4.0*Ai) / Pi;

	double Ao = (cw)*(nz - 1);
	double Po = 2.0*((nz - 1) + cw);
	double Dho = (4.0*Ao) / Po;

	Rei = (Umean*Dhi) / dneu;
	Reo = (Umean*Dho) / dneu;

	printf("Hydraulic Radius of the inlet is %lf;outlet is %lf\n", Dhi, Dho);
	printf("\nReynolds Number is Rei,Reo:%lf,%lf\n\n", Rei, Reo);
	printf("\nSchmidt Number for the species is NSc:%lf\n\n", (2.0*tau-1.0)/(2.0*taua-1.0));

	writeParameters(Reo);	
	initialization(); // It initialises all macroscopic variables
	equilibrium(); // It calculates equilibrium distribution function
	initialize_distribution_function(); //It initializes all distribution function
	
	//Time loop starts from here..
	for (itr = 0; itr <= time; itr++)
	{
		equilibrium();// It calculates equilibrium distribution function at each point
		collision();  // f'(r+dr,t+dt)=f(r,t)-1/tau(f(r,t)-feq(r,t)
					 // here f'(r+dr,d+dt) is distribution function just after collision
		streaming(); //f(r+dr,d+dt)=f'(r+dr,d+dt) inside the channel
		boundary_conditions(); // f(r+dr,d+dt)=f'(r+dr,d+dt) at boundary
		calculate_velocity_density(); // It calculates density and velocity at (r+dr,t+dt)

		if (itr%time_step == 0)
		{
			printf("iteration is going for Time=%d to Time=%d\n", itr, (itr + time_step));
			write_Data_P_V_C(itr);
			write_Pressure_Power(itr);
			write_MixingQuality(itr);
			if(itr%time_domaindata == 0)
			{
				write_Domain_Data(itr);
			}
		}
		if(itr == 100){write_Data_P_V_C(itr);}
	}
	deallocateMemory();
}
