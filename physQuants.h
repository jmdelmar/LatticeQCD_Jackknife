

#ifndef PHYS_QUANTS_H
#define PHYS_QUANTS_H

#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

// Calculates the effective masses from a given 'binNum' by 'timestepNum' matrix
// of resampled two-point functions and puts them in a matrix of the same size.

void effMass( vector< vector<double> > *, vector< vector<double> > * );


// Calculates the ratio from two- and three-point functions for each
// timeslice, momentum transfer, and bin.

void calcRatio( vector< vector< vector<double> > > *,
	    vector< vector< vector<double> > > *,
	    vector< vector< vector<double> > > * );


// Calculates the scalar charge from the ratio, the fitted mass, and q squared

void calcScalarFF( vector< vector< vector<double> > > *,
		   vector< vector< vector<double> > > *,
		   vector<double> *, vector<int> *, int );
  

#endif
