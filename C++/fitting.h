/* Library of functions for fitting multiple sets of data to a constant parameter, averaging the fits over all the sets,
   and calculating the error of the fit
*/

#ifndef FITTING_H
#define FITTING_H

#include <iostream>
#include <vector>
#include <cmath>

using namespace std;


// Fits multiple sets of data in each column of the given matrix to a constant parameter, weighted by the given error at
// each timeslice, from startDat to endDat. Stores fits in the given vector.

void fitData( vector<double> *, vector< vector<double> > *, vector<double> *, int, int );


// Averages a set of fits over all bins and calculates the error of the fit

void averageFit( double *, double *, vector<double> * );

  
#endif
