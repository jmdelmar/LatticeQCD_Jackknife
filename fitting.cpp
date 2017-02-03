/* Library of functions for fitting multiple sets of data to a constant parameter, averaging the fits over all the sets,
   and calculating the error of the fit
*/

#include "fitting.h"

using namespace std;


// Fits multiple sets of data in each column of the given matrix to a constant parameter, weighted by the given error
// at each timeslice, from startDat to endDat. Stores fits in the given vector.

void fitData( vector<double> *fit, vector< vector<double> > *val, vector<double> *err, int startDat, int endDat ) {
	
  int timeNum = val  ->  size();
  int binNum = val  ->  at(0).size();
	
  double sumErrSquare; // Normalizing factor
	
  for ( int t = startDat; t <= endDat; t++ ) {
		
      sumErrSquare += pow( err -> at(t), -2 );

  }
	
  for ( int b = 0; b < binNum; b++ ) {
		
    double weightedSum = 0;
		
    for ( int t = startDat; t <= endDat; t++ ) {
			
	weightedSum += val -> at(t).at(b) / pow( err -> at(t), 2 );
				
    }
		
    fit -> at(b) = weightedSum / sumErrSquare;
		
  }
	
  return;
}


// Averages a set of fits over all bins and calculates the error of the fit

void averageFit( double *avg, double *err, vector<double> *vals ) {

  int binNum = vals -> size();

  // Calculate average fit

  double sum = 0;
	
  for ( int b = 0; b < binNum; b++ ) { // Loop over bins
		
    sum += vals -> at(b);
			
  }

  *avg = sum / binNum;

  // Calculate fit error
  
  double sumDiffSquare = 0;
	
  for ( int b = 0; b < binNum; b++ ) { // Loop over bins
		
    sumDiffSquare += pow( *avg - vals -> at(b), 2 );
		
  }
	
  *err = sqrt( sumDiffSquare * (binNum - 1) / binNum );
	
  return;
}
