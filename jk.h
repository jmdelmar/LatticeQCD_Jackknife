/* Functions to be used for calculating JK averages and effective masses */

#ifndef JK_H
#define JK_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace std;

// Reads a given data file with the timesteps for a number of repeating
// configurations in the first of given colNum columns and determines the number
// of timesteps

int detTimestepNum( char *, int );


// Reads a given data file with the timesteps for a number of configurations
// in the first of given colNum columns and determines the number of
// configurations

int detConfigNum( char *, int );


// Reads a given file and puts the colNth column out of colTot columns into the
// given matrix

void readNthDataCol( vector< vector<double> > *, char *, int, int );

// Reads a given file with data for different momentum transfers and reads the
// colNth column out of colTot columns into the given matrix for each momentum

void readNthMomDataCol( vector< vector< vector< vector<double> > > > *, char *,
			vector<int> *, int, int );


// Reads a given file with colTot columns and timestrepNum timesteps per
// configuration and fills the given vector with the colNth double in the rowNth
// row for each configuration

void readNthDataRow( vector<double> *, char *, int, int, int, int );


// Calculates the resampled averages of values in rows of a given 'configNum' by
// 'timestepNum' matrix and puts them in a given 'binNum' by 'timestepNum' matrix

void jackknifeMatrix( vector< vector<double> > *, vector< vector<double> > * );


// Calculates the resampled averages of values in a given 'configNum'
// dimensional vector and puts them in a given 'binNum' dimensional vector

void jackknifeVector( vector<double> *, vector<double> * );


// Calculates the averages and standard deviations of the rows of a given matrix
// and puts them into given vectors whose demensions are the number of rows of
// the matrix

void averageRows( vector<double> *, vector<double> *, vector< vector<double> > * );


// Writes a file containing a matrix so that each row is a different timestep

void writeMatrixFile( char *, vector< vector<double> > * );

// Writes two vectors to an output file. The first column of the file is each
// timestep, the second is the vector given as the second argument, and the third
// is the vector given as the third argument

void writeVectorFile( char *, vector<double> *, vector<double> * );


// Writes a fit, its error, and the first and last time-slices in the fit. The
// 1st column is the 1st and last time-slices, given as 'firstT' and 'lastT',
// the 2nd column is the fit, repeated twice, and the last column is the fit
// error, repeated twice. This format is used so that the fit can easily be
// plotted as a line.

void writeFitFile( char *, double, double, int, int );


// Prints a tensor to standard out and formats it so that the first dimension is
// printed one after another as elements seperated by a column in each row

void printTensor( vector< vector< vector<double> > > *, vector<int> *, string );


// Prints a matrix to standard out and formats it so that each value is seperated
// by a comma and each row is on a different line

void printMatrix( vector< vector<double> > *, string );


// Prints a vector to standard out, formatted so that each component is on a
// seperate line

void printVector( vector<double> *, string );

void printVector( vector<int> *, string );

// Gives a matrix a given number of columns in each of its rows

void giveMatrixCols( vector< vector<double> > *, int );


// Gives a 3rd order tensor a matrix with given dimensions in each element of
// its 0th dimension

void giveTensorMatrix( vector< vector< vector<double> > > *,
		       int, int );


// Gives a 4th order tensor a 3rd order tensor with given dimensions in each
// element of it 0th dimension

void giveTensorTensor( vector< vector< vector< vector<double> > > > *,
		       int, int, vector<int> * );
  

// Fills a matrix with test values, starting at zero and going up by one for
// each entry

void fillTestMatrix( vector< vector<double> > * );


#endif
