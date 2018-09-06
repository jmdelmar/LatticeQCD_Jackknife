/* Functions to be used for calculating JK averages and effective masses */

#ifndef JK_H
#define JK_H

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <string.h>
#include <string>
#include <sstream>

using namespace std;

// Reads a given data file with the timesteps for a number of repeating
// configurations in the first of given colNum columns and determines the number
// of timesteps

int detTimestepNum( char *, int );


//Reads files by each configuration and returns the number of timesteps in the 
//files. If the timesteps in the 1st column of a file does not start at 0 and 
//increase by 1 for each new line or the number of timesteps in a file differ 
//from the first configuration read, the configuration of the file at fault 
//is thrown as an exception.

int detTimestepNum_rbc( vector<string> *, int );


// Reads a given data file with the timesteps for a number of configurations
// in the first of given colNum columns and determines the number of
// configurations

int detConfigNum( char *, int );


// Reads a given file and puts the colNth column out of colTot columns into the
// given matrix

void readNthDataCol( vector< vector<double> > *vals, char *, int, int );


// Reads a given file by configurations in seperate directories and puts the 
// colNth column out of colTot columns into the given matrix

void readNthDataCol_rbc( vector< vector<double> > *, vector<string> *, int, 
			 int );


// Reads a given file of strings and puts each string into the given vector

void readStringFile( vector<string> *, char * );
  

// Reads a given file with data for different momentum transfers and reads the
// colNth column out of colTot columns into the given matrix for each momentum

void readNthMomDataCol( vector< vector< vector< vector<double> > > > *, char *,
			vector<int> *, int, int );


// Reads a given file with colTot columns and timestrepNum timesteps per
// configuration and fills the given vector with the colNth double in the rowNth
// row for each configuration

void readNthDataRow( vector<double> *, char *, int, int, int, int );


// Reads a given file by configurations in different directories with colTot 
// columns and timestepNum timesteps per configuration and fills the given 
// vector with the colNth double in the rowNth row for each configuration

void readNthDataRow_rbc( vector<double> *, vector<string> *, int, int, int );


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


// Prints a vector of strings to standard out, formatted so that each component 
// is on a seperate line

void printVector( vector<string> *, string );

// Writes a fit, its error, and the first and last time-slices in the fit. The
// 1st column is the 1st and last time-slices, given as 'firstT' and 'lastT',
// the 2nd column is the fit, repeated twice, and the last column is the fit
// error, repeated twice. This format is used so that the fit can easily be
// plotted as a line.

void writeFitFile( char *, double, double, int, int, int );


// Prints a tensor to standard out and formats it so that the first dimension
// is printed one after another as elements seperated by a column in each row

void printTensor( vector< vector< vector<double> > > *vals, string title );


// Prints a tensor to standard out and formats it so that the first dimension,
// Q^2, is printed one after another as elements seperated by a column in each row

void printQsqTensor( vector< vector< vector<double> > > *vals, 
		     vector<int> *q2, string title );


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
  

// Splits a string into tokens seperated by a deliminator

void split( vector<string> *, char *, char * );


// Writes the name of a file contained in a sub-directory of the given home 
// directory. File should contain the name of its sub-directory in its name 
// only once, represented in the given filename template by a '*'.

// [This could be modified to include more than one *, I think using sting
// streams]

void setFilename( vector<string> *, char *, vector<string> *, char * );


// Fills a matrix with test values, starting at zero and going up by one for
// each entry

void fillTestMatrix( vector< vector<double> > * );


#endif
