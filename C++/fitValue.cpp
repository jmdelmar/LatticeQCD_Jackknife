/* Fits given values over given time interval, averages over all bins, and 
   calculates its error

Input:
1. file of values to be read. File should contain values for each bin.
2. file of errors to be read
3. first timestep to be fit
4. last timestep to be fit
5. name of value being fitted for naming output files. Output files will have 
   name in the format: out/<ARGUMENT 5>fit_bins.dat and out/avg<ARGUMENT 5>fit.dat
6. (Optional) 1 if a printout is wanted
*/

#include "jk.h"
#include "fitting.h"
#include <cstdlib> //for atoi()

using namespace std;

int main ( int argc, char *argv[] ) {


  //////////////////////
  // Read user inputs //
  //////////////////////

  
  // Check that all user inputs have been included

  if ( argc < 6 ) {

    cout << "ERROR: not enough argurments" << endl;
    cout << "Include:" << endl;
    cout << "1. file of values for each bin and timeslice" << endl;
    cout << "2. file of for each timeslice" << endl;
    cout << "3. first timestep to be fit" << endl;
    cout << "4. last timestep to be fit" << endl;
    cout << "5. name of value being fitted for naming output files" << endl;
    cout << "6. (Optional) 1 if a printout is wanted" << endl;
    
    return 1;
  }

  int timestepNum = detTimestepNum( argv[1], 2 );

  // detTimestepNum is a function in "jk.h" which reads a data file with the
  // timesteps for a number of repeating configurations in the first column and
  // determines the number of timesteps
  
  int binNum = detConfigNum( argv[1], 2 );

  // detConfigNum is a function in "jk.h" which reads a data file with the
  // timesteps for a given number of configurations in the first column and
  // determines the number of configurations

  int startTime = atoi( argv[3] );
  int endTime = atoi( argv[4] );
	
  // Set printout argument if it has been given ( Print = 1 for printout )
  
  int Print;
  
  if ( argc == 7 ) {

    Print = atoi( argv[6] );

  }


  // Print number of timesteps and bins

  if ( Print == 1 ) {

    cout << "number of timesteps: " << timestepNum << endl;

    cout << "number of bins: " << binNum << endl;

  }
  
  ///////////////////////////////////////////////
  // Initialize matrices, vectors, and doubles //
  ///////////////////////////////////////////////


  // Matrix of values
  // ( vals[t][b] )

  vector< vector<double> > vals ( timestepNum ); //matrix w/ 'timestepNum' rows

  giveMatrixCols( &vals, binNum ); //give each row 'binNum' columns

  // giveMatrixCols is a function in "jk.h" which gives a matrix a given number
  // of columns in each of its rows

  
  // Vector of errors
  // (vals_err[t] )

  vector<double> vals_err ( timestepNum );

  
  // Vector of fitted values
  // ( fit[b] )
  
  vector<double> fit ( binNum );

  // Average fitted value and its error
  
  double fit_avg;
  double fit_err;


  ////////////////////////////////////////////
  // Read data files and store their values //
  ////////////////////////////////////////////


  // Matrix of values

  readNthDataCol( &vals, argv[1], 2, 2 );

  // readNthDataCol is a functions in "jk.h"
  
  // Print marix of values

  if ( Print == 1 ) {

    printMatrix( &vals, "Values:" );

    // printMatrix is a function in "jk.h"

  }
  
  // Vector of errors

  {

    // vals_errMat needs to be a matrix wih one column so that it can be given
    // to readNthDataCol(). This is done in its own scope so that vals_errMat
    // does not have to be held in memory for the entire program

    vector< vector<double> > vals_errMat ( timestepNum );

    giveMatrixCols( &vals_errMat, 1 );
  
    readNthDataCol( &vals_errMat, argv[2], 3, 3 );

    // Set vals_err and vals_errMat values equal

    for ( int t = 0 ; t < timestepNum; t++ ) {
    
      vals_err[t] = vals_errMat[t][0];

    }

  }
  
  // Print errors

  if ( Print == 1 ) {

    printVector( &vals_err, "Errors:" );

    // printVector is a function in "jk.h"

  }


  ///////////////////////////////
  // Calculate fit from values //
  ///////////////////////////////

  
  fitData( &fit, &vals, &vals_err, startTime, endTime );

  // Print fits

  if ( Print == 1) {

    printVector( &fit, "Fitted values:" );

  }
  
  // Calculate average fit and its error

  averageFit( &fit_avg, &fit_err, &fit );

  // Print average fit and its error

  if ( Print == 1 ) {

    cout << "Average fit:" << endl;

    cout << fit_avg << endl;

    cout << "Fit error:" << endl;

    cout << fit_err << endl;

  }

  
  ////////////////////////
  // Write output files //
  ////////////////////////
  

  // Fitted value for each bin to be used for other calculations

  // Declare a null vector of same size as fit[] so that another vector can be
  // passed to writeVectorFile() without anything of value being written

  vector<double> nullVec ( binNum );

  char valFilename[256];

  sprintf( valFilename, "out/%s_bins.dat", argv[5] );
  
  writeVectorFile( valFilename, &fit, &nullVec );
  
  
  // Average fitted mass and its error

  char avgValFilename[256];

  sprintf( avgValFilename, "out/avg%s.dat", argv[5] );

  writeFitFile( avgValFilename, fit_avg, fit_err, startTime, endTime, timestepNum - 1 );

  // writeFtFile is a function in "jk.h"
  
  
  return 0;
}
