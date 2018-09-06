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

  int q2Tot = 55;
  
  // Check that all user inputs have been included

  if ( argc < 6 ) {

    cout << "ERROR: not enough arguments" << endl;
    cout << "Include:" << endl;
    cout << "1. prefix for file of values for each bin and timeslice" << endl;
    cout << "2. prefix for file of errors for each timeslice" << endl;
    cout << "3. file name of Q squared vales" << endl;
    cout << "4. first timestep to be fit" << endl;
    cout << "5. last timestep to be fit" << endl;
    cout << "6. (Optional) 1 if a printout is wanted" << endl;
    
    return 1;
  }

  // Initialize and fill momentum vector
  // ( q2[q] )
  
  vector<int> q2 ( q2Tot );
  
  { // Declare a matrix form of q2 in its own scope so that it can be sent to
    // readNthDataCol()

    vector< vector<double> > q2_mat ( q2Tot );

    giveMatrixCols( &q2_mat, 2 );

    // giveMatrixCols() is a function in "jk.h"

    readNthDataCol( &q2_mat, argv[3], 1, 4 );

    // readNthDataCol() is a function in "jk.h"

    
    // Set elements of q2[] equal to elements of q2_mat[][]

    for ( int q = 0; q < q2Tot; q++ ) {

      q2[q] = q2_mat[q][0];

    }

  }

  // Set Print argument if it has been given (Print=1 for print out)

  int Print;
  
  if ( argc == 7 ) {

    Print = atoi( argv[6] );

  }


  // Initialize vector for average fits and their errors for each Q squared

  vector<double> fit_avg ( q2Tot );

  vector<double> fit_err ( q2Tot );  

  
  // Calculate fitting for each Q squared

  for ( int q = 0; q < 47; q++ ) {

    // Print current Q^2 value

    if ( Print == 1 ) {

      cout << "Q^2=" << q2[q] << ":" << endl;

    }

    
    //////////////////////
    // Read user inputs //
    //////////////////////


    char inValFile[256];
    
    sprintf( inValFile, "%s%d.dat", argv[1], q2[q] );
    
    char inErrFile[256];

    sprintf( inErrFile, "%s%d.dat", argv[2], q2[q] );


    // Determine number of timesteps and bins
    
    int timestepNum = detTimestepNum( inValFile, 2 );

    // detTimestepNum is a function in "jk.h" which reads a data file with the
    // timesteps for a number of repeating configurations in the first column and
    // determines the number of timesteps
  
    int binNum = detConfigNum( inValFile, 2 );

    // detConfigNum is a function in "jk.h" which reads a data file with the
    // timesteps for a given number of configurations in the first column and
    // determines the number of configurations


    // Print number of timesteps and bins

    if ( Print == 1 ) {

      cout << "number of timesteps: " << timestepNum << endl;

      cout << "number of bins: " << binNum << endl;

    }


    // Set the starting and ending time slices to be included in fit from input
    
    int startTime = atoi( argv[3] );
    int endTime = atoi( argv[4] );

    
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
    // ( vals_err[t] )

    vector<double> vals_err ( timestepNum );

  
    // Vector of fitted values
    // ( fit[b] )
  
    vector<double> fit ( binNum );

  
    ////////////////////////////////////////////
    // Read data files and store their values //
    ////////////////////////////////////////////


    // Matrix of values

    readNthDataCol( &vals, inValFile, 2, 2 );

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
  
      readNthDataCol( &vals_errMat, inErrFile, 3, 3 );

      // giveMatrixCols() and readNtheDataCol() are functions in "jk.h"

      
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

    averageFit( &fit_avg[q], &fit_err[q], &fit );

    // averageFit() is a function in "fitting.h"

    
    // Print average fit and its error

    if ( Print == 1 ) {

      cout << "Average fit:" << endl;

      cout << fit_avg[q] << endl;

      cout << "Fit error:" << endl;

      cout << fit_err[q] << endl;

    }

  }

  
    ///////////////////////
    // Write output file //
    ///////////////////////
  

    char outFile[256];

    sprintf( outFile, "%sfit.dat", argv[1] );

    writeVectorFile( outFile, &fit_avg, &fit_err );

    // writeVectorFile is a function in "jk.h"
  
  
  return 0;
}
