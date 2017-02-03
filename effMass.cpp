/* Extracts the 5th column from a given data file and organizes it by timesteps 
   in rows and configurations in columns, calculates the JK averages for each 
   timestep and bin, the effective mass for each timestep and bin, the average 
   and error of the effective mass for each timestep, and writes a file 
   containing the average effective masses with their respective errors for each 
   timestep.

Input: 
1.	data file of two-point functions to be read
2.	bin size (make sure numbers of configurations are divided by bin size
 	exactly)
3.      (Optional) 1 if printouts of the JK averaged two-point functions and 
        effective masses are wanted
*/

#include "jk.h"
#include "physQuants.h"
#include <cstdlib> //for atoi()

using namespace std;

int main ( int argc, char *argv[] ) {

    
  //////////////////////
  // Read user inputs //
  //////////////////////

  
  if ( argc < 3 ) {

    cout << "Missing input arguments" << endl;
    cout << "Include:" << endl;
    cout << "1. two-point function data file" << endl;
    cout << "2. bin size" << endl;
    cout << "3. (Optional) 1 for printout of data" << endl;

    return 1;
  }

  int timestepNum = detTimestepNum( argv[1], 12 );
  int configNum = detConfigNum( argv[1], 12 );

  // detTimestepNum is a function in "jk.h" which reads a file of repeating
  // configurations with the timesteps in the first column out of the given
  // number of columns and determines the number of timesteps in the file before
  // a new configuration starts and the timesteps repeat

  int binSize = atoi( argv[2] );

  // Check that configNum is exactly divided by binSize

  while( configNum % binSize != 0 ) {
    
    cout << "WARNING: configNum " << configNum << " is divided by binSize ";
    cout << binSize << " with a remainder. ";
    cout << "Please enter a new binSize." << endl;

    cin >> binSize;

  }

  int binNum = configNum / binSize;

  // If a print argument has been given, set Print (Print = 1 to print)
  
  int Print;
  
  if ( argc == 4 ) {

    Print = atoi( argv[3] );

  }

    
  /////////////////////////////////////
  // Initialize matrices and vectors //
  /////////////////////////////////////

	
  // Matrix of two point functions
  // ( twoPtFuncs[t][c] )
  
  vector< vector<double> > twoPtFuncs( timestepNum ); //matrix w/ 'timestepNum' rows

  giveMatrixCols( &twoPtFuncs, configNum ); // give matrix 'configNum' columns

  // giveMatrixCols is a function in "jk.h" which fills each row of a
  // matrix with a given number of columns

  
  // Matrix of JK averaged TPFs
  // ( twoPtFuncs_jk[t][b] )
  
  vector< vector<double> > twoPtFuncs_jk( timestepNum );  //matrix w/ 'timestepNum' rows

  giveMatrixCols( &twoPtFuncs_jk, binNum ); // give matrix 'binNum' columns

  // Matrix of effective masses
  // ( eMass[t][b] )

  vector< vector<double> > eMass( timestepNum );  //matrix w/ 'timestepNum' rows

  giveMatrixCols( &eMass, binNum ); //give each row 'binNum' columns
  
  // Vector of average two-point functions
  // ( twoPtFuncs_avg[t] )
  
  vector<double> twoPtFuncs_avg( timestepNum );

  // Vector of two-point function errors
  // ( twoPtFuncs_err[t] )
  
  vector<double> twoPtFuncs_err( timestepNum );

  // Vector of average effective masses
  // ( eMass_avg[t] )
  
  vector<double> eMass_avg( timestepNum );

  // Vector of effective mass errors
  // ( eMass_err[t] )
  
  vector<double> eMass_err( timestepNum );


  ///////////////////////////////////////////////
  // Fill matrices with values from data files //
  ///////////////////////////////////////////////


  readNthDataCol( &twoPtFuncs, argv[1], 5 , 12 );

  // readNthDataCol is a function in "jk.h" which reads a file and stores
  // the numbers in the first given numbered column and stores them in a
  // matrix


  //Print matrix of two-point functions

  if( Print == 1 ) {

    printMatrix( &twoPtFuncs, "Two-point functions:" );

    // printMatrix is a function in "jk.h"

  }


  ///////////////
  // Jackknife //
  ///////////////
  
	
  jackknifeMatrix( &twoPtFuncs_jk, &twoPtFuncs );

  // jackkifeMatrix is a function in "jk.h" which calculates the resampled
  // averages of values orginized in a matrix by different timesteps in each row
  // and different configurations in each column

  
  // Print matrix of resampled two-point functions

  if( Print == 1 ) {

    printMatrix( &twoPtFuncs_jk, "Jackknifed two-point functions:" );

  }
  

  ////////////////////////////////
  // Calculate effective masses //
  ////////////////////////////////

  
  effMass( &eMass, &twoPtFuncs_jk );

  // effMass is a function in "physQuants.h" which calculates the effective mass
  // from two-point functions
  
  
  // Print matrix of effective masses

  if ( Print == 1 ) {

    printMatrix( &eMass, "Effective masses:" );

  }
	
  //Calculate average effective mass and its errors

  averageRows( &eMass_avg, &eMass_err, &eMass );

  // averageRows is a function in "jk.h" which averages the values in each row of
  // a matrix resulting in a vector with the same dimension as the number of rows
  // in the original matrix

  // Print average effective masses and their respective errors

  if ( Print == 1 ) {

    printVector( &eMass_avg, "Averaged effective masses:" );

    printVector( &eMass_err, "Effective mass errors:" );

  }
  
  
  ////////////////////////
  // Write output files //
  ////////////////////////


  // Effective masses for each bin and timestep

  char emjkFileName[] = "out/effMasses.dat";

  writeMatrixFile( emjkFileName, &eMass );

  // writeMatrixFile is a function "jk.h" which writes a file containing a matrix
  // organized so that each row is a different timestep

  
  // Averaged effective masses and their errors
  
  char emFileName[] = "out/avgEffMass.dat";

  writeVectorFile( emFileName, &eMass_avg, &eMass_err );
  
  // writeVectorFile is a function in "jk.h" which writes a file with each
  // timestep in the first column of the file, the values in the vector given as
  // the second argument in the second column of the file, and the values in the
  // vector given as the third argument in the third column of the file

  return 0;
	
}
