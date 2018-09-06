/* Extracts the 5th column from a given data file and organizes it by timesteps 
   in rows and configurations in columns, calculates the JK averages for each 
   timestep and bin, the effective mass for each timestep and bin, the average 
   and error of the effective mass for each timestep, and writes a file 
   containing the average effective masses with their respective errors for each 
   timestep.

Input: 
1. Path to home directory containing sub-directories of configurations
2. File name template for two-point functions
3. Bin size (make sure numbers of configurations are divided by bin size
   exactly)
4. (Optional) 1 if printouts of the JK averaged two-point functions and 
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

  
  // Check that all required arguments have been included

  if ( argc < 4 ) {

    cout << "Usage:" << endl;
    cout << "1. Path to home directory containing sub-directories of ";
    cout << "configurations" << endl;
    cout << "2. File name template for two-point functions" << endl;
    cout << "3. Bin size" << endl;
    cout << "4. (Optional) 1 for printout of data" << endl;

    return 1;
  }

  // Set path to home directory, where directories of each configuration are

  char *homeDir = argv[1];

  // Set file names for two-point functions contained in their 
  // configuration directory

  char *fnTemplate = argv[2];

  // Set bin size

  int binSize = atoi( argv[3] );

  // If a print argument has been given, set Print (Print = 1 to print)
  
  int Print;
  
  if ( argc == 5 ) {

    Print = atoi( argv[4] );

  }

  // Use ls bash command to get a file listing configuration replica and 
  // trajectories

  char confsFilename[] = "out/conf_list_effMass.txt";

  char lsCommand[256];

  sprintf( lsCommand, "ls -1 %s > %s", homeDir, confsFilename ); 

  system( lsCommand );

  // Read conf_list.txt and store configurations in a vector

  vector<string> confs;

  try {

    readStringFile( &confs, confsFilename );

    // readStringFile() is a function in "jk.h"
  
  }
  catch( string *badFile ) {

    cout << "ERROR(readStringFile): file " << *badFile << " cannot be opened";
    cout << endl;

    return 1;

  }

  // Set filenames

  vector<string> filenames;

  setFilename( &filenames, homeDir, &confs, fnTemplate );

  // setFilename() is a function in "jk.h"

  // Set number of configurations from list

  int configNum = confs.size();

  // Check that configNum is exactly divided by binSize

  while( configNum % binSize != 0 ) {
    
    cout << "WARNING: configNum " << configNum << " is divided by binSize ";
    cout << binSize << " with a remainder. ";
    cout << "Please enter a new binSize." << endl;

    cin >> binSize;

  }

  int binNum = configNum / binSize;

  // Print number of configurations, bins, and binsize

  if( Print == 1 ) {

    cout << "Number of configurations: " << configNum << endl;

    cout << "Bin size: " << binSize << endl;

    cout << "Number of bins: " << binNum << endl;

  }

  // Set number of timesteps based on input files

  int timestepNum;

  try {

    timestepNum = detTimestepNum_rbc( &filenames, 12 ); 
 
    // detTimestepNum_rbc is a function in "jk.h"
 
  }
  catch ( string badConf ) {

    cout << "ERROR(detTimestepNum_rbc): Two-point function in configuration ";
    cout << badConf << " is in error (either its timesteps in the 1st column ";
    cout << "do not start at 0 and/or increase by 1 for each new line, its ";
    cout << "number of timesteps do not match with the file in the first ";
    cout << "configuration, or it cannot be opened)." << endl;
    
    return 1;
 
  }

  if( Print == 1 ) {

    cout << "Number of timesteps: " << timestepNum << endl;

  }

    
  /////////////////////////////////////
  // Initialize matrices and vectors //
  /////////////////////////////////////

	
  // Matrix of two point functions
  // ( twoPtFuncs[t][c] )
  
  vector< vector<double> > twoPtFuncs( timestepNum ); //matrix w/ 'timestepNum' rows

  
  // Matrix of JK averaged TPFs
  // ( twoPtFuncs_jk[t][b] )
  
  vector< vector<double> > twoPtFuncs_jk( timestepNum );  //matrix w/ 'timestepNum' rows

  giveMatrixCols( &twoPtFuncs_jk, binNum );

  // giveMatrixCols() is a function in "jk.h"


  // Matrix of effective masses
  // ( eMass[t][b] )

  vector< vector<double> > eMass( timestepNum );  //matrix w/ 'timestepNum' rows

  giveMatrixCols( &eMass, binNum );


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


  try {

    readNthDataCol_rbc( &twoPtFuncs, &filenames, 5 , 12 );

    // readNthDataCol_rbc is a function in "jk.h"

  }
  catch( string badFile ) {

    cout << "ERROR(readNthDataCol_rbc): Two-point function file " << badFile;
    cout <<  " cannot be opened" << endl;

    return 1;

  }

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

    printMatrix( &twoPtFuncs_jk, "Resampled two-point functions:" );

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

  char emjkFileName[256];

  sprintf( emjkFileName, "out/effMasses_%dConfigs_binSize%d.dat", configNum, binSize );

  writeMatrixFile( emjkFileName, &eMass );

  // writeMatrixFile is a function "jk.h" which writes a file containing a matrix
  // organized so that each row is a different timestep

  
  // Averaged effective masses and their errors
  
  char emFileName[256];

  sprintf( emFileName, "out/avgEffMass_%dConfigs_binSize%d.dat", configNum, binSize );

  writeVectorFile( emFileName, &eMass_avg, &eMass_err );
  
  // writeVectorFile is a function in "jk.h" which writes a file with each
  // timestep in the first column of the file, the values in the vector given as
  // the second argument in the second column of the file, and the values in the
  // vector given as the third argument in the third column of the file

  return 0;
	
}
