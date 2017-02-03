/* Calculates the scalar charge from two- and three-point functions and fitted 
   mass. Calculates fitted mass from effective masses.

User input:

1. File name of three-point functions projected to the t direction (P0)
2. File name of two-point functions
3. File name of effective masses
4. File name of effective mass errors
5. File name of file containing momentum squared values and numbers
6. First timeslice to include in fit
7. Last timeslice to include in fit
8. (Optional) 1 for print out of data

*/

#include "jk.h"
#include "physQuants.h"
#include "fitting.h"
#include <cstdlib>

using namespace std;

int main ( int argc, char *argv[] ) {

  int latticeDim = 64; // Space dimension of lattice
  int q2Tot = 55; // Total number of momentum transfer squared's

  
  //////////////////////
  // Read user inputs //
  //////////////////////
  
  
  // Check that all required arguments have been included

  if ( argc < 8 ) {

    cout << "ERROR: Missing input arguments" << endl;
    cout << "Include:" << endl;
    cout << "1. File name of three-point functions projected to the t direction (P0)" << endl;
    cout << "2. File name of two-point functions" << endl;
    cout << "3. File name of effective masses" << endl;
    cout << "4. File name of effective mass errors" << endl;
    cout << "5. File name of file containing momentum squared values and numbers" << endl;
    cout << "6. First timeslice to include in fit" << endl;
    cout << "7. Last timeslice to include in fit" << endl;
    cout << "8. (Optional) 1 for print out of data" << endl;

    return 1;
  }

  
  // If a print argument has been given, set Print (Print = 1 to print)
  
  int Print;
  
  if ( argc >= 9 ) {

    Print = atoi( argv[8] );

  }

  
  // Set delta t and number of configurations based on input files
  
  int TsinkPlusOne = detTimestepNum( argv[1], 7 ); // Because of t=0 this will be Tsink + 1
  int timestepNum_2pt = detTimestepNum( argv[2], 12 ); 
  int timestepNum_effMass = detTimestepNum( argv[3], 2 );
  int timestepNum;
  
  // detTimestepNum is a function in "jk.h" which reads a file of repeating
  // configurations with the timesteps in the first column out of the given
  // number of columns and determines the number of timesteps in the file before
  // a new configuration starts and the timesteps repeat

  
  // Print the number of timesteps in each file
  
  if ( Print == 1 ) {

    cout << endl << "Tsink+1: " << TsinkPlusOne << endl;

    cout << endl << "Number of timesteps for two-point functions: ";
    cout << timestepNum_2pt << endl;

    cout << endl << "Number of timesteps for effective masses: ";
    cout << timestepNum_effMass << endl;
    
    cout << endl;
    
  }

  if ( timestepNum_2pt == timestepNum_effMass ) {

    timestepNum = timestepNum_2pt;

  }
  else {

    cout << "ERROR: Number of timesteps for two-point functions "; 
    cout << timestepNum_2pt << " and effective mass " << timestepNum_effMass;
    cout << " do not match." << endl;

    return 1;

  }
  
  int configNum_3pt = detConfigNum( argv[1], 7 );
  int configNum_2pt = detConfigNum( argv[2], 12 );
 
  // detConfigNum is a function in "jk.h" which reads a file of repeating
  // configurations with a given number of timesteps and columns and counts
  // the total number of configurations in the file

  int configNum;
  
  // Print the number of configurations in each file

  if ( Print == 1 ) {

    cout << endl << "Number of configurations for three-point functions: ";
    cout << configNum_3pt << endl;

    cout << endl << "Number of configurations for two-point functions: ";
    cout << configNum_2pt << endl;

    cout << endl;
    
  }

  
  // Check that both files have the same number of configurations

  if ( configNum_3pt == configNum_2pt ) {

    configNum = configNum_3pt;
  
  }
  else {

    cout << "ERROR: Number of configurations for two- and three-point";
    cout << " functions do not match" << endl;

    return 1;

  }
  
  
  // Set binNum based number of bins in effective mass file

  int binNum = detConfigNum( argv[3], 2 );


  // Check that configNum is exactly divided by binSize

  if( configNum % binNum != 0 ) {
    
    cout << "ERROR: configNum " << configNum << " is divided by binSize ";
    cout << binNum << " with a remainder" << endl;
    
    return 1;

  }

  int binSize = configNum / binNum;


  // Set first and last timeslices to include in the mass fitting

  int fitStart = atoi( argv[6] );

  int fitEnd = atoi( argv[7] );


  ////////////////
  // Initialize //
  ////////////////


  // Vector of momentum transfer squared
  // ( q2[q] )

  vector<int> q2( q2Tot );


  // Vector of number of momentum combinations for transfer squared
  // ( q2Num[q] )

  vector<int> q2Num( q2Tot );


  // Matrix of effective masses
  // ( effMass[t][b] )

  vector< vector<double> > effMass ( timestepNum );

  giveMatrixCols( &effMass, binNum );

  // giveMatrixCols() is a function in "jk.h"
  

  // Vector of effective mass errors
  // ( effMass_err[t] )

  vector<double> effMass_err ( timestepNum );

  
  // Vector of fitted masses
  // ( fitMass[b] )

  vector<double> fitMass ( binNum );


  // Average fitted mass and its error

  double fitMass_avg;
  double fitMass_err;


  // Tensor of three-point functions
  // ( thrPtFunc[q][t][c] )
  
  vector< vector< vector<double> > > thrPtFunc( q2Tot );

  giveTensorMatrix( &thrPtFunc, TsinkPlusOne, configNum );
 
  
  // Tensor of two-point functions
  // ( twoPtFunc[q][t][c] )
  
  vector< vector< vector<double> > > twoPtFunc( q2Tot );

  giveTensorMatrix( &twoPtFunc, timestepNum, configNum );

  
  // Tensor of resampled three-point functions
  // ( thrPtFunc_jk[q][t][b] )

  vector< vector< vector<double> > > thrPtFunc_jk( q2Tot );

  giveTensorMatrix( &thrPtFunc_jk, TsinkPlusOne, binNum );
  
  
  // Tensor of resampled two-point functions
  // ( twoPtFunc_jk[q][t][b] )

  vector< vector< vector<double> > > twoPtFunc_jk( q2Tot );

  giveTensorMatrix( &twoPtFunc_jk, timestepNum, binNum );


  // Tensor of ratio
  // ( ratio[q][t][b] )

  vector< vector< vector<double> > > ratio( q2Tot );

  giveTensorMatrix( &ratio, TsinkPlusOne, binNum );
  
  
  // Tensor of scalar form factors
  // ( scalarFF[q][t][b] )

  vector< vector< vector<double> > > scalarFF( q2Tot );

  giveTensorMatrix( &scalarFF, TsinkPlusOne, binNum );


  // Matrix of averaged scalar charge
  // ( scalarFF_avg[q][t] )

  vector< vector<double> > scalarFF_avg( q2Tot );

  giveMatrixCols( &scalarFF_avg, TsinkPlusOne );
  

  // Matrix of scalar charge errors
  // ( scalarFF_err[q][t] )

  vector< vector<double> > scalarFF_err( q2Tot );

  giveMatrixCols( &scalarFF_err, TsinkPlusOne );  


  ////////////////////////////////////////
  // Read input data files for mass fit //
  ////////////////////////////////////////

  
  // Effective masses

  readNthDataCol( &effMass, argv[3], 2, 2 );

  // readNthDataCol is a functions in "jk.h"

  
  // Print matrix of effective masses

  if ( Print == 1 ) {

    printMatrix( &effMass, "Effective massses:" );

    // printMatrix is a function in "jk.h"

  }
    
  
  // Vector of effective mass errors

  {

    // effMass_errMat needs to be a matrix wih one column so that it can be given
    // to readNthDataCol(). This is done in its own scope so that effMass_errMat
    // does not have to be held in memory for the entire program

    vector< vector<double> > effMass_errMat ( timestepNum );

    giveMatrixCols( &effMass_errMat, 1 );
  
    readNthDataCol( &effMass_errMat, argv[4], 3, 3 );

    // Set effMass_err and effMass_errMat values equal

    for ( int t = 0 ; t < timestepNum; t++ ) {
    
      effMass_err[t] = effMass_errMat[t][0];

    }

  }
  
  // Print effective mass errors

  if ( Print == 1 ) {

    printVector( &effMass_err, "Effective mass errors:" );

    // printVector is a function in "jk.h"

  }
      
    
  //////////////////////////////////////////////////
  // Calculate fitted mass from effective masses  //
  //////////////////////////////////////////////////

  
  fitData( &fitMass, &effMass, &effMass_err, fitStart, fitEnd );

  // Print fitted masses

  if ( Print == 1) {

    printVector( &fitMass, "Fitted masses:" );

  }
  
  // Calculate average fitted mass and its error

  averageFit( &fitMass_avg, &fitMass_err, &fitMass );

  // Print average fitted mass and its error

  if ( Print == 1 ) {

    cout << endl << "Average fitted mass:" << endl;

    cout << fitMass_avg << endl;

    cout << endl << "Fitted mass error:" << endl;

    cout << fitMass_err << endl;

  }


  //////////////////////////////////////
  // Read input files for form factor //
  //////////////////////////////////////


  // Momentum transfer squared
  
  { // Declare a 'q2Tot x 1' matrix to be given to readNthDataCol() and whose
    // elements will be set as the elements of q2[];

    vector< vector<double> > q2_matrix ( q2Tot );

    giveMatrixCols( &q2_matrix, 2 );

    readNthDataCol( &q2_matrix, argv[5], 1, 4 );

    // readNthDataCol() is a function in "jk.h"
    
    for ( int q = 0; q < q2Tot; q++ ) {

      q2[q] = q2_matrix[q][0];

    }

  }

  // Print Q^2 vector

  if ( Print == 1 ) {

    printVector( &q2, "Q^2 vector:" );

    // printVector() is a function of "jk.h"
    
  }
  
  
  // Number of momentum combinations

  { // Do similarly as for q2[]

    vector< vector<double> > q2Num_matrix ( q2Tot );

    giveMatrixCols( &q2Num_matrix, 2 );
    
    readNthDataCol( &q2Num_matrix, argv[5], 2, 4 );

    // readNthDataCol() is a function in "jk.h"
    
    for ( int q = 0; q < q2Tot; q++ ) {

      q2Num[q] = q2Num_matrix[q][0];

    }
       
  }  

  // Print number of combinations

  if ( Print == 1 ) {

    printVector( &q2Num, "Number of combinations for each Q^2:" );

    // printVector() is a function of "jk.h"
    
  }


  // Three-point functions
  
  { // Declare a 4th order tensor which has a dimension for each momentum
    // component combinations of q squared in its own scope and average over
    // momentum combinations

    vector< vector< vector< vector<double> > > > thrPtFunc_qCombo( q2Tot );
    // thrPtFunc_qCombo[q][t][b][n]
    
    giveTensorTensor( &thrPtFunc_qCombo, TsinkPlusOne, configNum, &q2Num );

    // Read data file

    readNthMomDataCol( &thrPtFunc_qCombo, argv[1], &q2Num, 5, 7 );

    // readNthMomDataCol() is a function in "jk.h"

    
    // Average over momentum combinations

    // Initialize a null vector so that averageRows() has a 2nd vector to be
    // filled with the errors
    
    vector<double> nullErr ( configNum );
    
    for ( int q = 0; q < q2Tot; q++ ) {

      for ( int t = 0; t < TsinkPlusOne; t++ ) {
      
	averageRows( &thrPtFunc[q][t], &nullErr, &thrPtFunc_qCombo[q][t] );

	// averageRows() is a function in "jk.h"
	
      }
    }

  }

  // Print three-point functions

  if ( Print == 1 ) {

    printTensor( &thrPtFunc, &q2, "Three-point functions:" );

    // printTensor() is a function of "jk.h"
    
  }
  
      
  // Two-point functions

  { // Do similarly as for three-point functions
    
    vector< vector< vector< vector<double> > > > twoPtFunc_qCombo( q2Tot );

    giveTensorTensor( &twoPtFunc_qCombo, timestepNum, configNum, &q2Num );
    
    // Read data file

    readNthMomDataCol( &twoPtFunc_qCombo, argv[2], &q2Num, 5, 12 );

    // readNthMomDataCol() is a function in "jk.h"

    
    // Average over momentum combinations

    // Initialize a null vector so that averageRows() has a 2nd vector to be
    // filled with the errors
    
    vector<double> nullErr ( configNum );
    
    for ( int q = 0; q < q2Tot; q++ ) {

      for ( int t = 0; t < timestepNum; t++ ) {
	
	averageRows( &twoPtFunc[q][t], &nullErr, &twoPtFunc_qCombo[q][t] );

	// averageRows() is a function in "jk.h"
	
      }
    }
    
  }

  // Print two-point functions

  if ( Print == 1 ) {

    printTensor( &twoPtFunc, &q2, "Two-point functions:" );

    // printTensor() is a function of "jk.h"
    
  }

  
  ///////////////
  // Jackknife //
  ///////////////
  

  // Jackknife three-point functions

  for ( int q = 0; q < q2Tot; q++ ) {
    
    jackknifeMatrix( &thrPtFunc_jk[q], &thrPtFunc[q] );

    // jackknifeMatrix() is a function in "jk.h"
    
  }

  // Print resampled three-point functions

  if ( Print == 1 ) {

    printTensor( &thrPtFunc_jk, &q2, "Resampled three-point functions" );

  }


  // Jackknife two-point functions
  
  for ( int q = 0; q < q2Tot; q++ ) {
  
    jackknifeMatrix( &twoPtFunc_jk[q], &twoPtFunc[q] );

  }

  // Print resample two-point functions

  if ( Print == 1 ) {

    printTensor( &twoPtFunc_jk, &q2, "Resample two-point functions:" );

  }
  
  ///////////////////////////
  // Calculate form factor //
  ///////////////////////////


  // Calculate ratio from two- and three-point functions

  calcRatio( &ratio, &thrPtFunc_jk, &twoPtFunc_jk );

  // calcRatio() is a fucntion in "physQuants.h"

  
  // Print ratios

  if ( Print == 1 ) {

    printTensor( &ratio, &q2, "Ratios:" );

  }

  
  // Calculate form factor from ratio
  
  calcScalarFF( &scalarFF, &ratio, &fitMass, &q2, latticeDim );

  // calcScalarFF() is a function in "physQuants.h"


  // Print form factors

  if ( Print == 1 ) {

    printTensor( &scalarFF, &q2, "Form factors:" );

  }

  
  ///////////////////////
  // Average over bins //
  ///////////////////////


  for ( int q = 0; q < q2Tot; q++ ) {

    averageRows( &scalarFF_avg[q], &scalarFF_err[q], &scalarFF[q] );
  
  }

  // Print averaged form factors

  if ( Print == 1 ) {

    printMatrix( &scalarFF_avg, "Averaged form factors:" );

  }


  ////////////////////////
  // Write output files //
  ////////////////////////
    
  
  // Average fitted mass and its error

  char avgMassFitFilename[] = "out/avgFittedMass.dat";

  writeFitFile( avgMassFitFilename, fitMass_avg, fitMass_err, fitStart, fitEnd );

  // writeFitFile is a function in "jk.h"


  // Average form factor for each Q squared

  for ( int q = 0 ; q < q2Tot; q++ ) {

    char formFactorFile[256];

    sprintf( formFactorFile, "out/avgScalarFF_q2_%d.dat", q2[q] );
	    
    writeVectorFile( formFactorFile, &scalarFF_avg[q], &scalarFF_err[q] );

    // writeVectorFile() is a function in "jk.h"

  }


  // Form factor for each Q squared and bin

  for ( int q = 0; q < q2Tot; q++ ) {

    char formFactorBinnedFile[256];

    sprintf( formFactorBinnedFile, "out/scalarFF_q2_%d.dat", q2[q] );

    writeMatrixFile( formFactorBinnedFile, &scalarFF[q] );

  }

  return 0;
}
