/* Reads a given data files containing three-point functions of a zero momentum transfer process projected to each momentum direction with
   the imagenary part of three-point function in the 6th column out of 6 and a given data file of jacknifed two-point functions. From these
   values, the program calculates helicity at each timestep, fits helicity vs. timestep data to a constant value, and
   calculates the error of the fit, outputing helicity vs. timestep to one file for plotting, and the fit and error to another file.

   Input arguments:

    1. Path to home directory containing sub-directories of configurations
    2. File name template for three-point functions projected by G5G1
    3. File name template for three-point functions projected by G5G2
    4. File name template for three-point functions projected by G5G3
    5. File name of two-point functions
    6. File name, including full path, of effective mass
    7. File name, including full path, of effective mass errors
    8. First timeslice to include in effective mass fit
    9. Last timeslice to include in effective mass fit
    10. (Optional) 1 if a printout of values stored in matrices is wanted   
*/

#include "jk.h"
#include "fitting.h"
#include <cstdlib>

using namespace std;

int main ( int argc, char *argv[] ) {

  double Z = 1.1357; // helicity renormalization factor

  
  //////////////////////
  // Read user inputs //
  //////////////////////

  
  // Check that all required arguments have been included
  
  if ( argc < 10 ) {

    cout << "Usage:" << endl;
    cout << "1. Path to home directory containing sub-directories of ";
    cout << "configurations" << endl;  
    
    cout << "2. File name template for three-point functions projected by ";
    cout << "G5G1" << endl;
    
    cout << "3. File name template for three-point functions projected by ";
    cout << "G5G2" << endl;
    
    cout << "4. File name template for three-point functions projected by ";
    cout << "G5G3" << endl;
    
    cout << "5. File name of two-point functions" << endl;
    
    cout << "6. File name, including full path, of effective mass" << endl;
    
    cout << "7. File name, including full path, of effective mass errors" << endl;
    
    cout << "8. First timeslice to include in effective mass fit" << endl;  			    
    
    cout << "9. Last timeslice to include in effective mass fit" << endl;  			    
    
    cout << "10. (Optional) 1 if a printout of values stored in matrices is ";
    cout << "wanted" << endl;

    return 1;
  }

  // Set path to home directory, where directories of each configuration are

  char *homeDir = argv[1];

  // Set file names for G5G1 three-point projected functions contained in their
  // configuration directory 

  char *fnTemplate_3ptG5G1 = argv[2];

  // Set file names for G5G2 three-point projected functions contained in their
  // configuration directory 

  char *fnTemplate_3ptG5G2 = argv[3];

  // Set file names for G5G3 three-point projected functions contained in their
  // configuration directory 

  char *fnTemplate_3ptG5G3 = argv[4];

  // Set file names for two-point functions contained in their 
  // configuration directory

  char *fnTemplate_2pt = argv[5];

  // Set file, including path to effective masses and their errors produced by 
  // effMass.cpp
  
  char *file_em= argv[6];
  char *file_emErr  = argv[7];

  // Set first and last timeslices to include in the mass fitting

  int fitStart = atoi( argv[8] );
  int fitEnd = atoi( argv[9] );

  // If a print argument has been given, set Print (Print = 1 to print)

  int Print;

  if ( argc == 11 ) {

    Print = atoi( argv[10] );

  }

  // Read conf_list.txt and store configurations in a vector

  char confsFilename[]="out/conf_list_effMass.txt";
  
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

  // Set filenames for G5G1 three-point functions

  vector<string> filenames_3ptG5G1;

  setFilename( &filenames_3ptG5G1, homeDir, &confs, fnTemplate_3ptG5G1 );

  // setFilename() is a function in "jk.h"


  // Set filenames for G5G2 three-point functions

  vector<string> filenames_3ptG5G2;

  setFilename( &filenames_3ptG5G2, homeDir, &confs, fnTemplate_3ptG5G2 );

  // Set filenames for G5G3 three-point functions

  vector<string> filenames_3ptG5G3;

  setFilename( &filenames_3ptG5G3, homeDir, &confs, fnTemplate_3ptG5G3 );

  // Set filenames for two-point functions

  vector<string> filenames_2pt;

  setFilename( &filenames_2pt, homeDir, &confs, fnTemplate_2pt );

  // setFilename() is a function in "jk.h"


  // Set number of configuration from list

  int configNum = confs.size();

  // Print the number of configurations
  
  if ( Print == 1 ) {

    cout << "Number of configurations: " << configNum << endl;
    cout << endl;
    
  }

  // Set binNum based on number of bins in effective mass file

  int binNum = detConfigNum( file_em, 2 );

  // Check that configNum is exactly divided by binSize

  if( configNum % binNum != 0 ) {
    
    cout << "ERROR: configNum " << configNum << " is divided by binSize ";
    cout << binNum << " with a remainder" << endl;
    
    return 1;

  }

  int binSize = configNum / binNum;

  // Set Tsink and number of timesteps for two-point functions based on input
  // files

  int TsinkPlusOne; // Because of t=0, this will  be Tsink+1
  int TsinkPlusOne_G5G1;          
  int TsinkPlusOne_G5G2;         
  int TsinkPlusOne_G5G3;         

  int timestepNum_2pt;

  // G5G1 three-point functions

  try {

    TsinkPlusOne_G5G1 = detTimestepNum_rbc( &filenames_3ptG5G1, 8 ); 
 
    // detTimestepNum_rbc is a function in "jk.h"
 
  }
  catch ( string badConf ) {

    cout << "ERROR(detTimestepNum_rbc): Three-point function in configuration ";
    cout << badConf << " is in error (either its timesteps in the 1st column ";
    cout << "do not start at 0 and/or increase by 1 for each new line, its ";
    cout << "number of timesteps do not match with the file in the first ";
    cout << "configuration, or it cannot be opened)." << endl;
    
    return 1;
 
  }

  // G5G2 three-point functions

  try {

    TsinkPlusOne_G5G2 = detTimestepNum_rbc( &filenames_3ptG5G2, 8 ); 
 
    // detTimestepNum_rbc is a function in "jk.h"
 
  }
  catch ( string badConf ) {

    cout << "ERROR(detTimestepNum_rbc): Three-point function in configuration ";
    cout << badConf << " is in error (either its timesteps in the 1st column ";
    cout << "do not start at 0 and/or increase by 1 for each new line, its ";
    cout << "number of timesteps do not match with the file in the first ";
    cout << "configuration, or it cannot be opened)." << endl;
    
    return 1;
 
  }

  // G5G3 three-point functions

  try {

    TsinkPlusOne_G5G3 = detTimestepNum_rbc( &filenames_3ptG5G3, 8 ); 
 
    // detTimestepNum_rbc is a function in "jk.h"
 
  }
  catch ( string badConf ) {

    cout << "ERROR(detTimestepNum_rbc): Three-point function in configuration ";
    cout << badConf << " is in error (either its timesteps in the 1st column ";
    cout << "do not start at 0 and/or increase by 1 for each new line, its ";
    cout << "number of timesteps do not match with the file in the first ";
    cout << "configuration, or it cannot be opened)." << endl;
    
    return 1;
 
  }

  // Two-point functions

  try {

    timestepNum_2pt = detTimestepNum_rbc( &filenames_2pt, 12 ); 
 
    // detTimestepNum_rbc is a function in "jk.h"
 
  }
  catch ( string badConf ) {

    cout << "ERROR(detTimestepNum_rbc): Two-point function in configuration ";
    cout << badConf << " is in error (either its timesteps in the 1st column ";
    cout << "do not start at 0 and/or increase by 1 for each new line or its ";
    cout << "number of timesteps do not match with the file in the first ";
    cout << "configuration, or the a file cannot be opened)." << endl;
    
    return 1;
 
  }

  // Effective mass

  int timestepNum_effMass = detTimestepNum( file_em, 2 );
  int timestepNum;

  // detTimestepNum is a function in "jk.h" which reads a file of repeating
  // configurations with the timesteps in the first column out of the given
  // number of columns and determines the number of timesteps in the file before
  // a new configuration starts and the timesteps repeat

  // Print the number of timesteps in each file
  
  if ( Print == 1 ) {

    cout << "G5G1 Tsink+1: " << TsinkPlusOne_G5G1 << endl;
    cout << "G5G2 Tsink+1: " << TsinkPlusOne_G5G2 << endl;
    cout << "G5G3 Tsink+1: " << TsinkPlusOne_G5G3 << endl;  
    cout << "Number of 2-point timesteps: " << timestepNum_2pt << endl;
    cout << endl;
    
  }
  
  // Check that all 3 files have the same Tsink
  
  if ( TsinkPlusOne_G5G1 == TsinkPlusOne_G5G2 && TsinkPlusOne_G5G2 == TsinkPlusOne_G5G3 ) {

    TsinkPlusOne = TsinkPlusOne_G5G1;
    
  }
  else {

    cout << "ERROR: Number of timesteps in each file do not match" << endl;
    return 1;

  }

  //Check that number of timesteps in 2-point file matches effective mass file

  if ( timestepNum_2pt == timestepNum_effMass ) {

    timestepNum = timestepNum_2pt;

  }
  else {

    cout << "ERROR: Number of timesteps for two-point functions "; 
    cout << timestepNum_2pt << " and effective mass " << timestepNum_effMass;
    cout << " do not match." << endl;

    return 1;

  }
  
  
  /////////////////////////////////////
  // Initialize matrices and vectors //
  /////////////////////////////////////


  // Matrix of effective masses
  // ( effMass[t][b] )

  vector< vector<double> > effMass ( timestepNum );

  giveMatrixCols( &effMass, binNum );

  //giveMatrixCols() is a fucntion in "jk.h"


  // Vector of effective mass errors
  // ( effMass_err[t] )

  vector<double> effMass_err ( timestepNum );

  
  // Vector of fitted masses
  // ( fitMass[b] )

  vector<double> fitMass ( binNum );


  // Average fitted mass and its error

  double fitMass_avg;
  double fitMass_err;


  // Matrix of three-point functions projected to by G5G1
  // ( thrPtFuncs_G5G1[t][c] )

  vector< vector<double> > thrPtFuncs_G5G1(TsinkPlusOne); // matrix w/ 
                                                          // 'TsinkPlusOne' rows


  // Matrix of three-point functions projected by G5G2
  // ( thrPtFuncs_G5G2[t][c] )

  vector< vector<double> > thrPtFuncs_G5G2(TsinkPlusOne); // matrix w/ 
                                                          // 'TsinkPlusOne' rows


  // Matrix of three-point functions projected by G5G3
  // ( thrPtFuncs_G5G3[t][c]

  vector< vector<double> > thrPtFuncs_G5G3(TsinkPlusOne); // matrix w/ 
                                                          // 'TsinkPlusOne' rows

  
  // Matrix of three point functions averaged over projections
  // ( thrPtFuncs_avg[t][c] )

  vector< vector<double> > thrPtFuncs_avg(TsinkPlusOne); // matrix w/ 
                                                         // 'TsinkPlusOne' rows
  
  giveMatrixCols( &thrPtFuncs_avg, configNum ); // Give matrix 'configNum' 
                                                // columns

  // giveMatrixCols() is a function in "jk.h"
    
  // Matrix of resampled three point functions
  // ( thrPtFuncs_jk[t][b] )
  
  vector< vector<double> > thrPtFuncs_jk( TsinkPlusOne ); // matrix w/ 'TsinkPlusOne' rows

  giveMatrixCols( &thrPtFuncs_jk, binNum );


  // Vector of two-point functions at Tsink
  // ( twoPtFuncs[c] )
  
  vector<double> twoPtFuncs;

  // Vector of resampled two-point functions
  // ( twoPtFuncs_jk[b] )
  
  vector<double> twoPtFuncs_jk( binNum ); //vector w/ 'binNum' rows

  // Matrix of helicity
  // ( helicity[t][b] )
  
  vector< vector<double> > helicity( TsinkPlusOne ); // matrix w/ 'TsinkPlusOne' rows

  giveMatrixCols( &helicity, binNum );


  // Vector of helicity averaged over bins
  // ( helicity_avg[t] )
  
  vector<double> helicity_avg( TsinkPlusOne );

  // Vector of helicity errors
  // ( helicity_err[t] )
  
  vector<double> helicity_err( TsinkPlusOne );


  ////////////////////////////////////////
  // Read input data files for mass fit //
  ////////////////////////////////////////

  
  // Effective masses

  readNthDataCol( &effMass, file_em, 2, 2 );

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
  
    readNthDataCol( &effMass_errMat, file_emErr, 3, 3 );

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


  ///////////////////////////////////////////////
  // Fill matrices with values from data files //
  ///////////////////////////////////////////////


  // Three-point functions projected by G5G1
  
  try {

    readNthDataCol_rbc( &thrPtFuncs_G5G1, &filenames_3ptG5G1, 6, 8 );

    // readNthDataCol_rbc is a function in "jk.h"

  }
  catch( string badFile ) {

    cout << "ERROR(readNthDataCol_rbc): Three-point function file " << badFile;
    cout <<  " cannot be opened" << endl;

    return 1;

  }

  // Print matrix of three-point functions

  if ( Print == 1 ) {

    printMatrix( &thrPtFuncs_G5G1, "Three-point functions projected by G5G1:" );
    // printMatrix is a function in "jk.h"
    
  }

  // Three-point functions projected by G5G2
  
  try {

    readNthDataCol_rbc( &thrPtFuncs_G5G2, &filenames_3ptG5G2, 6, 8 );

    // readNthDataCol_rbc is a function in "jk.h"

  }
  catch( string badFile ) {

    cout << "ERROR(readNthDataCol_rbc): Three-point function file " << badFile;
    cout <<  " cannot be opened" << endl;

    return 1;

  }

  // Print matrix of three-point functions

  if ( Print == 1 ) {

    printMatrix( &thrPtFuncs_G5G2, "Three-point functions projected by G5G2:" );
    // printMatrix is a function in "jk.h"
    
  }

  // Three-point functions projected by G5G3
  
  try {

    readNthDataCol_rbc( &thrPtFuncs_G5G3, &filenames_3ptG5G3, 6, 8 );

    // readNthDataCol_rbc is a function in "jk.h"

  }
  catch( string badFile ) {

    cout << "ERROR(readNthDataCol_rbc): Three-point function file " << badFile;
    cout <<  " cannot be opened" << endl;

    return 1;

  }

  // Print matrix of three-point functions

  if ( Print == 1 ) {

    printMatrix( &thrPtFuncs_G5G3, "Three-point functions projected by G5G3:" );
    // printMatrix is a function in "jk.h"
    
  }

  // Vector of two-point functions

  try {

    readNthDataRow_rbc( &twoPtFuncs, &filenames_2pt, TsinkPlusOne - 1, 5, 12 );

  }
  catch( string badFile ) {

    cout << "ERROR(readNthDataRow_rbc): File" << badFile << "could not be ";
    cout << "opened or is not formatted properly" << endl;
  
  }

  // Print two-point functions

  if ( Print == 1 ) {
        
    printVector( &twoPtFuncs, "Two-point functions:" );

    // printVector is a function in "jk.h"

  }


  ////////////////////////////////////////////////////////////
  // Average three point functions over momentum directions //
  ////////////////////////////////////////////////////////////

  
  for ( int t = 0; t < TsinkPlusOne; t++ ) { // Loop over timesteps

    for ( int c = 0; c < configNum; c++ ) { // Loop over configurations

      thrPtFuncs_avg[t][c] = ( thrPtFuncs_G5G1[t][c] + thrPtFuncs_G5G2[t][c] 
			       + thrPtFuncs_G5G3[t][c] ) / 3;

    } // End loop over configurations
  } // End loop over timesteps

  // Print averaged three-point functions

  if ( Print == 1 ) {

    printMatrix( &thrPtFuncs_avg, "Three-point functions averaged over three momentum directions:" );

  }


  ///////////////
  // Jackknife //
  ///////////////
  
  
  // Jackknife three-point functions

  jackknifeMatrix( &thrPtFuncs_jk, &thrPtFuncs_avg );

  // jackkifeMatrix is a function in "jk.h" which calculates the resampled
  // averages of values organized in a matrix by different timesteps in each row
  // and different configurations in each column

  
  // Print resampled three-point functions

  if ( Print == 1 ) {

    printMatrix( &thrPtFuncs_jk, "Resampled three-point functions:" );

  }

  // Jackknife two-point functions

  jackknifeVector ( &twoPtFuncs_jk, &twoPtFuncs );

  // jackknifeVector is a function in "jk.h" which calculates the resampled
  // averages of values organized in a vector by configuration
  
  // Print resampled two-point functions
  
  if ( Print == 1 ) {

    printVector ( &twoPtFuncs_jk, "Resampled two-point functions:" );

  }
  

  ////////////////////////
  // Calculate helicity //
  ////////////////////////
  

  for ( int t = 0; t < TsinkPlusOne; t++ ) { // Loop over timesteps

    for ( int b = 0; b < binNum; b++ ) { // Loop over bins

      helicity[t][b] = 2 * Z / fitMass[b] * thrPtFuncs_jk[t][b] / twoPtFuncs_jk[b];

    }
  }

  // Print matrix of helicity

  if ( Print == 1 ) {

    printMatrix( &helicity, "helicity:" );

  }


  ////////////////////////////////
  // Average helicity over bins //
  ////////////////////////////////
  

  averageRows( &helicity_avg, &helicity_err, &helicity );

  // averageRows is a function in "jk.h" which averages the values in each row of
  // a matrix resulting in a vector with the same dimension as the number of rows
  // in the original matrix

  // Print averaged helicity

  if ( Print == 1 ) {

    printVector( &helicity_avg, "helicity averaged over bins:" );
    
    printVector( &helicity_err, "helicity error:" );
    
  }
  

  ////////////////////////
  // Write output files //
  ////////////////////////

  
  // Average fitted mass and its error

  char avgMassFitFilename[256];

  sprintf( avgMassFitFilename, "out/avgFittedMass_%dConfigs_binSize%d.dat", configNum, binSize );

  writeFitFile( avgMassFitFilename, fitMass_avg, fitMass_err, fitStart, fitEnd, TsinkPlusOne - 1 );

  // writeFitFile is a function in "jk.h"


  char acFileName[256];

  // Set file name to include the proper delta t value
  
  sprintf( acFileName, "out/helicity_dt%d_%dConfigs_binSize%d.dat", TsinkPlusOne - 1, configNum, binSize );

  writeVectorFile( acFileName, &helicity_avg, &helicity_err );

  // writeVectorFile is a function in "jk.h" which writes a file with each
  // timestep in the first column of the file, the values in the vector given as
  // the second argument in the second column of the file, and the values in the
  // vector given as the third argument in the third column of the file


  //////////////////////////////////////////////////////////////////////
  // Fit helicity to a constant parameter for each bin, calculate the //
  // average fit and its error, and write them to an output file      //
  //////////////////////////////////////////////////////////////////////
  

  switch( TsinkPlusOne ) { // delta t determines which timeslices to include in the fit

  case 13 : {

    // t=4 to t=8

    // Fit each bin
    
    vector<double> helicity_fit_4_8( binNum );

    fitData( &helicity_fit_4_8, &helicity, &helicity_err, 4, 8 );

    // Calculate the average fit and it error

    double helicity_fitAvg_4_8;

    double helicity_fitErr_4_8;

    averageFit( &helicity_fitAvg_4_8, &helicity_fitErr_4_8, &helicity_fit_4_8 );

    // Print helicity fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &helicity_fit_4_8, "helicity fits (t=4 to t=8):" );

      cout << "Average helicity fit: " << helicity_fitAvg_4_8 << endl;

      cout << endl;

      cout << "helicity fit error: " << helicity_fitErr_4_8 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_4_8[256];

    sprintf( fitFileName_4_8, "out/helicityFit_dt12_4_8_%dConfigs_binSize%d.dat", configNum, binSize );

    writeFitFile( fitFileName_4_8, helicity_fitAvg_4_8, helicity_fitErr_4_8, 4, 8, TsinkPlusOne - 1 );

    // t=3 to t=9

    // Fit each bin
    
    vector<double> helicity_fit_3_9( binNum );

    fitData( &helicity_fit_3_9, &helicity, &helicity_err, 3, 9 );

    // Calculate the average fit and it error

    double helicity_fitAvg_3_9;

    double helicity_fitErr_3_9;

    averageFit( &helicity_fitAvg_3_9, &helicity_fitErr_3_9, &helicity_fit_3_9 );

    // Print helicity fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &helicity_fit_3_9, "helicity fits (t=3 to t=9):" );

      cout << "Average helicity fit: " << helicity_fitAvg_3_9 << endl;

      cout << endl;

      cout << "helicity fit error: " << helicity_fitErr_3_9 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_3_9[256];

    sprintf( fitFileName_3_9, "out/helicityFit_dt12_3_9_%dConfigs_binSize%d.dat", configNum, binSize );

    writeFitFile( fitFileName_3_9, helicity_fitAvg_3_9, helicity_fitErr_3_9, 3, 9, TsinkPlusOne - 1 );


    // t=2 to t=10

    // Fit each bin
    
    vector<double> helicity_fit_2_10( binNum );

    fitData( &helicity_fit_2_10, &helicity, &helicity_err, 2, 10 );

    // Calculate the average fit and it error

    double helicity_fitAvg_2_10;

    double helicity_fitErr_2_10;

    averageFit( &helicity_fitAvg_2_10, &helicity_fitErr_2_10, &helicity_fit_2_10 );

    // Print helicity fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &helicity_fit_2_10, "helicity fits (t=2 to t=10):" );

      cout << "Average helicity fit: " << helicity_fitAvg_2_10 << endl;

      cout << endl;

      cout << "helicity fit error: " << helicity_fitErr_2_10 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_2_10[256];

    sprintf( fitFileName_2_10, "out/helicityFit_dt12_2_10_%dConfigs_binSize%d.dat", configNum, binSize );

    writeFitFile( fitFileName_2_10, helicity_fitAvg_2_10, helicity_fitErr_2_10, 2, 10, TsinkPlusOne - 1 );

        
    break;
  }

  case 15 : {

    // t=5 to t=9

    // Fit each bin
    
    vector<double> helicity_fit_5_9( binNum );

    fitData( &helicity_fit_5_9, &helicity, &helicity_err, 5, 9 );

    // Calculate the average fit and it error

    double helicity_fitAvg_5_9;

    double helicity_fitErr_5_9;

    averageFit( &helicity_fitAvg_5_9, &helicity_fitErr_5_9, &helicity_fit_5_9 );

    // Print helicity fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &helicity_fit_5_9, "helicity fits (t=5 to t=9):" );

      cout << "Average helicity fit: " << helicity_fitAvg_5_9 << endl;

      cout << endl;

      cout << "helicity fit error: " << helicity_fitErr_5_9 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_5_9[256];

    sprintf( fitFileName_5_9, "out/helicityFit_dt14_5_9_%dConfigs_binSize%d.dat", configNum, binSize );

    writeFitFile( fitFileName_5_9, helicity_fitAvg_5_9, helicity_fitErr_5_9, 5, 9, TsinkPlusOne - 1 );

        
    // t=4 to t=10

    // Fit each bin

    vector<double> helicity_fit_4_10( binNum );
   
    fitData( &helicity_fit_4_10, &helicity, &helicity_err, 4, 10 );

    // Calculate the average fit and it error

    double helicity_fitAvg_4_10;

    double helicity_fitErr_4_10;

    averageFit( &helicity_fitAvg_4_10, &helicity_fitErr_4_10, &helicity_fit_4_10 );

    // Print helicity fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &helicity_fit_4_10, "helicity fits (t=4 to t=10):" );
      
      cout << "Average helicity fit: " << helicity_fitAvg_4_10 << endl;

      cout << endl;

      cout << "helicity fit error: " << helicity_fitErr_4_10 << endl;
      
      cout << endl;
      
    }
    
    // Write the average fit and its error to an output file

    char fitFileName_4_10[256];

    sprintf( fitFileName_4_10, "out/helicityFit_dt14_4_10_%dConfigs_binSize%d.dat", configNum, binSize );

    writeFitFile( fitFileName_4_10, helicity_fitAvg_4_10, helicity_fitErr_4_10, 4, 10, TsinkPlusOne - 1 );
    
    // t=3 to t=11

    // Fit each bin
    
    vector<double> helicity_fit_3_11( binNum );

    fitData( &helicity_fit_3_11, &helicity, &helicity_err, 3, 11 );

    // Calculate the average fit and it error

    double helicity_fitAvg_3_11;

    double helicity_fitErr_3_11;

    averageFit( &helicity_fitAvg_3_11, &helicity_fitErr_3_11, &helicity_fit_3_11 );

    // Print helicity fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &helicity_fit_3_11, "helicity fits (t=3 to t=11):" );

      cout << "Average helicity fit: " << helicity_fitAvg_3_11 << endl;

      cout << endl;

      cout << "helicity fit error: " << helicity_fitErr_3_11 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_3_11[256];

    sprintf( fitFileName_3_11, "out/helicityFit_dt14_3_11_%dConfigs_binSize%d.dat", configNum, binSize );

    writeFitFile( fitFileName_3_11, helicity_fitAvg_3_11, helicity_fitErr_3_11, 3, 11, TsinkPlusOne - 1 );
    
    
    // t=2 to t=12

    // Fit each bin
    
    vector<double> helicity_fit_2_12( binNum );

    fitData( &helicity_fit_2_12, &helicity, &helicity_err, 2, 12 );

    // Calculate the average fit and it error

    double helicity_fitAvg_2_12;

    double helicity_fitErr_2_12;

    averageFit( &helicity_fitAvg_2_12, &helicity_fitErr_2_12, &helicity_fit_2_12 );

    // Print helicity fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &helicity_fit_2_12, "helicity fits (t=2 to t=12):" );

      cout << "Average helicity fit: " << helicity_fitAvg_2_12 << endl;

      cout << endl;

      cout << "helicity fit error: " << helicity_fitErr_2_12 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_2_12[256];

    sprintf( fitFileName_2_12, "out/helicityFit_dt14_2_12_%dConfigs_binSize%d.dat", configNum, binSize );

    writeFitFile( fitFileName_2_12, helicity_fitAvg_2_12, helicity_fitErr_2_12, 2, 12, TsinkPlusOne - 1 );
    
    
    break;
  }
    
  case 17 : {
    
    // t=6 to t=10

    // Fit each bin
    
    vector<double> helicity_fit_6_10( binNum );
    
    fitData( &helicity_fit_6_10, &helicity, &helicity_err, 6, 10 );

    // Calculate the average fit and its error

    double helicity_fitAvg_6_10;

    double helicity_fitErr_6_10;

    averageFit( &helicity_fitAvg_6_10, &helicity_fitErr_6_10, &helicity_fit_6_10 );

    // Print helicity fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &helicity_fit_6_10, "helicity fits (t=6 to t=10):" );
      
      cout << "Average helicity fit: " << helicity_fitAvg_6_10 << endl;

      cout << endl;

      cout << "helicity fit error: " << helicity_fitErr_6_10 << endl;
      
      cout << endl;
      
    }

    
    // Write the average fit and its error to an output file

    char fitFileName_6_10[256];

    sprintf( fitFileName_6_10, "out/helicityFit_dt16_6_10_%dConfigs_binSize%d.dat", configNum, binSize );

    writeFitFile( fitFileName_6_10, helicity_fitAvg_6_10, helicity_fitErr_6_10, 6, 10, TsinkPlusOne - 1 );    

    // t=5 to t=11

    // Fit each bin
    
    vector<double> helicity_fit_5_11( binNum );
    
    fitData( &helicity_fit_5_11, &helicity, &helicity_err, 5, 11 );

    // Calculate the average fit and its error

    double helicity_fitAvg_5_11;

    double helicity_fitErr_5_11;

    averageFit( &helicity_fitAvg_5_11, &helicity_fitErr_5_11, &helicity_fit_5_11 );
    
    // Print helicity fits for each bin, the averaged fit, and its error
    
    if ( Print == 1 ) {

      printVector( &helicity_fit_5_11, "helicity fits (t=5 to t=11):" );
      
      cout << "Average helicity fit: " << helicity_fitAvg_5_11 << endl;

      cout << endl;

      cout << "helicity fit error: " << helicity_fitErr_5_11 << endl;
      
      cout << endl;
      
    }
    
    // Write the average fit and its error to an output file

    char fitFileName_5_11[256];

    sprintf( fitFileName_5_11, "out/helicityFit_dt16_5_11_%dConfigs_binSize%d.dat", configNum, binSize );

    writeFitFile( fitFileName_5_11, helicity_fitAvg_5_11, helicity_fitErr_5_11, 5, 11, TsinkPlusOne - 1 );


    // t=4 to t=12

    // Fit each bin
    
    vector<double> helicity_fit_4_12( binNum );
    
    fitData( &helicity_fit_4_12, &helicity, &helicity_err, 4, 12 );

    // Calculate the average fit and its error

    double helicity_fitAvg_4_12;

    double helicity_fitErr_4_12;

    averageFit( &helicity_fitAvg_4_12, &helicity_fitErr_4_12, &helicity_fit_4_12 );
    
    // Print helicity fits for each bin, the averaged fit, and its error
    
    if ( Print == 1 ) {

      printVector( &helicity_fit_4_12, "helicity fits (t=4 to t=12):" );
      
      cout << "Average helicity fit: " << helicity_fitAvg_4_12 << endl;

      cout << endl;

      cout << "helicity fit error: " << helicity_fitErr_4_12 << endl;
      
      cout << endl;
      
    }
    
    // Write the average fit and its error to an output file

    char fitFileName_4_12[256];

    sprintf( fitFileName_4_12, "out/helicityFit_dt16_4_12_%dConfigs_binSize%d.dat", configNum, binSize );

    writeFitFile( fitFileName_4_12, helicity_fitAvg_4_12, helicity_fitErr_4_12, 4, 12, TsinkPlusOne - 1 );


    // t=3 to t=13

    // Fit each bin
    
    vector<double> helicity_fit_3_13( binNum );
    
    fitData( &helicity_fit_3_13, &helicity, &helicity_err, 3, 13 );

    // Calculate the average fit and its error

    double helicity_fitAvg_3_13;

    double helicity_fitErr_3_13;

    averageFit( &helicity_fitAvg_3_13, &helicity_fitErr_3_13, &helicity_fit_3_13 );
    
    // Print helicity fits for each bin, the averaged fit, and its error
    
    if ( Print == 1 ) {

      printVector( &helicity_fit_3_13, "helicity fits (t=3 to t=13):" );
      
      cout << "Average helicity fit: " << helicity_fitAvg_3_13 << endl;

      cout << endl;

      cout << "helicity fit error: " << helicity_fitErr_3_13 << endl;
      
      cout << endl;
      
    }
    
    // Write the average fit and its error to an output file

    char fitFileName_3_13[256];

    sprintf( fitFileName_3_13, "out/helicityFit_dt16_3_13_%dConfigs_binSize%d.dat", configNum, binSize );

    writeFitFile( fitFileName_3_13, helicity_fitAvg_3_13, helicity_fitErr_3_13, 3, 13, TsinkPlusOne - 1 );


    break;
  }
    
  }

  return 0;
}
