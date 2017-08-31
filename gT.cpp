/* Reads given data files containing three-point functions of a zero momentum 
   transfer process projected by G5G1, G5G2, and G5G3 with the real part of 
   three-point function in the 5th out of 7 columns and given data files of 
   two-point functions. From these values, the program calculates the resampled 
   two- and three-point functions, calculates gT at each timestep, fits the gT 
   vs. timestep data to a constant value, and calculates the error of the fit, 
   outputing the gT vs. timestep to one file for plotting, and the average fits 
   and their errors to another file.

   Input arguments:

   1. Path to home directory containing sub-directories of configurations
   2. File name template for three-point functions projected by G5G1
   3. File name template for three-point functions projected by G5G2
   4. File name template for three-point functions projected by G5G3
   5. File name template for two-point functions
   6. Bin size
   7. (Optional) 1 for printout of data
*/

#include "jk.h"
#include "fitting.h"
#include <cstdlib>

using namespace std;

int main ( int argc, char *argv[] ) {

  double zT = 0.8551; // Renormalization factor

  
  //////////////////////
  // Read user inputs //
  //////////////////////

  
  // Check that all required arguments have been included
  
  if ( argc < 7 ) {

    cout << "Usage" << endl;

    cout << "1. Path to home directory containing sub-directories of ";
    cout << "configurations" << endl;

    cout << "2. File name template for three-point functions projected by";
    cout << " G5G1 with the configuration name represented by a '*'" << endl;

    cout << "3. File name template for three-point functions projected by";
    cout << " G5G2" << endl;

    cout << "4. File name template for three-point functions projected by";
    cout << " G5G3" << endl;

    cout << "5. File name template for two-point functions" << endl;

    cout << "6. Bin size" << endl;

    cout << "7. (Optional) 1 for printout of data" << endl;

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

  // Set bin size

  int binSize = atoi( argv[6] );

  // If a print argument has been given, set Print (Print = 1 to print)
  
  int Print;
  
  if ( argc == 8 ) {

    Print = atoi( argv[7] );

  }

  // Use ls bash command to get a file listing configuration replica and 
  // trajectories

  char confsFilename[] = "out/conf_list_gT.txt";

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

  // Set number of configurations from list

  int configNum = confs.size();

  // Check that configNum is exactly divided by binSize

  while( configNum % binSize != 0 ) {
    
    cout << "WARNING: configNum " << configNum << " is divided by binSize " << binSize << " with a remainder. ";
    cout << "Please enter a new binSize." << endl;

    cin >> binSize;

  }

  int binNum = configNum / binSize;

  // Print number of configurations, bins, and bin size

  if( Print == 1 ) {

    cout << "Number of configurations: " << configNum << endl;

    cout << "Bin size: " << binSize << endl;

    cout << "Number of bins: " << binNum << endl;

  }

  // Set Tsink based on input files
   
  int TsinkPlusOne; // Because of t=0, this will be Tsink+1
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

  // two-point functions

  try {

    timestepNum_2pt = detTimestepNum_rbc( &filenames_2pt, 12 ); 
 
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

  
  /////////////////////////////////////
  // Initialize matrices and vectors //
  /////////////////////////////////////


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
  
  vector< vector<double> > thrPtFuncs_jk(TsinkPlusOne); // matrix w/ 
                                                        // 'TsinkPlusOne' rows

  giveMatrixCols( &thrPtFuncs_jk, binNum ); // Give matrix 'binNum' columns

  // Vector of two-point functions at Tsink
  // ( twoPtFuncs[c] )
  
  vector<double> twoPtFuncs(configNum);


  // Vector of resampled two-point functions
  // ( twoPtFuncs_jk[b] )
  
  vector<double> twoPtFuncs_jk(binNum); //vector w/ 'binNum' rows

  // Matrix of gTs
  // ( gT[t][b] )
  
  vector< vector<double> > gT(TsinkPlusOne); // matrix w/ 
                                                      // 'TsinkPlusOne' rows

  giveMatrixCols( &gT, binNum ); // Give matrix 'binNum' columns
  
  // Vector of gTs averaged over bins
  // ( gT_[t] )
  
  vector<double> gT_avg(TsinkPlusOne);

  // Vector of gT errors
  // ( gT_err[t] )
  
  vector<double> gT_err(TsinkPlusOne);


  ///////////////////////////////////////////////
  // Fill matrices with values from data files //
  ///////////////////////////////////////////////


  // Three-point functions projected by G5G1
  
  try {

    readNthDataCol_rbc( &thrPtFuncs_G5G1, &filenames_3ptG5G1, 5, 8 );

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

    readNthDataCol_rbc( &thrPtFuncs_G5G2, &filenames_3ptG5G2, 5, 8 );

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

    readNthDataCol_rbc( &thrPtFuncs_G5G3, &filenames_3ptG5G3, 5, 8 );

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

  { // Read 2pt functions into matrix, set vector 2pt fucntions at Tsink

    vector< vector<double> > twoPtFuncs_mat(timestepNum_2pt);

    try {

      readNthDataCol_rbc( &twoPtFuncs_mat, &filenames_2pt, 5, 12 );

      // readNthDataCol_rbc is a function in "jk.h"

    }
    catch( string badFile ) {

      cout << "ERROR(readNthDataCol_rbc): Two-point function file " << badFile;
      cout <<  " cannot be opened" << endl;

      return 1;

    }

    for( int c = 0; c < configNum; c++ ) { // Loop over configurations

      twoPtFuncs[c] = twoPtFuncs_mat[TsinkPlusOne-1][c];

    } // End loop over configurations

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

      thrPtFuncs_avg[t][c] = ( thrPtFuncs_G5G1[t][c] - thrPtFuncs_G5G2[t][c] 
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

  jackknifeMatrix ( &thrPtFuncs_jk, &thrPtFuncs_avg );

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
  

  //////////////////
  // Calculate gT //
  //////////////////
  

  for ( int t = 0; t < TsinkPlusOne; t++ ) { // Loop over timesteps

    for ( int b = 0; b < binNum; b++ ) { // Loop over bins

      gT[t][b] = zT * thrPtFuncs_jk[t][b] / twoPtFuncs_jk[b];

    }
  }

  // Print matrix of gT

  if ( Print == 1 ) {

    printMatrix( &gT, "gT:" );

  }


  //////////////////////////
  // Average gT over bins //
  //////////////////////////
  

  averageRows( &gT_avg, &gT_err, &gT );

  // averageRows is a function in "jk.h" which averages the values in each row of
  // a matrix resulting in a vector with the same dimension as the number of rows
  // in the original matrix

  // Print averaged gT

  if ( Print == 1 ) {

    printVector( &gT_avg, "gT averaged over bins:" );
    
    printVector( &gT_err, "gT error:" );
    
  }
  

  ///////////////////////////////////////////////////////
  // Write output file containing gT vs. timestep data //
  ///////////////////////////////////////////////////////

  
  char acFileName[256];

  // Set file name to include the proper delta t value
  
  sprintf( acFileName, "out/gT_dt%d_%dConfigs_binSize%d.dat", 
	   TsinkPlusOne - 1, configNum, binSize );

  writeVectorFile( acFileName, &gT_avg, &gT_err );

  // writeVectorFile is a function in "jk.h" which writes a file with each
  // timestep in the first column of the file, the values in the vector given as
  // the second argument in the second column of the file, and the values in the
  // vector given as the third argument in the third column of the file


  ////////////////////////////////////////////////////////////////////
  // Fit the gT to a constant parameter for each bin, calculate the //
  // average fit and its error, and write them to an output file    //
  ////////////////////////////////////////////////////////////////////
  

  switch( TsinkPlusOne ) { // delta t determines which timeslices to include in the fit

  case 13 : { // Tsink = 12

    // t=4 to t=8

    // Fit each bin
    
    vector<double> gT_fit_4_8( binNum );

    fitData( &gT_fit_4_8, &gT, &gT_err, 4, 8 );

    // Calculate the average fit and it error

    double gT_fitAvg_4_8;

    double gT_fitErr_4_8;

    averageFit( &gT_fitAvg_4_8, &gT_fitErr_4_8, &gT_fit_4_8 );

    // Print the gT fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &gT_fit_4_8, "gT fits (t=4 to t=8):" );

      cout << "Average gT fit: " << gT_fitAvg_4_8 << endl;

      cout << endl;

      cout << "gT fit error: " << gT_fitErr_4_8 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_4_8[256];
    
    sprintf( fitFileName_4_8, "out/gTFit_dt%d_%d_%d_%dConfigs_binSize%d.dat",
	     TsinkPlusOne - 1, 4, 8, configNum, binSize );

    writeFitFile( fitFileName_4_8, gT_fitAvg_4_8, gT_fitErr_4_8, 4, 8, TsinkPlusOne - 1 );

    // t=3 to t=9

    // Fit each bin
    
    vector<double> gT_fit_3_9( binNum );

    fitData( &gT_fit_3_9, &gT, &gT_err, 3, 9 );

    // Calculate the average fit and it error

    double gT_fitAvg_3_9;

    double gT_fitErr_3_9;

    averageFit( &gT_fitAvg_3_9, &gT_fitErr_3_9, &gT_fit_3_9 );

    // Print the gT fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &gT_fit_3_9, "gT fits (t=3 to t=9):" );

      cout << "Average gT fit: " << gT_fitAvg_3_9 << endl;

      cout << endl;

      cout << "gT fit error: " << gT_fitErr_3_9 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_3_9[256];

    sprintf( fitFileName_3_9, "out/gTFit_dt%d_%d_%d_%dConfigs_binSize%d.dat",
	     TsinkPlusOne - 1, 3, 9, configNum, binSize );

    writeFitFile( fitFileName_3_9, gT_fitAvg_3_9, gT_fitErr_3_9, 3, 9, TsinkPlusOne - 1 );


    // t=2 to t=10

    // Fit each bin
    
    vector<double> gT_fit_2_10( binNum );

    fitData( &gT_fit_2_10, &gT, &gT_err, 2, 10 );

    // Calculate the average fit and it error

    double gT_fitAvg_2_10;

    double gT_fitErr_2_10;

    averageFit( &gT_fitAvg_2_10, &gT_fitErr_2_10, &gT_fit_2_10 );

    // Print the gT fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &gT_fit_2_10, "gT fits (t=2 to t=10):" );

      cout << "Average gT fit: " << gT_fitAvg_2_10 << endl;

      cout << endl;

      cout << "gT fit error: " << gT_fitErr_2_10 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_2_10[256];

    sprintf( fitFileName_2_10, "out/gTFit_dt%d_%d_%d_%dConfigs_binSize%d.dat",
	     TsinkPlusOne - 1, 2, 10, configNum, binSize );

    writeFitFile( fitFileName_2_10, gT_fitAvg_2_10, gT_fitErr_2_10, 2, 10, TsinkPlusOne - 1 );

        
    break;
  }

  case 15 : { // Tsink = 14

    // t=5 to t=9

    // Fit each bin
    
    vector<double> gT_fit_5_9( binNum );

    fitData( &gT_fit_5_9, &gT, &gT_err, 5, 9 );

    // Calculate the average fit and it error

    double gT_fitAvg_5_9;

    double gT_fitErr_5_9;

    averageFit( &gT_fitAvg_5_9, &gT_fitErr_5_9, &gT_fit_5_9 );

    // Print the gT fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &gT_fit_5_9, "gT fits (t=5 to t=9):" );

      cout << "Average gT fit: " << gT_fitAvg_5_9 << endl;

      cout << endl;

      cout << "gT fit error: " << gT_fitErr_5_9 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_5_9[256];

    sprintf( fitFileName_5_9, "out/gTFit_dt%d_%d_%d_%dConfigs_binSize%d.dat",
	     TsinkPlusOne - 1, 5, 9, configNum, binSize );

    writeFitFile( fitFileName_5_9, gT_fitAvg_5_9, gT_fitErr_5_9, 5, 9, TsinkPlusOne - 1 );

        
    // t=4 to t=10

    // Fit each bin

    vector<double> gT_fit_4_10( binNum );
   
    fitData( &gT_fit_4_10, &gT, &gT_err, 4, 10 );

    // Calculate the average fit and it error

    double gT_fitAvg_4_10;

    double gT_fitErr_4_10;

    averageFit( &gT_fitAvg_4_10, &gT_fitErr_4_10, &gT_fit_4_10 );

    // Print the gT fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &gT_fit_4_10, "gT fits (t=4 to t=10):" );
      
      cout << "Average gT fit: " << gT_fitAvg_4_10 << endl;

      cout << endl;

      cout << "gT fit error: " << gT_fitErr_4_10 << endl;
      
      cout << endl;
      
    }
    
    // Write the average fit and its error to an output file

    char fitFileName_4_10[256];

    sprintf( fitFileName_4_10, "out/gTFit_dt%d_%d_%d_%dConfigs_binSize%d.dat",
	     TsinkPlusOne - 1, 4, 10, configNum, binSize );

    writeFitFile( fitFileName_4_10, gT_fitAvg_4_10, gT_fitErr_4_10, 4, 10, TsinkPlusOne - 1 );
    
    // t=3 to t=11

    // Fit each bin
    
    vector<double> gT_fit_3_11( binNum );

    fitData( &gT_fit_3_11, &gT, &gT_err, 3, 11 );

    // Calculate the average fit and it error

    double gT_fitAvg_3_11;

    double gT_fitErr_3_11;

    averageFit( &gT_fitAvg_3_11, &gT_fitErr_3_11, &gT_fit_3_11 );

    // Print the gT fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &gT_fit_3_11, "gT fits (t=3 to t=11):" );

      cout << "Average gT fit: " << gT_fitAvg_3_11 << endl;

      cout << endl;

      cout << "gT fit error: " << gT_fitErr_3_11 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_3_11[256];

    sprintf( fitFileName_3_11, "out/gTFit_dt%d_%d_%d_%dConfigs_binSize%d.dat",
	     TsinkPlusOne - 1, 3, 11, configNum, binSize );

    writeFitFile( fitFileName_3_11, gT_fitAvg_3_11, gT_fitErr_3_11, 3, 11, TsinkPlusOne - 1 );
    
    
    // t=2 to t=12

    // Fit each bin
    
    vector<double> gT_fit_2_12( binNum );

    fitData( &gT_fit_2_12, &gT, &gT_err, 2, 12 );

    // Calculate the average fit and it error

    double gT_fitAvg_2_12;

    double gT_fitErr_2_12;

    averageFit( &gT_fitAvg_2_12, &gT_fitErr_2_12, &gT_fit_2_12 );

    // Print the gT fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &gT_fit_2_12, "gT fits (t=2 to t=12):" );

      cout << "Average gT fit: " << gT_fitAvg_2_12 << endl;

      cout << endl;

      cout << "gT fit error: " << gT_fitErr_2_12 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_2_12[256];

    sprintf( fitFileName_2_12, "out/gTFit_dt%d_%d_%d_%dConfigs_binSize%d.dat",
	     TsinkPlusOne - 1, 2, 12, configNum, binSize );

    writeFitFile( fitFileName_2_12, gT_fitAvg_2_12, gT_fitErr_2_12, 2, 12, TsinkPlusOne - 1 );
    
    
    break;
  }
    
  case 17 : { // Tsink = 16
    
    // t=6 to t=10

    // Fit each bin
    
    vector<double> gT_fit_6_10( binNum );
    
    fitData( &gT_fit_6_10, &gT, &gT_err, 6, 10 );

    // Calculate the average fit and its error

    double gT_fitAvg_6_10;

    double gT_fitErr_6_10;

    averageFit( &gT_fitAvg_6_10, &gT_fitErr_6_10, &gT_fit_6_10 );

    // Print the gT fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &gT_fit_6_10, "gT fits (t=6 to t=10):" );
      
      cout << "Average gT fit: " << gT_fitAvg_6_10 << endl;

      cout << endl;

      cout << "gT fit error: " << gT_fitErr_6_10 << endl;
      
      cout << endl;
      
    }

    
    // Write the average fit and its error to an output file

    char fitFileName_6_10[256];

    sprintf( fitFileName_6_10, "out/gTFit_dt%d_%d_%d_%dConfigs_binSize%d.dat",
	     TsinkPlusOne - 1, 6, 10, configNum, binSize );

    writeFitFile( fitFileName_6_10, gT_fitAvg_6_10, gT_fitErr_6_10, 6, 10, TsinkPlusOne - 1 );    

    // t=5 to t=11

    // Fit each bin
    
    vector<double> gT_fit_5_11( binNum );
    
    fitData( &gT_fit_5_11, &gT, &gT_err, 5, 11 );

    // Calculate the average fit and its error

    double gT_fitAvg_5_11;

    double gT_fitErr_5_11;

    averageFit( &gT_fitAvg_5_11, &gT_fitErr_5_11, &gT_fit_5_11 );
    
    // Print the gT fits for each bin, the averaged fit, and its error
    
    if ( Print == 1 ) {

      printVector( &gT_fit_5_11, "gT fits (t=5 to t=11):" );
      
      cout << "Average gT fit: " << gT_fitAvg_5_11 << endl;

      cout << endl;

      cout << "gT fit error: " << gT_fitErr_5_11 << endl;
      
      cout << endl;
      
    }
    
    // Write the average fit and its error to an output file

    char fitFileName_5_11[256];

    sprintf( fitFileName_5_11, "out/gTFit_dt%d_%d_%d_%dConfigs_binSize%d.dat",
	     TsinkPlusOne - 1, 5, 11, configNum, binSize );

    writeFitFile( fitFileName_5_11, gT_fitAvg_5_11, gT_fitErr_5_11, 5, 11, TsinkPlusOne - 1 );


    // t=4 to t=12

    // Fit each bin
    
    vector<double> gT_fit_4_12( binNum );
    
    fitData( &gT_fit_4_12, &gT, &gT_err, 4, 12 );

    // Calculate the average fit and its error

    double gT_fitAvg_4_12;

    double gT_fitErr_4_12;

    averageFit( &gT_fitAvg_4_12, &gT_fitErr_4_12, &gT_fit_4_12 );
    
    // Print the gT fits for each bin, the averaged fit, and its error
    
    if ( Print == 1 ) {

      printVector( &gT_fit_4_12, "gT fits (t=4 to t=12):" );
      
      cout << "Average gT fit: " << gT_fitAvg_4_12 << endl;

      cout << endl;

      cout << "gT fit error: " << gT_fitErr_4_12 << endl;
      
      cout << endl;
      
    }
    
    // Write the average fit and its error to an output file

    char fitFileName_4_12[256];

    sprintf( fitFileName_4_12, "out/gTFit_dt%d_%d_%d_%dConfigs_binSize%d.dat",
	     TsinkPlusOne - 1, 4, 12, configNum, binSize );

    writeFitFile( fitFileName_4_12, gT_fitAvg_4_12, gT_fitErr_4_12, 4, 12, TsinkPlusOne - 1 );


    // t=3 to t=13

    // Fit each bin
    
    vector<double> gT_fit_3_13( binNum );
    
    fitData( &gT_fit_3_13, &gT, &gT_err, 3, 13 );

    // Calculate the average fit and its error

    double gT_fitAvg_3_13;

    double gT_fitErr_3_13;

    averageFit( &gT_fitAvg_3_13, &gT_fitErr_3_13, &gT_fit_3_13 );
    
    // Print the gT fits for each bin, the averaged fit, and its error
    
    if ( Print == 1 ) {

      printVector( &gT_fit_3_13, "gT fits (t=3 to t=13):" );
      
      cout << "Average gT fit: " << gT_fitAvg_3_13 << endl;

      cout << endl;

      cout << "gT fit error: " << gT_fitErr_3_13 << endl;
      
      cout << endl;
      
    }
    
    // Write the average fit and its error to an output file

    char fitFileName_3_13[256];

    sprintf( fitFileName_3_13, "out/gTFit_dt%d_%d_%d_%dConfigs_binSize%d.dat",
	     TsinkPlusOne - 1, 3, 13, configNum, binSize );

    writeFitFile( fitFileName_3_13, gT_fitAvg_3_13, gT_fitErr_3_13, 3, 13, TsinkPlusOne - 1 );


    break;
  }
    
  }

  return 0;
}
