/* Reads a given data files containing three-point functions of a zero momentum transfer process projected to each momentum direction with
   the imagenary part of three-point function in the 6th column out of 6 and a given data file of jacknifed two-point functions. From these
   values, the program calculates the axial charge at each timestep, fits the axial charge vs. timestep data to a constant value, and
   calculates the error of the fit, outputing the axial charge vs. timestep to one file for plotting, and the fit and error to another file.

   Input arguments:

   1. Data file containing three-point functions in the 6th of 6 columns projected to the x direction for multiple repeating configurations
   2. Data file "                                                                       " y direction "                                   "
   3. Data file "                                                                       " z direction "                                   "
   4. Data file containing two-point functions in the 5th of 12 columns for the same number of repeating configurations and 128 timesteps
   5. Bin size
   6. (Optional) 1 if a printout of values stored in matrices is wanted
*/

#include "jk.h"
#include "fitting.h"
#include <cstdlib>

using namespace std;

int main ( int argc, char *argv[] ) {

  double zA = 0.791; // Axial current renormalization factor

  
  //////////////////////
  // Read user inputs //
  //////////////////////

  
  // Check that all required arguments have been included
  
  if ( argc < 6 ) {

    cout << "Missing input arguments" << endl;
    cout << "Include:" << endl;
    cout << "1. P4 three-point function data file" << endl;
    cout << "2. P5 three-point function data file" << endl;
    cout << "3. P6 three-point function data file" << endl;
    cout << "4. two-point function data file" << endl;
    cout << "5. bin size" << endl;
    cout << "6. (Optional) 1 for printout of data" << endl;

    return 1;
  }

  // If a print argument has been given, set Print (Print = 1 to print)
  
  int Print;
  
  if ( argc == 7 ) {

    Print = atoi( argv[6] );

  }

  // Set Tsink and number of configurations based on input files
   
  int TsinkPlusOne; // Because of t=0, this will be Tsink+1
  int TsinkPlusOne_x = detTimestepNum( argv[1], 7 );          
  int TsinkPlusOne_y = detTimestepNum( argv[2], 7 );         
  int TsinkPlusOne_z = detTimestepNum( argv[3], 7 );         
  int timestepNum_2pt = detTimestepNum( argv[4], 12 ); 

  // detTimestepNum is a function in "jk.h" which reads a file of repeating
  // configurations with the timesteps in the first column out of the given
  // number of columns and determines the number of timesteps in the file before
  // a new configuration starts and the timesteps repeat
  
  // Print the number of timesteps in each file
  
  if ( Print == 1 ) {

    cout << "x Tsink+1: " << TsinkPlusOne_x << endl;
    cout << "y Tsink+1: " << TsinkPlusOne_y << endl;
    cout << "z Tsink+1: " << TsinkPlusOne_z << endl;  
    cout << "Number of 2-point timesteps: " << timestepNum_2pt << endl;
    cout << endl;
    
  }
  
  int configNum;
  int configNum_x;
  int configNum_y;
  int configNum_z;

  int binSize;

  // Check that all 3 files have the same Tsink
  
  if ( TsinkPlusOne_x == TsinkPlusOne_y && TsinkPlusOne_x == TsinkPlusOne_z ) {

    TsinkPlusOne = TsinkPlusOne_x;
    
    configNum_x = detConfigNum( argv[1], 7 );
    configNum_y = detConfigNum( argv[2], 7 );
    configNum_z = detConfigNum( argv[3], 7 );

    // detConfigNum is a function in "jk.h" which reads a file of repeating
    // configurations with a given number of timesteps and columns and counts
    // the total number of configurations in the file

  }
  else {

    cout << "ERROR: Number of timesteps in each file do not match" << endl;
    return 1;

  }

  // Print the number of configurations in each file
  
  if ( Print == 1 ) {

    cout << "Number of configurations in x projection: " << configNum_x << endl;
    cout << "Number of configurations in y projection: " << configNum_y << endl;
    cout << "Number of configurations in z projection: " << configNum_z << endl;  
    cout << endl;
    
  }

  // Check that all 3 files have the same number of configurations

  if ( configNum_x == configNum_y && configNum_x == configNum_z ) {

    configNum = configNum_x;

    // Set binSize based on user input
    
    binSize = atoi( argv[5] );

  }
  else {

    cout << "ERROR: Number of configurations in each file do not match" << endl;
    return 1;

  }
  
  // Check that configNum is exactly divided by binSize

  while( configNum % binSize != 0 ) {
    
    cout << "WARNING: configNum " << configNum << " is divided by binSize " << binSize << " with a remainder. ";
    cout << "Please enter a new binSize." << endl;

    cin >> binSize;

  }

  int binNum = configNum / binSize;

  
  /////////////////////////////////////
  // Initialize matrices and vectors //
  /////////////////////////////////////


  // Matrix of three-point functions projected to the x direction (P4)
  // ( thrPtFuncs_x[t][c] )

  vector< vector<double> > thrPtFuncs_x(TsinkPlusOne); // matrix w/ 'TsinkPlusOne' rows

  giveMatrixCols( &thrPtFuncs_x, configNum ); // give matrix 'configNum' columns

  // giveMatrixCols is a function in "jk.h" which fills each row of a
  // matrix with a given number of columns

    
  // Matrix of three-point functions projected to the y direction (P5)
  // ( thrPtFuncs_y[t][c] )

  vector< vector<double> > thrPtFuncs_y(TsinkPlusOne); // matrix w/ 'TsinkPlusOne' rows

  giveMatrixCols( &thrPtFuncs_y, configNum ); // Give matrix 'configNum' columns
    
  // Matrix of three-point functions projected to the z direction (P6)
  // ( thrPtFuncs_z[t][c]

  vector< vector<double> > thrPtFuncs_z(TsinkPlusOne); // matrix w/ 'TsinkPlusOne' rows

  giveMatrixCols( &thrPtFuncs_z, configNum ); // Give matrix 'configNum' columns
  
  // Matrix of three point functions averaged over momentum directions
  // ( thrPtFuncs_avg[t][c] )

  vector< vector<double> > thrPtFuncs_avg(TsinkPlusOne); // matrix w/ 'TsinkPlusOne' rows
  
  giveMatrixCols( &thrPtFuncs_avg, configNum ); // Give matrix 'configNum' columns

  // Matrix of resampled three point functions
  // ( thrPtFuncs_jk[t][b] )
  
  vector< vector<double> > thrPtFuncs_jk(TsinkPlusOne); // matrix w/ 'TsinkPlusOne' rows

  giveMatrixCols( &thrPtFuncs_jk, binNum ); // Give matrix 'binNum' columns

  // Vector of two-point functions at Tsink
  // ( twoPtFuncs[c] )
  
  vector<double> twoPtFuncs(configNum);

  // Vector of resampled two-point functions
  // ( twoPtFuncs_jk[b] )
  
  vector<double> twoPtFuncs_jk(binNum); //vector w/ 'binNum' rows

  // Matrix of axial charges
  // ( axialCharge[t][b] )
  
  vector< vector<double> > axialCharge(TsinkPlusOne); // matrix w/ 'TsinkPlusOne' rows

  giveMatrixCols( &axialCharge, binNum ); // Give matrix 'binNum' columns
  
  // Vector of axial charges averaged over bins
  // ( axialCharge_[t] )
  
  vector<double> axialCharge_avg(TsinkPlusOne);

  // Vector of axial charge errors
  // ( axialCharge_err[t] )
  
  vector<double> axialCharge_err(TsinkPlusOne);


  ///////////////////////////////////////////////
  // Fill matrices with values from data files //
  ///////////////////////////////////////////////


  // Three-point functions projected to x direction
  
  readNthDataCol( &thrPtFuncs_x, argv[1], 6, 7 );

  // readNthDataCol is a function in "jk.h" which reads a file and stores
  // the numbers in the first given numbered column and stores them in a
  // matrix

  
  // Print matrix of three-point functions

  if ( Print == 1 ) {

    printMatrix( &thrPtFuncs_x, "Three-point functions projected to x direction:" );

    // printMatrix is a function in "jk.h"
    
  }

  // Three-point functions projected to y direction

  readNthDataCol( &thrPtFuncs_y, argv[2], 6, 7 );

  // Print matrix of three-point functions

  if ( Print == 1 ) {

    printMatrix( &thrPtFuncs_y, "Three-point functions projected to y direction:" );

  }

  // Three-point functions project to z direction

  readNthDataCol( &thrPtFuncs_z, argv[3], 6, 7 );

  // Print matrix of three-point functions

  if ( Print == 1 ) {

    printMatrix( &thrPtFuncs_z, "Three-point functions projected to z direction:" );

  }

  // Vector of two-point functions

  readNthDataRow ( &twoPtFuncs, argv[4], timestepNum_2pt - 1, TsinkPlusOne - 1, 5, 12 );

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

      thrPtFuncs_avg[t][c] = ( thrPtFuncs_x[t][c] + thrPtFuncs_y[t][c] + thrPtFuncs_z[t][c] ) / 3;

    }
  }

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
  

  ////////////////////////////
  // Calculate axial charge //
  ////////////////////////////
  

  for ( int t = 0; t < TsinkPlusOne; t++ ) { // Loop over timesteps

    for ( int b = 0; b < binNum; b++ ) { // Loop over bins

      axialCharge[t][b] = -2 * zA * thrPtFuncs_jk[t][b] / twoPtFuncs_jk[b];

    }
  }

  // Print matrix of axial charges

  if ( Print == 1 ) {

    printMatrix( &axialCharge, "Axial charge:" );

  }


  ////////////////////////////////////
  // Average axial charge over bins //
  ////////////////////////////////////
  

  averageRows( &axialCharge_avg, &axialCharge_err, &axialCharge );

  // averageRows is a function in "jk.h" which averages the values in each row of
  // a matrix resulting in a vector with the same dimension as the number of rows
  // in the original matrix

  // Print averaged axial charges

  if ( Print == 1 ) {

    printVector( &axialCharge_avg, "Axial charge averaged over bins:" );
    
    printVector( &axialCharge_err, "Axial charge error:" );
    
  }
  

  /////////////////////////////////////////////////////////////////
  // Write output file containing axial charge vs. timestep data //
  /////////////////////////////////////////////////////////////////

  
  char acFileName[256];

  // Set file name to include the proper delta t value
  
  sprintf( acFileName, "out/axialCharge_dt%d.dat", TsinkPlusOne - 1 );

  writeVectorFile( acFileName, &axialCharge_avg, &axialCharge_err );

  // writeVectorFile is a function in "jk.h" which writes a file with each
  // timestep in the first column of the file, the values in the vector given as
  // the second argument in the second column of the file, and the values in the
  // vector given as the third argument in the third column of the file


  //////////////////////////////////////////////////////////////////////////////
  // Fit the axial charge to a constant parameter for each bin, calculate the //
  // average fit and its error, and write them to an output file              //
  //////////////////////////////////////////////////////////////////////////////
  

  switch( TsinkPlusOne ) { // delta t determines which timeslices to include in the fit

  case 13 : {

    // t=4 to t=8

    // Fit each bin
    
    vector<double> axialCharge_fit_4_8( binNum );

    fitData( &axialCharge_fit_4_8, &axialCharge, &axialCharge_err, 4, 8 );

    // Calculate the average fit and it error

    double axialCharge_fitAvg_4_8;

    double axialCharge_fitErr_4_8;

    averageFit( &axialCharge_fitAvg_4_8, &axialCharge_fitErr_4_8, &axialCharge_fit_4_8 );

    // Print the axial charge fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &axialCharge_fit_4_8, "Axial charge fits (t=4 to t=8):" );

      cout << "Average axial charge fit: " << axialCharge_fitAvg_4_8 << endl;

      cout << endl;

      cout << "Axial charge fit error: " << axialCharge_fitErr_4_8 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_4_8[] = "out/axialChargeFit_dt12_4_8.dat";

    writeFitFile( fitFileName_4_8, axialCharge_fitAvg_4_8, axialCharge_fitErr_4_8, 4, 8, TsinkPlusOne - 1 );

    // t=3 to t=9

    // Fit each bin
    
    vector<double> axialCharge_fit_3_9( binNum );

    fitData( &axialCharge_fit_3_9, &axialCharge, &axialCharge_err, 3, 9 );

    // Calculate the average fit and it error

    double axialCharge_fitAvg_3_9;

    double axialCharge_fitErr_3_9;

    averageFit( &axialCharge_fitAvg_3_9, &axialCharge_fitErr_3_9, &axialCharge_fit_3_9 );

    // Print the axial charge fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &axialCharge_fit_3_9, "Axial charge fits (t=3 to t=9):" );

      cout << "Average axial charge fit: " << axialCharge_fitAvg_3_9 << endl;

      cout << endl;

      cout << "Axial charge fit error: " << axialCharge_fitErr_3_9 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_3_9[] = "out/axialChargeFit_dt12_3_9.dat";

    writeFitFile( fitFileName_3_9, axialCharge_fitAvg_3_9, axialCharge_fitErr_3_9, 3, 9, TsinkPlusOne - 1 );


    // t=2 to t=10

    // Fit each bin
    
    vector<double> axialCharge_fit_2_10( binNum );

    fitData( &axialCharge_fit_2_10, &axialCharge, &axialCharge_err, 2, 10 );

    // Calculate the average fit and it error

    double axialCharge_fitAvg_2_10;

    double axialCharge_fitErr_2_10;

    averageFit( &axialCharge_fitAvg_2_10, &axialCharge_fitErr_2_10, &axialCharge_fit_2_10 );

    // Print the axial charge fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &axialCharge_fit_2_10, "Axial charge fits (t=2 to t=10):" );

      cout << "Average axial charge fit: " << axialCharge_fitAvg_2_10 << endl;

      cout << endl;

      cout << "Axial charge fit error: " << axialCharge_fitErr_2_10 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_2_10[] = "out/axialChargeFit_dt12_2_10.dat";

    writeFitFile( fitFileName_2_10, axialCharge_fitAvg_2_10, axialCharge_fitErr_2_10, 2, 10, TsinkPlusOne - 1 );

        
    break;
  }

  case 15 : {

    // t=5 to t=9

    // Fit each bin
    
    vector<double> axialCharge_fit_5_9( binNum );

    fitData( &axialCharge_fit_5_9, &axialCharge, &axialCharge_err, 5, 9 );

    // Calculate the average fit and it error

    double axialCharge_fitAvg_5_9;

    double axialCharge_fitErr_5_9;

    averageFit( &axialCharge_fitAvg_5_9, &axialCharge_fitErr_5_9, &axialCharge_fit_5_9 );

    // Print the axial charge fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &axialCharge_fit_5_9, "Axial charge fits (t=5 to t=9):" );

      cout << "Average axial charge fit: " << axialCharge_fitAvg_5_9 << endl;

      cout << endl;

      cout << "Axial charge fit error: " << axialCharge_fitErr_5_9 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_5_9[] = "out/axialChargeFit_dt14_5_9.dat";

    writeFitFile( fitFileName_5_9, axialCharge_fitAvg_5_9, axialCharge_fitErr_5_9, 5, 9, TsinkPlusOne - 1 );

        
    // t=4 to t=10

    // Fit each bin

    vector<double> axialCharge_fit_4_10( binNum );
   
    fitData( &axialCharge_fit_4_10, &axialCharge, &axialCharge_err, 4, 10 );

    // Calculate the average fit and it error

    double axialCharge_fitAvg_4_10;

    double axialCharge_fitErr_4_10;

    averageFit( &axialCharge_fitAvg_4_10, &axialCharge_fitErr_4_10, &axialCharge_fit_4_10 );

    // Print the axial charge fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &axialCharge_fit_4_10, "Axial charge fits (t=4 to t=10):" );
      
      cout << "Average axial charge fit: " << axialCharge_fitAvg_4_10 << endl;

      cout << endl;

      cout << "Axial charge fit error: " << axialCharge_fitErr_4_10 << endl;
      
      cout << endl;
      
    }
    
    // Write the average fit and its error to an output file

    char fitFileName_4_10[] = "out/axialChargeFit_dt14_4_10.dat";

    writeFitFile( fitFileName_4_10, axialCharge_fitAvg_4_10, axialCharge_fitErr_4_10, 4, 10, TsinkPlusOne - 1 );
    
    // t=3 to t=11

    // Fit each bin
    
    vector<double> axialCharge_fit_3_11( binNum );

    fitData( &axialCharge_fit_3_11, &axialCharge, &axialCharge_err, 3, 11 );

    // Calculate the average fit and it error

    double axialCharge_fitAvg_3_11;

    double axialCharge_fitErr_3_11;

    averageFit( &axialCharge_fitAvg_3_11, &axialCharge_fitErr_3_11, &axialCharge_fit_3_11 );

    // Print the axial charge fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &axialCharge_fit_3_11, "Axial charge fits (t=3 to t=11):" );

      cout << "Average axial charge fit: " << axialCharge_fitAvg_3_11 << endl;

      cout << endl;

      cout << "Axial charge fit error: " << axialCharge_fitErr_3_11 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_3_11[] = "out/axialChargeFit_dt14_3_11.dat";

    writeFitFile( fitFileName_3_11, axialCharge_fitAvg_3_11, axialCharge_fitErr_3_11, 3, 11, TsinkPlusOne - 1 );
    
    
    // t=2 to t=12

    // Fit each bin
    
    vector<double> axialCharge_fit_2_12( binNum );

    fitData( &axialCharge_fit_2_12, &axialCharge, &axialCharge_err, 2, 12 );

    // Calculate the average fit and it error

    double axialCharge_fitAvg_2_12;

    double axialCharge_fitErr_2_12;

    averageFit( &axialCharge_fitAvg_2_12, &axialCharge_fitErr_2_12, &axialCharge_fit_2_12 );

    // Print the axial charge fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &axialCharge_fit_2_12, "Axial charge fits (t=2 to t=12):" );

      cout << "Average axial charge fit: " << axialCharge_fitAvg_2_12 << endl;

      cout << endl;

      cout << "Axial charge fit error: " << axialCharge_fitErr_2_12 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_2_12[] = "out/axialChargeFit_dt14_2_12.dat";

    writeFitFile( fitFileName_2_12, axialCharge_fitAvg_2_12, axialCharge_fitErr_2_12, 2, 12, TsinkPlusOne - 1 );
    
    
    break;
  }
    
  case 17 : {
    
    // t=6 to t=10

    // Fit each bin
    
    vector<double> axialCharge_fit_6_10( binNum );
    
    fitData( &axialCharge_fit_6_10, &axialCharge, &axialCharge_err, 6, 10 );

    // Calculate the average fit and its error

    double axialCharge_fitAvg_6_10;

    double axialCharge_fitErr_6_10;

    averageFit( &axialCharge_fitAvg_6_10, &axialCharge_fitErr_6_10, &axialCharge_fit_6_10 );

    // Print the axial charge fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &axialCharge_fit_6_10, "Axial charge fits (t=6 to t=10):" );
      
      cout << "Average axial charge fit: " << axialCharge_fitAvg_6_10 << endl;

      cout << endl;

      cout << "Axial charge fit error: " << axialCharge_fitErr_6_10 << endl;
      
      cout << endl;
      
    }

    
    // Write the average fit and its error to an output file

    char fitFileName_6_10[] = "out/axialChargeFit_dt16_6_10.dat";

    writeFitFile( fitFileName_6_10, axialCharge_fitAvg_6_10, axialCharge_fitErr_6_10, 6, 10, TsinkPlusOne - 1 );    

    // t=5 to t=11

    // Fit each bin
    
    vector<double> axialCharge_fit_5_11( binNum );
    
    fitData( &axialCharge_fit_5_11, &axialCharge, &axialCharge_err, 5, 11 );

    // Calculate the average fit and its error

    double axialCharge_fitAvg_5_11;

    double axialCharge_fitErr_5_11;

    averageFit( &axialCharge_fitAvg_5_11, &axialCharge_fitErr_5_11, &axialCharge_fit_5_11 );
    
    // Print the axial charge fits for each bin, the averaged fit, and its error
    
    if ( Print == 1 ) {

      printVector( &axialCharge_fit_5_11, "Axial charge fits (t=5 to t=11):" );
      
      cout << "Average axial charge fit: " << axialCharge_fitAvg_5_11 << endl;

      cout << endl;

      cout << "Axial charge fit error: " << axialCharge_fitErr_5_11 << endl;
      
      cout << endl;
      
    }
    
    // Write the average fit and its error to an output file

    char fitFileName_5_11[] = "out/axialChargeFit_dt16_5_11.dat";

    writeFitFile( fitFileName_5_11, axialCharge_fitAvg_5_11, axialCharge_fitErr_5_11, 5, 11, TsinkPlusOne - 1 );


    // t=4 to t=12

    // Fit each bin
    
    vector<double> axialCharge_fit_4_12( binNum );
    
    fitData( &axialCharge_fit_4_12, &axialCharge, &axialCharge_err, 4, 12 );

    // Calculate the average fit and its error

    double axialCharge_fitAvg_4_12;

    double axialCharge_fitErr_4_12;

    averageFit( &axialCharge_fitAvg_4_12, &axialCharge_fitErr_4_12, &axialCharge_fit_4_12 );
    
    // Print the axial charge fits for each bin, the averaged fit, and its error
    
    if ( Print == 1 ) {

      printVector( &axialCharge_fit_4_12, "Axial charge fits (t=4 to t=12):" );
      
      cout << "Average axial charge fit: " << axialCharge_fitAvg_4_12 << endl;

      cout << endl;

      cout << "Axial charge fit error: " << axialCharge_fitErr_4_12 << endl;
      
      cout << endl;
      
    }
    
    // Write the average fit and its error to an output file

    char fitFileName_4_12[] = "out/axialChargeFit_dt16_4_12.dat";

    writeFitFile( fitFileName_4_12, axialCharge_fitAvg_4_12, axialCharge_fitErr_4_12, 4, 12, TsinkPlusOne - 1 );


    // t=3 to t=13

    // Fit each bin
    
    vector<double> axialCharge_fit_3_13( binNum );
    
    fitData( &axialCharge_fit_3_13, &axialCharge, &axialCharge_err, 3, 13 );

    // Calculate the average fit and its error

    double axialCharge_fitAvg_3_13;

    double axialCharge_fitErr_3_13;

    averageFit( &axialCharge_fitAvg_3_13, &axialCharge_fitErr_3_13, &axialCharge_fit_3_13 );
    
    // Print the axial charge fits for each bin, the averaged fit, and its error
    
    if ( Print == 1 ) {

      printVector( &axialCharge_fit_3_13, "Axial charge fits (t=3 to t=13):" );
      
      cout << "Average axial charge fit: " << axialCharge_fitAvg_3_13 << endl;

      cout << endl;

      cout << "Axial charge fit error: " << axialCharge_fitErr_3_13 << endl;
      
      cout << endl;
      
    }
    
    // Write the average fit and its error to an output file

    char fitFileName_3_13[] = "out/axialChargeFit_dt16_3_13.dat";

    writeFitFile( fitFileName_3_13, axialCharge_fitAvg_3_13, axialCharge_fitErr_3_13, 3, 13, TsinkPlusOne - 1 );


    break;
  }
    
  }

  return 0;
}
