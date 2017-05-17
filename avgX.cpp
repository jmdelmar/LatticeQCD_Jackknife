/* Reads a given data files containing three-point functions of a zero momentum transfer process projected to each momentum direction with
   the imagenary part of three-point function in the 6th column out of 6 and a given data file of jacknifed two-point functions. From these
   values, the program calculates <x> at each timestep, fits <x> vs. timestep data to a constant value, and
   calculates the error of the fit, outputing <x> vs. timestep to one file for plotting, and the fit and error to another file.

   Input arguments:

   1. File name of three-point
   2. File name of two-point functions
   3. File name of effective mass
   4. File name of effective mass errors
   5. First timeslice to include in effective mass fit
   6. Last timeslice to include in effective mass fit
   7. (Optional) 1 if a printout of values stored in matrices is wanted
*/

#include "jk.h"
#include "fitting.h"
#include <cstdlib>

using namespace std;

int main ( int argc, char *argv[] ) {

  double Z = 1.1251; // <x> renormalization factor

  
  //////////////////////
  // Read user inputs //
  //////////////////////

  
  // Check that all required arguments have been included
  
  if ( argc < 7 ) {

    cout << "Missing input arguments" << endl;
    cout << "Include:" << endl;
    cout << "1. three-point function data file" << endl;
    cout << "2. two-point function data file" << endl;
    cout << "3. effective mass data file" << endl;
    cout << "4. effective mass error data file" << endl;
    cout << "5. First timeslice to include in fit" << endl;
    cout << "6. Last timeslice to include in fit" << endl;    
    cout << "7. (Optional) 1 for printout of data" << endl;

    return 1;
  }

  // If a print argument has been given, set Print (Print = 1 to print)
  
  int Print;
  
  if ( argc == 8 ) {

    Print = atoi( argv[7] );

  }

  // Set Tsink and number of configurations based on input files
   
  int TsinkPlusOne = detTimestepNum( argv[1], 8 ); // Because of t=0, this will 
                                                   //be Tsink+1
  int timestepNum_2pt = detTimestepNum( argv[2], 12 ); 
  int timestepNum_effMass = detTimestepNum( argv[3], 2 );
  int timestepNum;

  // detTimestepNum is a function in "jk.h" which reads a file of repeating
  // configurations with the timesteps in the first column out of the given
  // number of columns and determines the number of timesteps in the file before
  // a new configuration starts and the timesteps repeat
  
  // Print the number of timesteps in each file
  
  if ( Print == 1 ) {

    cout << "Tsink+1: " << TsinkPlusOne << endl;
    cout << "Number of 2-point timesteps: " << timestepNum_2pt << endl;

    cout << endl << "Number of timesteps for effective masses: ";
    cout << timestepNum_effMass << endl;
    cout << endl;
    
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
  
  int configNum;
  int configNum_3pt = detConfigNum( argv[1], 8 );
  int configNum_2pt = detConfigNum( argv[2], 12 );

  // detConfigNum is a function in "jk.h" which reads a file of repeating
  // configurations with a given number of timesteps and columns and counts
  // the total number of configurations in the file


  // Print the number of configurations in each file
  
  if ( Print == 1 ) {

    cout << "Number of 3-point configurations: " << configNum_3pt << endl;
    cout << "Number of 2-point configurations: " << configNum_2pt << endl;
    cout << endl;
    
  }

  // Check that both files have the same number of configurations

  if ( configNum_3pt == configNum_2pt ) {

    configNum = configNum_2pt;

  }
  else {

    cout << "ERROR: Number of configurations in each file do not match" << endl;
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

  int fitStart = atoi( argv[5] );

  int fitEnd = atoi( argv[6] );

  
  /////////////////////////////////////
  // Initialize matrices and vectors //
  /////////////////////////////////////


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


  // Matrix of three-point functions
  // ( thrPtFuncs[t][c] )

  vector< vector<double> > thrPtFuncs(TsinkPlusOne); // matrix w/ 'TsinkPlusOne' rows

  giveMatrixCols( &thrPtFuncs, configNum ); // give matrix 'configNum' columns

  // giveMatrixCols is a function in "jk.h" which fills each row of a
  // matrix with a given number of columns

    
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

  // Matrix of <x>
  // ( avgX[t][b] )
  
  vector< vector<double> > avgX(TsinkPlusOne); // matrix w/ 'TsinkPlusOne' rows

  giveMatrixCols( &avgX, binNum ); // Give matrix 'binNum' columns
  
  // Vector of <x> averaged over bins
  // ( avgX_avg[t] )
  
  vector<double> avgX_avg(TsinkPlusOne);

  // Vector of <x> errors
  // ( avgX_err[t] )
  
  vector<double> avgX_err(TsinkPlusOne);


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


  ///////////////////////////////////////////////
  // Fill matrices with values from data files //
  ///////////////////////////////////////////////


  // Three-point functions
  
  readNthDataCol( &thrPtFuncs, argv[1], 5, 8 );

  // readNthDataCol is a function in "jk.h" which reads a file and stores
  // the numbers in the first given numbered column and stores them in a
  // matrix

  
  // Print matrix of three-point functions

  if ( Print == 1 ) {

    printMatrix( &thrPtFuncs, "Three-point functions:" );

    // printMatrix is a function in "jk.h"
    
  }

  // Vector of two-point functions

  readNthDataRow ( &twoPtFuncs, argv[2], timestepNum_2pt - 1, TsinkPlusOne - 1, 5, 12 );

  // Print two-point functions

  if ( Print == 1 ) {
        
    printVector( &twoPtFuncs, "Two-point functions:" );

    // printVector is a function in "jk.h"

  }


  ///////////////
  // Jackknife //
  ///////////////
  
  
  // Jackknife three-point functions

  jackknifeMatrix ( &thrPtFuncs_jk, &thrPtFuncs );

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
  

  ///////////////////
  // Calculate <x> //
  ///////////////////
  

  for ( int t = 0; t < TsinkPlusOne; t++ ) { // Loop over timesteps

    for ( int b = 0; b < binNum; b++ ) { // Loop over bins

      avgX[t][b] = -4/3 * Z / fitMass[b] * thrPtFuncs_jk[t][b] / twoPtFuncs_jk[b];

    }
  }

  // Print matrix of <x>

  if ( Print == 1 ) {

    printMatrix( &avgX, "<x>:" );

  }


  ///////////////////////////
  // Average <x> over bins //
  ///////////////////////////
  

  averageRows( &avgX_avg, &avgX_err, &avgX );

  // averageRows is a function in "jk.h" which averages the values in each row of
  // a matrix resulting in a vector with the same dimension as the number of rows
  // in the original matrix

  // Print averaged <x>

  if ( Print == 1 ) {

    printVector( &avgX_avg, "<x> averaged over bins:" );
    
    printVector( &avgX_err, "<x> error:" );
    
  }
  

  ///////////////////////////////////////
  // Write effective mass output files //
  ///////////////////////////////////////

  
  // Average fitted mass and its error

  char avgMassFitFilename[] = "out/avgFittedMass.dat";

  writeFitFile( avgMassFitFilename, fitMass_avg, fitMass_err, fitStart, fitEnd, TsinkPlusOne - 1 );

  // writeFitFile is a function in "jk.h"


  char acFileName[256];

  // Set file name to include the proper delta t value
  
  sprintf( acFileName, "out/avgX_dt%d.dat", TsinkPlusOne - 1 );

  writeVectorFile( acFileName, &avgX_avg, &avgX_err );

  // writeVectorFile is a function in "jk.h" which writes a file with each
  // timestep in the first column of the file, the values in the vector given as
  // the second argument in the second column of the file, and the values in the
  // vector given as the third argument in the third column of the file


  /////////////////////////////////////////////////////////////////
  // Fit <x> to a constant parameter for each bin, calculate the //
  // average fit and its error, and write them to an output file //
  /////////////////////////////////////////////////////////////////
  

  switch( TsinkPlusOne ) { // delta t determines which timeslices to include in the fit

  case 13 : {

    // t=4 to t=8

    // Fit each bin
    
    vector<double> avgX_fit_4_8( binNum );

    fitData( &avgX_fit_4_8, &avgX, &avgX_err, 4, 8 );

    // Calculate the average fit and it error

    double avgX_fitAvg_4_8;

    double avgX_fitErr_4_8;

    averageFit( &avgX_fitAvg_4_8, &avgX_fitErr_4_8, &avgX_fit_4_8 );

    // Print <x> fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &avgX_fit_4_8, "<x> fits (t=4 to t=8):" );

      cout << "Average <x> fit: " << avgX_fitAvg_4_8 << endl;

      cout << endl;

      cout << "<x> fit error: " << avgX_fitErr_4_8 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_4_8[] = "out/avgXFit_dt12_4_8.dat";

    writeFitFile( fitFileName_4_8, avgX_fitAvg_4_8, avgX_fitErr_4_8, 4, 8, TsinkPlusOne - 1 );

    // t=3 to t=9

    // Fit each bin
    
    vector<double> avgX_fit_3_9( binNum );

    fitData( &avgX_fit_3_9, &avgX, &avgX_err, 3, 9 );

    // Calculate the average fit and it error

    double avgX_fitAvg_3_9;

    double avgX_fitErr_3_9;

    averageFit( &avgX_fitAvg_3_9, &avgX_fitErr_3_9, &avgX_fit_3_9 );

    // Print <x> fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &avgX_fit_3_9, "<x> fits (t=3 to t=9):" );

      cout << "Average <x> fit: " << avgX_fitAvg_3_9 << endl;

      cout << endl;

      cout << "<x> fit error: " << avgX_fitErr_3_9 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_3_9[] = "out/avgXFit_dt12_3_9.dat";

    writeFitFile( fitFileName_3_9, avgX_fitAvg_3_9, avgX_fitErr_3_9, 3, 9, TsinkPlusOne - 1 );


    // t=2 to t=10

    // Fit each bin
    
    vector<double> avgX_fit_2_10( binNum );

    fitData( &avgX_fit_2_10, &avgX, &avgX_err, 2, 10 );

    // Calculate the average fit and it error

    double avgX_fitAvg_2_10;

    double avgX_fitErr_2_10;

    averageFit( &avgX_fitAvg_2_10, &avgX_fitErr_2_10, &avgX_fit_2_10 );

    // Print <x> fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &avgX_fit_2_10, "<x> fits (t=2 to t=10):" );

      cout << "Average <x> fit: " << avgX_fitAvg_2_10 << endl;

      cout << endl;

      cout << "<x> fit error: " << avgX_fitErr_2_10 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_2_10[] = "out/avgXFit_dt12_2_10.dat";

    writeFitFile( fitFileName_2_10, avgX_fitAvg_2_10, avgX_fitErr_2_10, 2, 10, TsinkPlusOne - 1 );

        
    break;
  }

  case 15 : {

    // t=5 to t=9

    // Fit each bin
    
    vector<double> avgX_fit_5_9( binNum );

    fitData( &avgX_fit_5_9, &avgX, &avgX_err, 5, 9 );

    // Calculate the average fit and it error

    double avgX_fitAvg_5_9;

    double avgX_fitErr_5_9;

    averageFit( &avgX_fitAvg_5_9, &avgX_fitErr_5_9, &avgX_fit_5_9 );

    // Print <x> fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &avgX_fit_5_9, "<x> fits (t=5 to t=9):" );

      cout << "Average <x> fit: " << avgX_fitAvg_5_9 << endl;

      cout << endl;

      cout << "<x> fit error: " << avgX_fitErr_5_9 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_5_9[] = "out/avgXFit_dt14_5_9.dat";

    writeFitFile( fitFileName_5_9, avgX_fitAvg_5_9, avgX_fitErr_5_9, 5, 9, TsinkPlusOne - 1 );

        
    // t=4 to t=10

    // Fit each bin

    vector<double> avgX_fit_4_10( binNum );
   
    fitData( &avgX_fit_4_10, &avgX, &avgX_err, 4, 10 );

    // Calculate the average fit and it error

    double avgX_fitAvg_4_10;

    double avgX_fitErr_4_10;

    averageFit( &avgX_fitAvg_4_10, &avgX_fitErr_4_10, &avgX_fit_4_10 );

    // Print <x> fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &avgX_fit_4_10, "<x> fits (t=4 to t=10):" );
      
      cout << "Average <x> fit: " << avgX_fitAvg_4_10 << endl;

      cout << endl;

      cout << "<x> fit error: " << avgX_fitErr_4_10 << endl;
      
      cout << endl;
      
    }
    
    // Write the average fit and its error to an output file

    char fitFileName_4_10[] = "out/avgXFit_dt14_4_10.dat";

    writeFitFile( fitFileName_4_10, avgX_fitAvg_4_10, avgX_fitErr_4_10, 4, 10, TsinkPlusOne - 1 );
    
    // t=3 to t=11

    // Fit each bin
    
    vector<double> avgX_fit_3_11( binNum );

    fitData( &avgX_fit_3_11, &avgX, &avgX_err, 3, 11 );

    // Calculate the average fit and it error

    double avgX_fitAvg_3_11;

    double avgX_fitErr_3_11;

    averageFit( &avgX_fitAvg_3_11, &avgX_fitErr_3_11, &avgX_fit_3_11 );

    // Print <x> fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &avgX_fit_3_11, "<x> fits (t=3 to t=11):" );

      cout << "Average <x> fit: " << avgX_fitAvg_3_11 << endl;

      cout << endl;

      cout << "<x> fit error: " << avgX_fitErr_3_11 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_3_11[] = "out/avgXFit_dt14_3_11.dat";

    writeFitFile( fitFileName_3_11, avgX_fitAvg_3_11, avgX_fitErr_3_11, 3, 11, TsinkPlusOne - 1 );
    
    
    // t=2 to t=12

    // Fit each bin
    
    vector<double> avgX_fit_2_12( binNum );

    fitData( &avgX_fit_2_12, &avgX, &avgX_err, 2, 12 );

    // Calculate the average fit and it error

    double avgX_fitAvg_2_12;

    double avgX_fitErr_2_12;

    averageFit( &avgX_fitAvg_2_12, &avgX_fitErr_2_12, &avgX_fit_2_12 );

    // Print <x> fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &avgX_fit_2_12, "<x> fits (t=2 to t=12):" );

      cout << "Average <x> fit: " << avgX_fitAvg_2_12 << endl;

      cout << endl;

      cout << "<x> fit error: " << avgX_fitErr_2_12 << endl;
      
      cout << endl;
      
    }
        
    // Write the average fit and its error to an output file

    char fitFileName_2_12[] = "out/avgXFit_dt14_2_12.dat";

    writeFitFile( fitFileName_2_12, avgX_fitAvg_2_12, avgX_fitErr_2_12, 2, 12, TsinkPlusOne - 1 );
    
    
    break;
  }
    
  case 17 : {
    
    // t=6 to t=10

    // Fit each bin
    
    vector<double> avgX_fit_6_10( binNum );
    
    fitData( &avgX_fit_6_10, &avgX, &avgX_err, 6, 10 );

    // Calculate the average fit and its error

    double avgX_fitAvg_6_10;

    double avgX_fitErr_6_10;

    averageFit( &avgX_fitAvg_6_10, &avgX_fitErr_6_10, &avgX_fit_6_10 );

    // Print <x> fits for each bin, the averaged fit, and its error

    if ( Print == 1 ) {

      printVector( &avgX_fit_6_10, "<x> fits (t=6 to t=10):" );
      
      cout << "Average <x> fit: " << avgX_fitAvg_6_10 << endl;

      cout << endl;

      cout << "<x> fit error: " << avgX_fitErr_6_10 << endl;
      
      cout << endl;
      
    }

    
    // Write the average fit and its error to an output file

    char fitFileName_6_10[] = "out/avgXFit_dt16_6_10.dat";

    writeFitFile( fitFileName_6_10, avgX_fitAvg_6_10, avgX_fitErr_6_10, 6, 10, TsinkPlusOne - 1 );    

    // t=5 to t=11

    // Fit each bin
    
    vector<double> avgX_fit_5_11( binNum );
    
    fitData( &avgX_fit_5_11, &avgX, &avgX_err, 5, 11 );

    // Calculate the average fit and its error

    double avgX_fitAvg_5_11;

    double avgX_fitErr_5_11;

    averageFit( &avgX_fitAvg_5_11, &avgX_fitErr_5_11, &avgX_fit_5_11 );
    
    // Print <x> fits for each bin, the averaged fit, and its error
    
    if ( Print == 1 ) {

      printVector( &avgX_fit_5_11, "<x> fits (t=5 to t=11):" );
      
      cout << "Average <x> fit: " << avgX_fitAvg_5_11 << endl;

      cout << endl;

      cout << "<x> fit error: " << avgX_fitErr_5_11 << endl;
      
      cout << endl;
      
    }
    
    // Write the average fit and its error to an output file

    char fitFileName_5_11[] = "out/avgXFit_dt16_5_11.dat";

    writeFitFile( fitFileName_5_11, avgX_fitAvg_5_11, avgX_fitErr_5_11, 5, 11, TsinkPlusOne - 1 );


    // t=4 to t=12

    // Fit each bin
    
    vector<double> avgX_fit_4_12( binNum );
    
    fitData( &avgX_fit_4_12, &avgX, &avgX_err, 4, 12 );

    // Calculate the average fit and its error

    double avgX_fitAvg_4_12;

    double avgX_fitErr_4_12;

    averageFit( &avgX_fitAvg_4_12, &avgX_fitErr_4_12, &avgX_fit_4_12 );
    
    // Print <x> fits for each bin, the averaged fit, and its error
    
    if ( Print == 1 ) {

      printVector( &avgX_fit_4_12, "<x> fits (t=4 to t=12):" );
      
      cout << "Average <x> fit: " << avgX_fitAvg_4_12 << endl;

      cout << endl;

      cout << "<x> fit error: " << avgX_fitErr_4_12 << endl;
      
      cout << endl;
      
    }
    
    // Write the average fit and its error to an output file

    char fitFileName_4_12[] = "out/avgXFit_dt16_4_12.dat";

    writeFitFile( fitFileName_4_12, avgX_fitAvg_4_12, avgX_fitErr_4_12, 4, 12, TsinkPlusOne - 1 );


    // t=3 to t=13

    // Fit each bin
    
    vector<double> avgX_fit_3_13( binNum );
    
    fitData( &avgX_fit_3_13, &avgX, &avgX_err, 3, 13 );

    // Calculate the average fit and its error

    double avgX_fitAvg_3_13;

    double avgX_fitErr_3_13;

    averageFit( &avgX_fitAvg_3_13, &avgX_fitErr_3_13, &avgX_fit_3_13 );
    
    // Print <x> fits for each bin, the averaged fit, and its error
    
    if ( Print == 1 ) {

      printVector( &avgX_fit_3_13, "<x> fits (t=3 to t=13):" );
      
      cout << "Average <x> fit: " << avgX_fitAvg_3_13 << endl;

      cout << endl;

      cout << "<x> fit error: " << avgX_fitErr_3_13 << endl;
      
      cout << endl;
      
    }
    
    // Write the average fit and its error to an output file

    char fitFileName_3_13[] = "out/avgXFit_dt16_3_13.dat";

    writeFitFile( fitFileName_3_13, avgX_fitAvg_3_13, avgX_fitErr_3_13, 3, 13, TsinkPlusOne - 1 );


    break;
  }
    
  }

  return 0;
}
