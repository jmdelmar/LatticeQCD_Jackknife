// Extracts the 5th column from a given data file and organizes it by timesteps 
// in rows and configurations in columns, calculates the JK averages for each 
// timestep and bin, the effective mass for each timestep and bin, the average 
// and error of the effective mass for each timestep, and writes a file 
// containing the average effective masses with their respective errors for each 
// timestep.

#include "jk.h"
#include "physQuants.h"
#include "readWrite.h"
#include "util.h"

extern char *twopDir;
extern char *sourceDir;
extern char *confList;
extern char *twopTemplate;
extern char *outputPre;
extern int binSize;
extern int timeDim;
extern bool Print;

using namespace lQCD_jk;
using std::cout;
using std::endl;

int main ( int argc, char *argv[] ) {

    
  //////////////////////
  // Read user inputs //
  //////////////////////


  if ( argc == 1 )
    usage( argv[0] );

  for (int i = 1; i < argc; i++){
    if( process_command_line_option(argc, argv, &i) == 0 ){
      continue;
    } 
    cout << "ERROR: Invalid option:" << argv[i] << endl;
    usage( argv[0] );
  }
  /*
  cout << "twopDir: " << twopDir << endl;
  cout << "sourceDir: " << sourceDir << endl;
  cout << "confList: " << confList << endl;
  cout << "twopTemplate: " << twopTemplate << endl;
  cout << "outputPre: " << outputPre << endl;
  cout << "binSize: " << binSize << endl;
  cout << "timeDim: " << timeDim << endl;

  return 0;
  */
  // Read configuration list and store configurations in a vector

  vector<string> confs;

  try {

    readStringFile( &confs, confList );

    // readStringFile() is a function in "jk.h"
  
  }
  catch( string *badFile ) {

    cout << "ERROR(readStringFile): file " << *badFile << " cannot be opened";
    cout << endl;

    return 1;

  }

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

  if( Print ) {

    cout << "Number of configurations: " << configNum << endl;

    cout << "Bin size: " << binSize << endl;

    cout << "Number of bins: " << binNum << endl;

  }

  // Read source files and store in matrix
  // srcPos[c][sP][sC]

  vector< vector< vector<int> > > srcPos( configNum );

  try {

    getSourcePositions( &srcPos, sourceDir, &confs );

    // getSourcePositions() is a function in "readWrite.h"

  }
  catch( string *badFile ) {

    cout << "ERROR(readStringFile): file " << *badFile << " cannot be opened";
    cout << endl;

    return 1;

  }

  // Check that there is the correct of number of sources

  for( int c=0; c<configNum; c++ ) { // Loop over configs
  
    int srcNum = 16;

    if( srcPos.at(c).size() != srcNum ) {

      cout << "WARNING: Configuration " << confs.at(c);
      cout << " has " << srcPos.at(c).size() << " sources ";
      cout << "(should be " << srcNum << ")." << endl;

    }
    
  } // End config loop


  /////////////////////////////////////
  // Initialize matrices and vectors //
  /////////////////////////////////////

	
  // Matrix of two point functions
  // ( twoPtFuncs[t][c][s] )
  
  vector< vector < vector<double> > > twoPtFuncs( timeDim ); // Tensor w/ 'timeDim' matrices

  // Matrix of two-point functions averaged over source positions
  // ( twoPtFuncs_srcAvg[t][c] )
  
  vector< vector<double> > twoPtFuncs_srcAvg( timeDim );  // Matrix w/ 'timeDim' rows

  giveMatrixCols( &twoPtFuncs_srcAvg, configNum );

  // giveMatrixCols() is a function in "jk.h"


  // Matrix of JK averaged TPFs
  // ( twoPtFuncs_jk[t][b] )
  
  vector< vector<double> > twoPtFuncs_jk( timeDim );  // Matrix w/ 'timeDim' rows

  giveMatrixCols( &twoPtFuncs_jk, binNum );

  // giveMatrixCols() is a function in "jk.h"


  // Matrix of effective masses
  // ( eMass[t][b] )

  vector< vector<double> > eMass( timeDim );  // Matrix w/ 'timeDim' rows

  giveMatrixCols( &eMass, binNum );


  // Vector of average two-point functions
  // ( twoPtFuncs_avg[t] )
  
  vector<double> twoPtFuncs_avg( timeDim );


  // Vector of two-point function errors
  // ( twoPtFuncs_err[t] )
  
  vector<double> twoPtFuncs_err( timeDim );


  // Vector of average effective masses
  // ( eMass_avg[t] )
  
  vector<double> eMass_avg( timeDim );


  // Vector of effective mass errors
  // ( eMass_err[t] )
  
  vector<double> eMass_err( timeDim );


  ///////////////////////////////////////////////
  // Fill matrices with values from data files //
  ///////////////////////////////////////////////


  readInfo rinfo;

  rinfo.timeDim = timeDim;
  rinfo.complex = 0; // Get real part
  rinfo.meson = 1; // Get twop for pion+

  readTwop_pseudoscalarMesons_Qsq0( &twoPtFuncs, twopDir, &confs, &srcPos, twopTemplate, rinfo);

  // readTwop_pseudoscalarMesons_Qsq0() is a function in "readWrite.h"

  cerr << "Flag" << endl;

  //Print matrix of two-point functions

  if( Print ) {

    printTensor( &twoPtFuncs, "Two-point functions:" );

    // printMatrix is a function in "jk.h"

  }


  //////////////////////////
  // Average over sources //
  //////////////////////////


  for( int t=0; t<timeDim; t++ ) {

    vector<double> stdDev; // Throwaway standard deviation to give to average function

    averageRows( &twoPtFuncs_srcAvg.at(t), &stdDev, &twoPtFuncs.at(t) );

    // averageRows() is a function in "jk.h"

  }


  ///////////////
  // Jackknife //
  ///////////////
  
	
  jackknifeMatrix( &twoPtFuncs_jk, &twoPtFuncs_srcAvg );

  // jackkifeMatrix is a function in "jk.h" which calculates the resampled
  // averages of values orginized in a matrix by different timesteps in each row
  // and different configurations in each column

  
  // Print matrix of resampled two-point functions

  if( Print ) {

    printMatrix( &twoPtFuncs_jk, "Resampled two-point functions:" );

  }
  

  ////////////////////////////////
  // Calculate effective masses //
  ////////////////////////////////


  effMass( &eMass, &twoPtFuncs_jk );

  // effMass is a function in "physQuants.h" which calculates the effective mass
  // from two-point functions
  
  
  // Print matrix of effective masses

  if ( Print ) {

    printMatrix( &eMass, "Effective masses:" );

  }
	
  //Calculate average effective mass and its errors

  averageRows( &eMass_avg, &eMass_err, &eMass );

  // averageRows is a function in "jk.h" which averages the values in each row of
  // a matrix resulting in a vector with the same dimension as the number of rows
  // in the original matrix

  // Print average effective masses and their respective errors

  if ( Print ) {

    printVector( &eMass_avg, "Averaged effective masses:" );

    printVector( &eMass_err, "Effective mass errors:" );

  }
  
  
  ////////////////////////
  // Write output files //
  ////////////////////////


  // Effective masses for each bin and timestep

  char emjkFileName[256];

  sprintf( emjkFileName, "%s_%dConfigs_binSize%d.dat", outputPre, configNum, binSize );

  writeMatrixFile( emjkFileName, &eMass );

  // writeMatrixFile is a function "jk.h" which writes a file containing a matrix
  // organized so that each row is a different timestep

  
  // Averaged effective masses and their errors
  
  char emFileName[256];

  sprintf( emFileName, "%s_avg_%dConfigs_binSize%d.dat", outputPre, configNum, binSize );

  writeVectorFile( emFileName, &eMass_avg, &eMass_err );
  
  // writeVectorFile is a function in "jk.h" which writes a file with each
  // timestep in the first column of the file, the values in the vector given as
  // the second argument in the second column of the file, and the values in the
  // vector given as the third argument in the third column of the file

  return 0;
	
}
