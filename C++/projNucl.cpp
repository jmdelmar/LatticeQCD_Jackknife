#include <cstdlib>
#include "jk.h"
#include "physQuants.h"

using namespace std;

int main ( int argc, char *argv[] ) {


  //////////////////////
  // Read user inputs //
  //////////////////////

  
  // Check that all required arguments have been included
  
  if ( argc < 2 ) {

    cout << "Missing input arguments" << endl;
    cout << "Include:" << endl;
    cout << "1. two-point function data file" << endl;
    cout << "(OPTIONAL) 1 if a print-out is wanted" << endl;
    return 1;
  }

  // If a print argument has been given, set Print (Print = 1 to print)
  
  int Print;
  
  if ( argc == 3 ) {

    Print = atoi( argv[2] );

  }

  // Set delta and number of configurations based on input files
  
  int Tsink;
  int Tsink = detTimestepNum( argv[1], 11 ); 

  // detTimestepNum is a function in "jk.h"

  int configNum;

  configNum = detConfigNum( argv[1], 11 );

  // detConfigNum() is a function in "jk.h"


  ////////////////
  // Initialize //
  ////////////////


  // 4th order tensor of two-point functions
  // ( twop[4][4][t][c] )

  vector< vector< vector< vector<double> > > > twop ( 4 );

  giveTensorTensor( &twop, 4, Tsink, configNum );

  // Matrix of projected two-point functions
  // ( proj[t][c] )
  
  vector< vector<double> > proj ( Tsink );

  giveMatrixCols( &proj, configNum );


  ///////////////////////////////////
  // Read and store values in file //
  ///////////////////////////////////


  for ( int i = 0; i < 4; i++ ) {

    for ( int j = 0; j < 4; j++ ) {
   
      

    }
  }

  return 0;

}


