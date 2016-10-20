/* Functions to be used for calculating JK averages and effective masses */

#include "jk.h"

using namespace std;

/* Reads a given data file with the timesteps for a number of configurations
   in the first of five columns and determines the number of timesteps
*/

int detTimestepNum( char *file ) {
  
  double d;
  int t = 0;
  int timestepNum = 0;

  ifstream data(file);
  data >> d;
  
  if ( data.is_open() ) {

    int i = 0;

    while ( t >= timestepNum-1  ) {

      if ( i % 5 == 0 ) {

	t = d;

	timestepNum++;

      }
      
      data >> d;
      i++;

    }

    data.close();

    timestepNum--;

  }

  return timestepNum;
}

/* Reads a given data file with the timesteps for a number of configurations
in the first of five columns and determines the number of configurations
*/

int detConfigNum( char *file, int timestepNum ) {

  double d;
  int c = 0;
  int configNum = 0;

  ifstream data(file);
  data >> d;

  if ( data.is_open() ) {

    while ( !data.eof() ) {

      for ( int i = 0; i < 5 * timestepNum; i++ ) {

	data >> d;

      }

      configNum++;

    }

    data.close();

    return configNum;

  }
}


/* Reads a given file and puts 5th column into a given matrix */

void readFifthColumnOfData(vector< vector<double> > *twoPtFuncs, char *fileName, int timestepNum, int configNum) {

  double d; // placeholder for doubles in data files to be set in vector
	
  ifstream data(fileName);
  data >> d;
	
  int t = 0;
  int c = 0;
		
  if (data.is_open()) {
		
    int i = 0;
		
    while ( !data.eof() ) {
				
      //Extract 5th column of data

      if((i + 1) % 5 == 0) {
				
	twoPtFuncs->at(t).at(c) = d;
	t++;
				
      }
			
      //Change to next configuration if all timesteps have been stored
			
      if(t == timestepNum) {
					
	t = 0;
	c++;
				
      }
				
      data >> d; // Move to next number in data file
      i++;
			
    }

    data.close();
		
  }
	
  else cout << "Could not open file to read data\n";
	
		
  return;
}


/*  Calculates JK averages of values in rows of a given 'configNum' x 'timestepNum' matrix and puts them in a given 'binNum' x 'timestepNum' matrix */

void jackknife( vector< vector<double> > *jkAvg, vector< vector<double> > *vals ) {
	
  int timestepNum = vals->size();
  int configNum = vals->at(0).size();
  int binNum = jkAvg->at(0).size();
  int binSize = configNum / binNum;
	
  /*[In future, add a check that binSize goes into configNum exactly]*/

  // 	Calculate the JK averages  \\

  double sum = 0;

  for(int t = 0; t < timestepNum; t++) { //go through each timestep
	
    for(int b = 0; b < binNum; b++) { //go through which bin is excluded
		
      for(int c = 0; c < b * binSize; c++) { //go through each config up to excluded bin
				
	sum += vals->at(t).at(c);
				
      }
					
      for(int c = b * binSize + binSize; c < configNum; c++) { //go through each config starting after excluded bin
				
	sum += vals->at(t).at(c);
				
      }
			
      jkAvg->at(t).at(b) = sum / (configNum - binSize);
			
      sum = 0;
    }
  }
	
  return;
	
}


/* Calculates the effective masses from a given 'binNum' x 'timestepNum' matrix of JK averaged two-point functions and puts them in a matrix of the same size. */

void effMass( vector< vector<double> > *eMass, vector< vector<double> > *twoPtFuncsJK ) {
	
  int timestepNum = twoPtFuncsJK->size();
  int binNum = twoPtFuncsJK->at(0).size();
	
  twoPtFuncsJK->push_back(twoPtFuncsJK->at(0)); //set periodic boundary conditions
	
  /*//	Print matrix with updated bc's  \\
	
    for (int t = 0; t <= timestepNum; t++) { 
		
    for(int c = 0; c < binNum - 1; c++) {
			
    cout << twoPtFuncsJK[t][c] << ", ";
						
    }
		
    cout << twoPtFuncsJK[t][binNum - 1]<< endl;
		
    }	*/
		
  //Calculate effective masses from two-point functions

  for(int t = 0; t < timestepNum; t++) {
		
    for(int b = 0; b < binNum; b++) {
			
      eMass->at(t).at(b) = log( twoPtFuncsJK->at(t).at(b) / twoPtFuncsJK->at(t+1).at(b) );
		
    }
  }
	
  return;
	
}


/* Calculates the averages and standard deviations of the rows of a given matrix and puts
   them into given vectors whose demensions are the number of rows of the matrix */

void averageRows( vector<double> *avg, vector<double> *stdDev, vector< vector<double> > *val ) {
	
  int rowNum = val->size();
  int columnNum = val->at(0).size();
		
  for(int r = 0; r < rowNum; r++) {
		
    double sum = 0;
	
    for(int c = 0; c < columnNum; c++) {
		
      sum += val->at(r).at(c);
    }
		
    avg->at(r) = sum / columnNum;
		
  }
	
  for(int r = 0; r < rowNum; r++) {
		
    double sumDiffSquare = 0;
	
    for(int c = 0; c < columnNum; c++) {
		
      sumDiffSquare += pow( avg->at(r) - val->at(r).at(c) , 2 );
			
    }
		
    stdDev->at(r) = sqrt( sumDiffSquare * (columnNum - 1) / columnNum );
		
  }
	
  return;
}
