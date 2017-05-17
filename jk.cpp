/* Functions to be used for calculating JK averages and effective masses */

#include "jk.h"

using namespace std;

// Reads a given data file with the timesteps for a number of configurations
// in the first of given colNum columns and determines the number of timesteps

int detTimestepNum( char *file, int colTot ) {
  
  double d; // Placeholder for doubles in file

  int tOld = -1; // Placeholder for timestep values in file
  int tNew; // Placeholder for timestep values in file to be compared to tOld
            // so that files with repeating timesteps still work

  int timestepNum = 0;

  ifstream data(file);
  
  if ( data.is_open() ) {

    // Move through the values in the data file until the timestep value in the
    // first column of the file resets to zero. This corresponds to the beginning
    // of a new configuration

    data >> d; // Move to first double in data file
    
    while ( tOld >= timestepNum-1 ) { 

      tNew = d;

      if ( tNew != tOld ) { // If timestep is different than the last

	tOld = tNew;
	  
	timestepNum++;

      }

      // Move through rest of the row
	
      for ( int j = 0; j < colTot; j++ ) {

	data >> d;
	  
      }
	
    }

    data.close();

    timestepNum--; // Need to do this because for loop will go through to end
                   // and increase timestepNum once t goes back to 0

  }
  else cout << "ERROR (detTimestepNum): Could not open file to determine the number of timesteps" << endl;

  return timestepNum;
}

// Reads a given data file with the timesteps for a number of configurations
// in the first of given colNum columns and determines the number of configurations

int detConfigNum( char *file, int colTot ) {

  double d;
  
  int tOld = 1;
  int tNew;

  int configNum = 0;

  ifstream data(file);


  if ( data.is_open() ) {

    data >> d; // Move to first double in file
    
    while ( !data.eof() ) {

      tNew = d;

      if ( tNew < tOld ) {

	configNum++;

      }

      tOld = tNew;
	
      for ( int i = 0; i < colTot; i++ ) {

	data >> d;

      }

    }
      
    data.close();

    return configNum;

  }
  else cout << "ERROR (detConfigNum): Could not open file to determine the number of configurations" << endl;

}


// Reads a given file and puts the colNth column out of colTot columns into the given matrix

void readNthDataCol( vector< vector<double> > *vals, char *fileName, int colN, int colTot ) {
  
  ifstream data(fileName);

  double d; // placeholder for doubles in data files to be set in matrix

  int timestepNum = vals -> size();
  int configNum = vals -> at(0).size();
  
  int t = 0;
  int c = 0;
	
  if ( data.is_open() ) {

    for ( int i = 0; !data.eof(); i++ ) {
      
      data >> d; // Set placeholder to next number in data file
     
      //Extract colNth column of data
      
      if( ( i % colTot + 1 ) == colN ) {

	vals -> at(t).at(c) = d;

	t++;
				
      }

      //Change to next configuration if all timesteps have been stored
			
      if(t == timestepNum) {
   
	t = 0;

	c++;
				
      }

    }

    data.close();

  }
	
  else cout << "ERROR (readNthDataCol): Could not open file to read data \n";
  
  return;
}


// Reads a given file with data for different momentum transfers and reads the
// colNth column out of colTot columns into the given matrix for each momentum

void readNthMomDataCol( vector< vector< vector< vector<double> > > > *vals,
			char *fileName, vector<int> *q2Num, int colN, int colTot ) {

  // vals[q][t][c][n]

  ifstream data( fileName );

  double d; // placeholder for doubles in data files to be set in matrix

  int q2Tot = vals -> size(); // Number of time slices
  int timeNum = vals -> at(0).size(); // Total number of different q squares
  int configNum = vals -> at(0).at(0).size(); // Number of configurations

  if ( data.is_open() ) {

    // Loop through configurations
    
    for ( int c = 0; c < configNum; c++ ) { 

      // Loop through time slices
      
      for ( int t = 0; t < timeNum; t++ ) {

	// Loop through each momentum transfer
	
	for ( int q = 0; q < q2Tot; q++ ) {

	  int n = 0; // Dimension to store each value for the same q squared

	  // Loop through each column in as many rows as there are possible
	  // momentum combinations
	  
	  for ( int i = 0; i < ( q2Num -> at(q) * colTot ); i++ ) { 
	                                                            
	    data >> d;                                              
	                                                            
	    if ( ( i % colTot ) + 1 == colN ) {                         

	      vals -> at(q).at(t).at(c).at(n) = d;

	      n++; // Change to next value for this q squared
	      /*
	      cout << "q=" << q << " t=" << t << " c=" << c << " n=" << endl;
	      cout << d << endl;
	      */
	    }
	  }
	}
      }
    }

    data.close();

  }
  else cout << "ERROR (ReadNthMomDataCol): Could not open file to read data \n";

  return;
}


// Reads a given file with colTot columns and timestepNum timesteps per
// configuration and fills the given vector with the colNth double in the rowNth
// row for each configuration

void readNthDataRow( vector<double> *vals, char *fileName, int timestepNum, int rowN, int colN, int colTot ) {

  ifstream data( fileName );

  double d; // Placeholder for doubles in data file

  int configNum = vals -> size();

  if ( data.is_open() ) {

    for ( int c = 0 ; c < configNum; c++ ) { // Loop through configurations

      // Run through doubles in data file until the rowNth row is reached
    
      for ( int i = 0; i < rowN * colTot; i++ ) {

	data >> d;

      }

      // Run through doubles in rowNth row until colNth column is reached

      for ( int i = 0; i < colN; i++ ) {

	data >> d;

      }

      vals -> at(c) = d; // Set cth component of vals to the colNth double in the
                         // rowNth row for configuration c

      // Run through doubles until the end of the rowNth row

      for ( int i = 0; i < colTot - colN; i++ ) {

	data >> d;

      }
      
      //Run through doubles until the end of the configuration

      for ( int i = 0; i < ( timestepNum - rowN ) * colTot; i++ ) {

	data >> d;

      }
    }

    data.close();

  }
  else cout << "ERROR (readNthDataRow): Could not open file to read data" << endl;

  return;
}


// Calculates the resampled averages of values in rows of a given 'configNum' by
// 'timestepNum' matrix and puts them in a given 'binNum' by 'timestepNum' matrix

void jackknifeMatrix( vector< vector<double> > *vals_jk, vector< vector<double> > *vals ) {

  int timestepNum = vals -> size();
  int configNum = vals -> at(0).size();
  int binNum = vals_jk -> at(0).size();
  int binSize = configNum / binNum;

  //Calculate the resampled averages

  for ( int t = 0; t < timestepNum; t++ ) { // Loop through timesteps
	
    for(int b = 0; b < binNum; b++) { // Loop through which bin is excluded

      double sum = 0;

      // Loop through configurations up to excluded bin
      
      for(int c = 0; c < b * binSize; c++) {
				
	sum += vals -> at(t).at(c);
				
      }

      // Loop through configurations starting after excluded bin
				
      for(int c = b * binSize + binSize; c < configNum; c++) {
	
	sum += vals -> at(t).at(c);
				
      }

      // Calculate and store averages
      
      vals_jk -> at(t).at(b) = sum / (configNum - binSize);

    }
  }
	
  return;	
}


// Calculates the resampled averages of values in a given 'configNum'
// dimensional vector and puts them in a given 'binNum' dimensional vector

void jackknifeVector( vector<double> *vals_jk, vector<double> *vals ) {
	
  int configNum = vals -> size();
  int binNum = vals_jk -> size();
  int binSize = configNum / binNum;
	
  //Calculate the resampled averages

  for(int b = 0; b < binNum; b++) { // Loop through which bin is excluded

    double sum = 0;

    // Loop through configurations up to excluded bin
    
    for(int c = 0; c < b * binSize; c++) { 
				
      sum += vals -> at(c);
				
    }

    // Loop through configurations starting after excluded bin
    
    for(int c = b * binSize + binSize; c < configNum; c++) { 
				
      sum += vals -> at(c);
				
    }

    // Calculate and store averages
    
    vals_jk -> at(b) = sum / (configNum - binSize);
			
  }
	
  return;
	
}


// Calculates the averages and standard deviations of the rows of a given matrix
// and puts them into given vectors whose demensions are the number of rows of
// the matrix

void averageRows( vector<double> *avg, vector<double> *stdDev, vector< vector<double> > *vals ) {
	
  int rowNum = vals -> size();
  int columnNum = vals -> at(0).size();
		
  for(int r = 0; r < rowNum; r++) { // Loop through rows

    double sum = 0;
	
    for(int c = 0; c < columnNum; c++) { // Loop through columns

      // Sum each value in a row
      
      sum += vals -> at(r).at(c);
      /*
      cout << vals -> at(r).at(c) << "+";
      */
    }

    // Calculate and store averages
    
    avg -> at(r) = sum / columnNum;
    /*
    cout << "/" << columnNum << "=" << avg -> at(r) << endl;
    */
  }
	
  for(int r = 0; r < rowNum; r++) { // Loop through rows
		
    double sumDiffSquare = 0;
	
    for(int c = 0; c < columnNum; c++) { //Loop through columns

      // Sum the difference squared between each value in row and the average
      
      sumDiffSquare += pow( avg -> at(r) - vals -> at(r).at(c) , 2 );
			
    }

    // Calculate and store the standard deviations
    
    stdDev -> at(r) = sqrt( sumDiffSquare * (columnNum - 1) / columnNum );
		
  }
	
  return;
}


// Writes a file containing a matrix so that the 1st column is the row and the
// 2nd column is the value in the matrix. The columns of the matrix are repeated
// once the row is complete so that it can be read by readNthDataCol()

void writeMatrixFile( char *fileName, vector< vector<double> > *data ) {

  int rowNum = data -> size();
  int colNum = data -> at(0).size();

  ofstream file;

  file.open( fileName );

  if( file.is_open() ) {

    for ( int c = 0; c < colNum; c++ ) {
      
      for ( int r = 0; r < rowNum; r++ ) {

	file << left << setw( 15 ) << r;

	file << left << setw( 15 ) << data -> at(r).at(c) << endl;

      }
    }

    file.close();
    
  }
  else {

    cout << "ERROR (writeMatrixFile): Could not open output file\n";

  }
  
    return;
}


// Writes two vectors to an output file. The first column of the file is each
// timestep, the second is the vector given as the second argument, and the third
// is the vector given as the third argument

void writeVectorFile( char *fileName, vector<double> *data, vector<double> *err ) {

  int timestepNum = data -> size();
  
  ofstream file;

  file.open( fileName );
	
  if( file.is_open() ) {
    		
    for (int t = 0; t < timestepNum; t++) { // Loop through timesteps
		
      file << left << setw(5) << t; // Write t in 1st column

      file << left << setw(15) << data -> at(t); // Write data in 2nd column

      file << err -> at(t) << endl; // Write error in 3rd column
						
    }
    
    file.close();
		
  }
  else {

    cout << "ERROR (writeVectorFile): Could not open output file\n";

  }

  return;
}

// Writes a fit, its error, and the first and last time-slices in the fit. The
// 1st column is the 1st and last time-slices, given as 'firstT' and 'lastT',
// the 2nd column is the fit, repeated twice, and the last column is the fit
// error, repeated twice. This format is used so that the fit can easily be
// plotted as a line.

void writeFitFile( char *fileName, double fit, double err, int firstT, int lastT, int Tsink ) {

  ofstream file;

  file.open( fileName );
	
  if( file.is_open() ) {

    file << left << setw(15) << firstT; // Write the 1st time-slice
    file << left << setw(15) << fit; // Write 1st double
    file << left << setw(15) << err; // Write 2nd double
    file << Tsink << endl; // Write Tsink

    file << left << setw(15) << lastT; // Write the last time-slice
    file << left << setw(15) << fit; // Write 1st double
    file << left << setw(15) << err; // Write 2nd double
    file << Tsink << endl; // Write Tsink						
      
    file.close();
  }
  else {
    
    cout << "ERROR (writeFitFile): Could not open output file\n";

  }

  return;
}


// Prints a tensor to standard out and formats it so that the first dimension is
// printed one after another as elements seperated by a column in each row

void printTensor( vector< vector< vector<double> > > *vals, vector<int> *q2, string title ) {

  int dim0 = vals -> size();
  int dim1 = vals -> at(0).size();
  int dim2 = vals -> at(0).at(0).size();

  cout << endl << title << endl; // Print title

  for ( int d0 = 0; d0 < dim0; d0++ ) { // Loop through seperate matrices

    cout << endl << "Q^2=" << q2 -> at(d0) << endl; // Print which Q squared
      
    for ( int d1 = 0; d1 < dim1; d1++ ) { // Loop through rows

      for ( int d2 = 0; d2 < dim2 - 1; d2++ ) { // Loop through columns

	cout << vals -> at(d0).at(d1).at(d2) << ", "; // Print each value, seperated
                                                     // by commas 
      }

      cout << vals -> at(d0).at(d1).at(dim2-1) << endl; // Print last value of of
                                                       // each row w/o a comma after
    }                                                  // it

  }

  return;
}


// Prints a matrix to standard out and formats it so that each value is seperated
// by a comma and each row is on a different line

void printMatrix( vector< vector<double> > *vals, string title ) {

  int rowNum = vals -> size();
  int colNum = vals -> at(0).size();

  cout << endl << title << endl << endl; // Print title
  
  for (int r = 0; r < rowNum; r++) { // Loop through rows
		
    for(int c = 0; c < colNum - 1; c++) { // Loop through columns
			
      cout << vals -> at(r).at(c) << ", "; // Print each value, seperated by
                                           // commas 
    }
		
    cout << vals -> at(r).at(colNum-1) << endl; // Print last value of of each
                                                // row w/o a comma after it		
  }
	
  cout << endl;


  return;
}


// Prints a vector to standard out, formatted so that each component is on a
// seperate line

void printVector( vector<double> *vals, string title ) {
  
  int timestepNum = vals -> size();

  cout << endl << title << endl << endl; // Print title
  
  for (int t = 0; t < timestepNum; t++) { // Loop through timesteps
		
    cout << vals -> at(t) << endl; // Print each each component of the vector
						
  }

  return;
}

void printVector( vector<int> *vals, string title ) {
  
  int timestepNum = vals -> size();

  cout << endl << title << endl << endl; // Print title
  
  for (int t = 0; t < timestepNum; t++) { // Loop through timesteps
		
    cout << vals -> at(t) << endl; // Print each each component of the vector
						
  }

  return;
}


// Gives a matrix a given number of columns in each of its rows

void giveMatrixCols( vector< vector<double> > *vals, int colNum ) {

  int rowNum = vals -> size();
  
  for (int r = 0; r < rowNum; r++) { // Loop through rows
		
    vals -> at(r) = vector<double>(colNum); //give each row 'colNum' columns
    //	vals[r][c]
	
  }

  return;
}


// Gives a 3rd order tensor a matrix with given dimensions in each element of
// its 0th dimension

void giveTensorMatrix( vector< vector< vector<double> > > *vals,
		       int dim1, int dim2 ) {

  int dim0 = vals -> size();
  
  for (int d0 = 0; d0 < dim0; d0++) { // Loop through 0th dimension

    // Initialize matrix to be placed in 0th dimension
    
    vector< vector<double> > a ( dim1 ); 

    giveMatrixCols( &a, dim2 );

    // giveMatrixCols() is another function in "jk.h"
    
    vals -> at(d0) = a;
    //	vals[d0][d1][d2]
	
  }

  return;
}


// Gives a 4th order tensor a 3rd order tensor with given dimensions in each
// element of it 0th dimension

void giveTensorTensor( vector< vector< vector< vector<double> > > > *vals,
		       int dim1, int dim2, vector<int> *dim3 ) {

  int dim0 = vals -> size(); 

  if ( dim0 != dim3 -> size() ) {

    cout << "ERROR(giveTensorTensor): The dimension of the 3rd dimension vector";
    cout << "must be the same as the dimension of the tensor." << endl;

    return;

  }
  
  for (int d0 = 0; d0 < dim0; d0++) { // Loop through 0th dimension

    // Initialize tensor to be placed in 0th dimension
    
    vector< vector< vector<double> > > a ( dim1 ); 

    giveTensorMatrix( &a, dim2, dim3 -> at(d0) );

    // giveTensorMatrix() is another function in "jk.h"
    
    vals -> at(d0) = a;
    //	vals[d0][d1][d2][d3]
	
  }

  return;
}

// Fills a matrix with test values, starting at zero and going up by one for
// each entry

void fillTestMatrix( vector< vector<double> > *vals ) {

  int rowNum = vals -> size();
  int colNum = vals -> at(0).size();

  int a = 0; // Test value to be put in matrix
	
  for (int r = 0; r < rowNum; r++) { // Loop through rows 
		
    for(int c = 0; c < colNum; c++) { // Loop through columns
			
      vals -> at(r).at(c) = a; // Set each entry as a
      a++; // Increase a by 1
			
    }
  }	

  return;
}
