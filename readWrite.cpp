#include "readWrite.h"

using namespace lQCD_jk;
using std::cout;
using std::endl;

void readTwopMesonFile_Qsq0(vector<double> *data, char *file, char *dataset, readInfo info) {

  herr_t status;

  int dimNum = 3;

  hsize_t subDim[dimNum];

  subDim[0] = info.timeDim; // Set the number of timesteps to read
  subDim[1] = 1; // Only get one Q^2 component
  subDim[2] = 1; // Only get one complex component

  hsize_t count[dimNum];
  hsize_t offset[dimNum];
  hsize_t stride[dimNum];
  hsize_t block[dimNum];

  offset[0] = 0; // Start at t=0
  offset[1] = 0; // Only get Q^2=0
  offset[2] = info.complex; // Re if 0, Im if 1

  for(int d=0; d<dimNum; d++)
    count[d] = subDim[d]; // Data are next to each other
  
  for(int d=0; d<dimNum; d++)
    stride[d] = 1; // Data are next to each other

  for(int d=0; d<dimNum; d++)
    block[d] = 1; // Data are next to each other

  float buff[ subDim[0] ][ subDim[1] ][ subDim[2] ];
      
  hid_t file_id = H5Fopen (file, H5F_ACC_RDWR, H5P_DEFAULT);
  hid_t dataset_id = H5Dopen2 (file_id, dataset, H5P_DEFAULT);
  hid_t memspace_id = H5Screate_simple (dimNum, subDim, NULL); 
  hid_t dataspace_id = H5Dget_space (dataset_id);

  status = H5Sselect_hyperslab (dataspace_id, H5S_SELECT_SET, offset, stride, count, block);  

  status = H5Dread (dataset_id, H5T_IEEE_F32LE, memspace_id, dataspace_id, H5P_DEFAULT, buff);

  status = H5Sclose (memspace_id);
  status = H5Sclose (dataspace_id);
  status = H5Dclose (dataset_id);
  status = H5Fclose (file_id);

  for(int t=0; t<subDim[0]; t++) 
    data->push_back( buff[t][0][0] );

return;
}

void readTwop_g5Mesons_Qsq0( vector< vector< vector<double> > > *data, 
			     char *homeDir, vector<string> *confs, 
			     vector< vector< vector<int> > > *srcPos, 
			     char *fnTemplate, readInfo info ) {

  int confsNum = confs->size();

  // Substitute configurations into filename template

  char confDelim[] = "_CONF_";

  vector<string> filenames_conf; 

  setFilename( &filenames_conf, homeDir, confs, fnTemplate, confDelim );
  
  // setFilename() is a function in "readWrite.h"

  for( int c=0; c<confsNum; c++ ) { // Loop over confs

    giveMatrixCols( &data->at(c), srcPos->at(c).size() );

    // giveMatrixCols is a function in "jk.h"

    // Substitute source position into filename template

    char srcDelim[] = "_SRC_";

    //char *fnTemplate_conf = filenames_conf.at(c);

    vector<string> filenames_src;

    char *fn_conf;

    sprintf( fn_conf, "%s", filenames_conf.at(c) );

    setFilename( &filenames_src, homeDir, confs, fn_conf, srcDelim );

    //setFilename( &filenames_src, homeDir, confs, fnTemplate_conf, srcDelim );

    // setFilename() is a function in "readWrite.h"

    for( int s=0; s<srcPos->at(c).size(); s++ ) { // Loop over sources

      vector<double> buff(info.timeDim);

      char dataset[256];

      sprintf( dataset, "conf_%s/sx%0dsy%0dsz%0dst%0d/g5/twop_meson_%d", 
	       confs->at(c), srcPos->at(c).at(s).at(0), srcPos->at(c).at(s).at(1), 
	       srcPos->at(c).at(s).at(2), srcPos->at(c).at(s).at(3), info.meson );
      // CJL: This needs to be changed to support source positions with t component > 99
      // Maybe read source group name directly from h5 file?

      char fn_src[256];

      sprintf( fn_src, "%s", filenames_src.at(s) );

      readTwopMesonsFile_Qsq0( &buff, fn_src, dataset, info );

      for( int t=0; t<info.timeDim; t++ ) // Loop over timesteps
	data->at(t).at(c).push_back( buff.at(t) );

    } // End source loop
  } // End confs loop

  return;
}


// Writes the name of a file contained in a sub-directory of the given home 
// directory. File should contain the name of its sub-directory in its name, 
// represented in the given filename template by delim.

void setFilename( vector<string> *filename, char *homeDir, 
		  vector<string> *subDirs, char *fnTemplate, char *delim ) { 

  vector<string> fnTokens; // The parts of the filename template seperated by delim

  split( &fnTokens, fnTemplate, delim );

  // split() is a funtion in "jk.h"

  int tokNum = fnTokens.size();

  for ( int sD = 0; sD < subDirs -> size(); sD++ ) { // Loop over sub-directories
    
    stringstream fnss;

    // Write path to file to string stream

    fnss << homeDir << "/" << subDirs -> at(sD) << "/";

    // Write current sub-directory name between filename tokens

    for ( int t = 0; t < tokNum - 1; t++ ) { // Loop through tokens

      fnss << fnTokens[t] << subDirs -> at(sD);
      
    }
   
    fnss << fnTokens[ tokNum - 1 ]; // Write last token to end of fnss

    string fn = fnss.str(); // Set fn to the contents of fnss

    filename -> push_back( fn ); // Set next component of filename[] equal to fn

  }

  return;

}

// Writes the name of a file contained in a sub-directory of the given home 
// directory. The name of the file should contain the name of its sub-directory,
// represented in the given filename template by a '_CONF_' and its source
// postition represented by _SRC_.

void setFilename_wSrc( vector< vector<string> > *filename, char *homeDir, 
		       vector<string> *subDirs, vector< vector<string> > *srcPos, 
		       char *fnTemplate ) { 

  vector<string> fnTokens_conf; // The parts of the filename template seperated by '_CONF_'

  char delim_conf[] = "_CONF_";

  char delim_src[] = "_SRC_";

  split( &fnTokens_conf, fnTemplate, delim_conf );

  // split() is a funtion in "jk.h"

  int tokNum_conf = fnTokens_conf.size();

  for ( int sD = 0; sD < subDirs -> size(); sD++ ) { // Loop over sub-directories
    
    stringstream fnss_conf;

    // Replace *CONF*'s with configuration

    fnss_conf << homeDir << "/" << subDirs -> at(sD) << "/";

    // Write current sub-directory name between filename tokens

    for ( int t = 0; t < tokNum_conf - 1; t++ ) { // Loop through tokens

      fnss_conf << fnTokens_conf[t] << subDirs -> at(sD);
      
    }
    
    fnss_conf << fnTokens_conf[ tokNum_conf - 1 ]; // Write last token to end of fnss_conf

    string filename_conf = fnss_conf.str(); // Filename with _CONF_'s replaced with configuration name

    // Replace _SRC_'s with source position

    for( int isrc=0; isrc<srcPos->at(sD).size(); isrc++) { // Loop over source positions

      vector<string> fnTokens_src; // The parts of the filename template seperated by '_SRC_'

      char fn_conf[256];

      sprintf( fn_conf, "%s", filename_conf );

      split( &fnTokens_src, fn_conf, delim_src );

      int tokNum_src = fnTokens_src.size();

      stringstream fnss_src;

      // Write current sub-directory name between filename tokens

      for ( int t = 0; t < tokNum_src - 1; t++ ) { // Loop through tokens

	fnss_src << fnTokens_src[t] << srcPos -> at(sD).at(isrc);
      
      }
    
      fnss_src << fnTokens_src[ tokNum_src - 1 ]; // Write last token to end of fnss_src

      string fn_src = fnss_src.str(); // Filename with _SRC_'s replaced with srciguration name    

      filename -> at(sD).push_back( fn_src ); // Set next component of filename[] equal to fn

    } // End source position loop
  } // End sub-directory loop

  return;

}

void getSourcePositions( vector< vector< vector<int> > > *srcPos, char *srcDir, 
			 vector<string> *confs ) {
  
  for ( int c = 0; c < confs -> size(); c++ ) { // Loop over configs

    char srcFile[256];

    sprintf( srcFile, "%s/%s_src.list", srcDir, confs->at(c) );

    ifstream data;

    data.open( srcFile, ifstream::in );

    int src; // Placeholder for source components in source file

    if ( data.is_open() ) {

      // Run through each row of four components in source

      data >> src;

      for( int sP=0; !data.eof(); sP++ ) { // Loop through source positions

	srcPos->at(c).at(sP) = vector<int>(4); // give next source position 4 components

	for( int sC=0; sC<4; sC++ ) { // Loop through source components

	  srcPos->at(c).at(sP).at(sC) = src;

	  data >> src;

	} // End component loop

      } // End position loop

      data.close();

    }
    else throw srcFile;

  } // End configs loop

  return;
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


// Reads a given file of strings and puts each string into the given vector.
// Throws the filename as an expection if the file cannot be opened.

void readStringFile( vector<string> *vals, char *fileName ) {
  
  ifstream file(fileName);

  string s; // placeholder for strings in file to be set in vector

  if ( file.is_open() ) {

    file >> s; // Set placeholder to first string in file

    for ( int i = 0; !file.eof(); i++ ) {
      
      vals -> push_back( s ); //store s in vector

      file >> s; // Set placeholder to next string in file

    }

    file.close();

  }
  else throw fileName;
  
  return;
}


// Reads a given file by configurations in seperate directories and puts the 
// colNth column out of colTot columns into the given matrix. File should 
// contain the name of its sub-directory in its name only once, represented in 
// the given filename template by a '*'.

void readNthDataCol_rbc( vector< vector<double> > *vals, vector<string> *filenames, 
			 int colN, int colTot ) {

  for( int c = 0; c < filenames -> size(); c++ ) {

    ifstream data( filenames -> at(c).c_str() );

    double d; // placeholder for doubles in data files to be set in matrix

    int timestepNum = vals -> size();
  
    int t = 0;
	
    if ( data.is_open() ) {

      for ( int i = 0; !data.eof(); i++ ) {
      
	data >> d; // Set placeholder to next number in data file
     
	//If d is in the colNth column, store it in vals[t][c]
      
	if( ( i % colTot + 1 ) == colN ) {

	  vals -> at(t).push_back( d );

	  t++;
				
	}

      }

      data.close();

    }
    else throw filenames -> at(c);
      
  }
  
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


// Reads a given file by configurations in different directories with colTot 
// columns and timestepNum timesteps per configuration and fills the given 
// vector with the colNth double in the rowNth row for each configuration.
// File should contain the name of its sub-directory in its name only once, 
// represented in the given filename template by a '*'.

void readNthDataRow_rbc( vector<double> *vals, vector<string> *filenames, 
			 int rowN, int colN, int colTot ) {

  ifstream data;

  for( int c = 0; c < filenames -> size(); c++ ) {

    data.open( filenames -> at(c).c_str(), ifstream::in );

    double d; // Placeholder for doubles in data file

    if ( data.is_open() ) {

      // Run through doubles in data file until the rowNth row is reached
      
      data >> d;

      for ( int i = 0; !data.eof(); i++ ) {

	int check = 0; // This will be used to check that data is only stored once 
                       // per configuration
	
	// If d is in the rowNth row and colNth column, store it in vals

	if( i / colTot == rowN && ( i % colTot +1 ) == colN ) {

	  vals -> push_back( d );

	  check++;

	}

	// Check that this has only been done once per configuration

	if ( check > 1 ) {

	  throw filenames -> at(c);

	}

	data >> d;

      }

      data.close();

    }
    else throw filenames -> at(c);

  }

  return;
}
