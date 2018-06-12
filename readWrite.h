#ifndef READWRITE_H
#define READWRITE_H
#endif

#include <iostream>
#include <vector>
#include "hdf5.h"
#include "jk.h"
#include <dirent.h>
namespace lQCD_jk {

  typedef struct readInfo {

    int timeDim;
    int complex; // 0 if Re, 1 if Im
    int meson; // 1 or 2

  } readInfo;

}

using namespace lQCD_jk;

using std::vector;

void readTwopMesonFile_Qsq0( vector<double> *data, char *file, 
			     char *dataset, readInfo info );


void readTwop_pseudoscalarMesons_Qsq0( vector< vector< vector<double> > > *data, 
				       char *homeDir, vector<string> *confs, 
				       vector< vector< vector<int> > > *srcPos, 
				       char *fnTemplate, readInfo info );


// Writes the name of a file contained in a sub-directory of the given home 
// directory. File should contain the name of its sub-directory in its name 
// only once, represented in the given filename template by a deliminator.

void setFilename( vector<string> *, char *, vector<string> *, char *, char * );


// Writes the name of a file contained in a sub-directory of the given home 
// directory. The name of the file should contain the name of its sub-directory,
// represented in the given filename template by a "_CONF_" and its source
// postition represented by "_SRC_".

void setFilename_wSrc( vector< vector<string> > *filename, char *homeDir, 
		       vector<string> *subDirs, vector< vector < vector <int> > > *srcPos, 
		       char *fnTemplate );


void getSourcePositions( vector< vector< vector<int> > > *srcPos, char *srcDir, 
			 vector<string> *confs );


// Reads a given file and puts the colNth column out of colTot columns into the
// given matrix

void readNthDataCol( vector< vector<double> > *vals, char *, int, int );


// Reads a given file by configurations in seperate directories and puts the 
// colNth column out of colTot columns into the given matrix

void readNthDataCol_rbc( vector< vector<double> > *, vector<string> *, int, 
			 int );


// Reads a given file of strings and puts each string into the given vector

void readStringFile( vector<string> *, char * );
  

// Reads a given file with data for different momentum transfers and reads the
// colNth column out of colTot columns into the given matrix for each momentum

void readNthMomDataCol( vector< vector< vector< vector<double> > > > *, char *,
			vector<int> *, int, int );


// Reads a given file with colTot columns and timestrepNum timesteps per
// configuration and fills the given vector with the colNth double in the rowNth
// row for each configuration

void readNthDataRow( vector<double> *, char *, int, int, int, int );


// Reads a given file by configurations in different directories with colTot 
// columns and timestepNum timesteps per configuration and fills the given 
// vector with the colNth double in the rowNth row for each configuration

void readNthDataRow_rbc( vector<double> *, vector<string> *, int, int, int );


void readSourcePositionFile( vector <vector<int> > *srcPos, char *srcFile );
