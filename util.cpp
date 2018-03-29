#include "util.h"

extern char *threepDir;
extern char *twopDir;
extern char *sourceDir;
extern char *confList;
extern char *threepTemplate;
extern char *twopTemplate;
extern char *outputPre;
extern int binSize;
extern int tsink;
extern int timeDim;
extern bool Print;

using namespace std;

void usage( char *exe ) {

  if( strcmp( exe, "effMass_Mesons" ) == 0 ) {

    cout << "Usage:" << endl;

    cout << "--twop-dir: ";
    cout << "Path to home directory containing configuration ";
    cout << "sub-directories of two-point functions"<< endl;

    cout << "--twop-template: ";
    cout << "Template for two-point function files with ";
    cout << "_CONF_ representing the configurations and _SRC_ representing ";
    cout << "the source positions" << endl;

    cout << "--source-dir: ";
    cout << "Path to directory containing source lists named ";
    cout << "_src.list and prefixed with the configuration" << endl; 

    cout << "--config-list:";
    cout << "Path to and filename of configuration list" << endl; 

    cout << "-o: ";
    cout << "Output prefix" << endl; 

    cout << "--bin-size: ";
    cout << "Bin size" << endl;

    cout << "--time-dimension: ";
    cout << "Length of time dimension in two-point function files" << endl;

    cout << "-p: ";
    cout << "Sets printout option on" << endl;

  }
  else cout << "Usage: Executable is not supported" << endl;

  exit(1);
  return;
}

int process_command_line_option( int argc, char** argv, int* idx ) {

  int ret = -1;
  
  int i = *idx;

  if( strcmp( argv[i], "--help" ) == 0 ) {

    usage( argv[0] );

  }
  else if( strcmp( argv[i], "--threep-dir" ) == 0 ) {

    if( i+1 >= argc)
      usage( argv[0] );

    threepDir = argv[i+1];
    i++;
    ret = 0;

  }
  else if( strcmp( argv[i], "--twop-dir" ) == 0 ) {

    if( i+1 >= argc)
      usage( argv[0] );

    twopDir = argv[i+1];
    i++;
    ret = 0;

  }
  else if( strcmp( argv[i], "--source-dir" ) == 0 ) {

    if( i+1 >= argc)
      usage( argv[0] );

    sourceDir = argv[i+1];
    i++;
    ret = 0;

  }
  else if( strcmp( argv[i], "--config-list" ) == 0 ) {

    if( i+1 >= argc)
      usage( argv[0] );

    confList = argv[i+1];
    i++;
    ret = 0;

  }
  else if( strcmp( argv[i], "--threep-template" ) == 0 ) {

    if( i+1 >= argc)
      usage( argv[0] );

    threepTemplate = argv[i+1];
    i++;
    ret = 0;

  }
  else if( strcmp( argv[i], "--twop-template" ) == 0 ) {

    if( i+1 >= argc)
      usage( argv[0] );

    twopTemplate = argv[i+1];
    i++;
    ret = 0;

  }
  else if( strcmp( argv[i], "--bin-size" ) == 0 ) {

    if( i+1 >= argc)
      usage( argv[0] );

    binSize = atoi (argv[i+1] );
    i++;
    ret = 0;

  }
  else if( strcmp( argv[i], "--tsink" ) == 0 ) {

    if( i+1 >= argc)
      usage( argv[0] );

    tsink = atoi( argv[i+1] );
    i++;
    ret = 0;

  }
  else if( strcmp( argv[i], "--time-dimension" ) == 0 ) {

    if( i+1 >= argc)
      usage( argv[0] );

    timeDim = atoi( argv[i+1] );
    i++;
    ret = 0;

  }
  else if( strcmp( argv[i], "-o" ) == 0 ) {

    if( i+1 >= argc)
      usage( argv[0] );

    outputPre = argv[i+1];
    i++;
    ret = 0;

  }
  else if( strcmp( argv[i], "-p" ) == 0 ) {

    Print = true;
    i++;
    ret = 0;

  }

  *idx = i;
  return ret;

}
