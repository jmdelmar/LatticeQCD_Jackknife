 /* Extracts the 5th column from a given data file and organizes it by
timesteps in rows and configurations in columns, calculates the JK averages
for each timestep and bin, the effective mass for each timestep and bin,
the average and error of the effective mass for each timestep, and writes
a file containing the average effective masses with their respective
errors for each timestep.

Input: 
1.	data file of two-point functions to be read
2.	bin size (make sure numbers of configurations are divided by bin size
 	exactly)
3.      1 if printouts of the JK averaged two-point functions and effective masses are wanted, otherwise,
	any integer which is not 1
*/

#include "jk.h"
#include <cstdlib> //for atoi()

using namespace std;

int main(int argc, char *argv[]) {

  //Set number of timesteps and cofigurations and bin size based on user input  \ \
  
  int timestepNum = detTimestepNum( argv[1] );
  int configNum = detConfigNum( argv[1], timestepNum );
  int binSize = atoi(argv[2]);

  //Check that configNum is exactly divided by binSize

  while( configNum % binSize != 0 ) {
    
    cout << "configNum " << configNum << " is divided by binSize " << binSize << " with a remainder. ";
    cout << "Please enter a new binSize." << endl;

    cin >> binSize;

  }

  int binNum = configNum / binSize;
  int Print = atoi(argv[3]);
  
  //Initialize vectors/matrices and their pointers
	
  //Matrix of two point functions

  vector< vector<double> > twoPointFuncs(timestepNum); //matrix w/ 'timestepNum' rows
	
  for (int i=0; i<timestepNum; i++) {
	  
    twoPointFuncs[i] = vector<double>(configNum); //give each row 'configNum' columns
    //twoPointFuncs[t][c]
	  
  }

  vector< vector<double> > *tpf = &twoPointFuncs;
	
  //Matrix of JK averaged TPFs

  vector< vector<double> > twoPointFuncs_jk(timestepNum);  //matrix w/ 'timestepNum' rows

  for (int i=0; i<timestepNum; i++) {

    twoPointFuncs_jk[i] = vector<double>(binNum); //give each row 'binNum' columns
    //twoPointFuncs_jk[t][b]

  }

  vector< vector<double> > *tpf_jk = &twoPointFuncs_jk;

  //Matrix of effective masses

  vector< vector<double> > eMass(timestepNum);  //matrix w/ 'timestepNum' rows

  for (int i=0; i<timestepNum; i++) {

    eMass[i] = vector<double>(binNum); //give each row 'binNum' columns
    //eMass[t][b]

  }

  vector< vector<double> > *em = &eMass;

  //Vector of average two-point functions;

  vector<double> twoPointFuncs_avg(timestepNum);

  vector<double> *tpf_avg = &twoPointFuncs_avg;


  vector<double> twoPointFuncs_err(timestepNum);

  vector<double> *tpf_err = &twoPointFuncs_err;

  //Vector of average effective masses and errors

  vector<double> eMass_avg(timestepNum);

  vector<double> *em_avg = &eMass_avg;


  vector<double> eMass_err(timestepNum);

  vector<double> *em_err = &eMass_err;

  //Extract 5th column from data file and organize in matrix

  readFifthColumnOfData(tpf, argv[1], timestepNum, configNum);

  //Print matrix of two-point functions

  if(Print == 1){

    cout << "Two-point functions:" << endl << endl;

    for (int t = 0; t < timestepNum; t++) {

      cout << "t = " << t << ":       ";

      for(int c = 0; c < configNum - 1; c++) {

	cout << twoPointFuncs[t][c] << ", ";

      }

      cout << twoPointFuncs[t][configNum-1] << endl << endl;

    }
  }
	
  //Calculate JK averages
	
  jackknife(tpf_jk, tpf);
	
  //Print matrix of jackknife averages

  if(Print == 1){

    cout << "Jackknifed two-point functions:" << endl << endl;

    for (int t = 0; t < timestepNum; t++) {
		
      cout << "t = " << t << ":	";
		
      for(int b = 0; b < binNum - 1; b++) {
			
	cout << twoPointFuncs_jk[t][b] << ", ";
						
      }
	
      cout << twoPointFuncs_jk[t][binNum-1] << endl << endl;
		
    } 
  }

  //Calculate average two-point functions

  averageRows( tpf_avg, tpf_err, tpf_jk );
	
  //Calculate effective masses

  effMass(em, tpf_jk);
	
  //Print matrix of effective masses

  if(Print == 1){

    cout << "Effective masses:" << endl << endl;
	
    for (int t = 0; t < timestepNum; t++) {
		
      cout << "t = " << t << ":	";
		
      for(int b = 0; b < binNum - 1; b++) {
			
	cout << eMass[t][b] << ", ";
						
      }
	
      cout << eMass[t][binNum-1] << endl << endl;
    } 
  }
	
  //Calculate average effective masses and errors

  averageRows(em_avg, em_err, em);

  //Write file containing effective masses for each timestep and bin
	
  ofstream effMassesFile;
	
  effMassesFile.open("effMasses.dat");
	
  if(effMassesFile.is_open() ) {
		
    for (int t = 0; t < timestepNum; t++) {
		
      for(int b = 0; b < binNum; b++) {
			
	effMassesFile << eMass[t][b] << " ";
						
      }
			
      effMassesFile << endl;
			
    }

    effMassesFile.close();
		
  }
	
  else cout << "Could not open file to write effective masses\n";
	
  //Write file containing effective mass error for each timestep for fit mass calculation

  ofstream errFile;
	
  errFile.open("effMassErrors.dat");
	
  if(errFile.is_open() ) {
		
    for(int t = 0; t < timestepNum; t++) {
			
      errFile << eMass_err[t] << endl;
			
    }
		
    errFile.close();
  }
	
  else cout << "Could not open file to write effective mass errors\n";
	
  //Write file containing average two-point functions and their respective errors and timesteps

  ofstream avgTPFuncFile;

  avgTPFuncFile.open ("avgTwoPointFunc.dat");

  if(avgTPFuncFile.is_open() ) {

    for(int t = 0; t < timestepNum; t++) {

      avgTPFuncFile << t << " " << twoPointFuncs_avg[t];
      avgTPFuncFile << " " << twoPointFuncs_err[t] << endl;

    }

    avgTPFuncFile.close();

  }

  else cout << "Could not open file to write average effective masses\n";

  //Write file containing average effective masses and their respective errors and timesteps

  ofstream avgEffMassFile;
	
  avgEffMassFile.open ("avgEffMass.dat");
	
  if(avgEffMassFile.is_open() ) {
		
    for(int t = 0; t < timestepNum; t++) {
		
      avgEffMassFile << t << " " << eMass_avg[t];
      avgEffMassFile << " " << eMass_err[t] << endl;
			
    }
		
    avgEffMassFile.close();
		
  }
	
  else cout << "Could not open file to write average effective masses\n";
	
	
  return 0;
	
}
