/* Fits given effective masses over given time interval, averages fitted
masses over all bins, and calculates its error

Input:
1.	file of effective masses to be read
2.	file of errors to be read
3.	number of timesteps
4.	number of bins
5.	first timestep to be fit
6.	last timestep to be fit 

	*/

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <cstdlib> //for atoi()

using namespace std;

  ////////////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
 //Function to calculate fitted masses for each bin from the effective masses\\
//////////////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

vector<double> fitMass(vector< vector<double> > val, vector<double> err, int startTime, int endTime) {
	
	int timeNum = val.size();
	int binNum = val[0].size();
	
	vector<double> fMass(binNum);
	
	double sumErrSquare; // Normalizing factor
	
	for(int t = startTime; t < endTime; t++) {
		
		if(err[t] > 0) {
			
			sumErrSquare += pow(err[t], -2);
		}
	}
	
	for(int b = 0; b < binNum; b++) {
		
		double weightedSum = 0;
		
		for(int t = startTime; t < endTime; t++) {
			
			if(err[t] > 0) {
		
				weightedSum += val[t][b] / pow(err[t], 2);
				
			}
		}
		
		fMass[b] = weightedSum / sumErrSquare;
		
	}
	
	return fMass;
}

  ///////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
 //Function to calculate the averaged fitted mass over all bins\\
/////////////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

double averageFit(vector<double> vals) {
	
	int binNum = 0;
	
	for(int a = 0; a < vals.size(); a++) {
		
		if(vals[a] > -100) {
			
			binNum++;
			
		}
	}
	
	double sum = 0;
	
	for(int b = 0; b < vals.size(); b++) {
		
		if(vals[b] > -100) {
			
			sum += vals[b];
			
		}
		
	}
	
	double avg = sum / binNum;
	
	return avg;
}
  /////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\
 //Function to calculate the error of the fitted mass\\
 //////////////////////////\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

double fittedMassError(vector<double> vals) {
	
	int binNum = 0;
	
	for(int a = 0; a < vals.size(); a++) {
		
		if(vals[a] > -100) {
			
			binNum++;
			
		}
	}
	
	double err;
	
	double avg = averageFit(vals);
	
	double sumDiffSquare = 0;
	
	for(int b = 0; b < vals.size(); b++) {
		
		if(vals[b] > 0 ) {
			
		sumDiffSquare += pow(avg - vals[b], 2);
		
		}
	}
	
	err = sqrt( sumDiffSquare * (binNum - 1) / binNum );
	
	return err;
}

  ////////////////\\\\\\\\\\\\\\\
 //////////Main function\\\\\\\\\\
//////////////////\\\\\\\\\\\\\\\\\

int main(int argc, char *argv[]) {

	int timestepNum = atoi(argv[3]);
	int binNum = atoi(argv[4]);
	int startTime = atoi(argv[5]);
	int endTime = atoi(argv[6]);
	
	vector<double> effMass_err(timestepNum);
	vector<double> fMass(binNum);
	double fMass_avg;
	double fMass_err;
	
//	Initialize effective mass matrix  \\
	
	vector< vector<double> > effMass(timestepNum); //matrix w/ 'timestepNum' rows
	
	for (int t=0; t < timestepNum; t++) {
		
		effMass[t] = vector<double>(binNum); //give each row 'binNum' columns
	//	effMass[t][c]
	
	}
			
//	Read effective mass data and place values into effMass  \\
	
	ifstream effMassData(argv[1]);
	
	if (effMassData.is_open()) {
		
		double em; // Placeholder for values to go into effMass
		int max_i = 10000000;
		int t = 0;
		int b = 0;
		
		
		int i=0;
		while (t < timestepNum) {
		//for (int i = 0; i < max_i; i++) {
			
			effMassData >> em;
				
			if(em == em) {
			
				effMass[t][b] = em;
				
			}
			
			else effMass[t][b] = -1000;
			
			b++;
			
		//	Change to next timestep if all bins have been stored  \\	
			
			if(b == binNum) {
					
				b = 0;
				t++;
					
			}
			
			
			
			i++;
			/*
			if (effMassData.eof()) {
				
				break;
				
			}
			*/
			
		}
		
		effMassData.close();
		
	}	
	
	else cout << "Could not open effective mass file to read data\n";
	
//	Read effective mass error file and place values into effMass_err  \\

	ifstream errData(argv[2]);
	
	if (errData.is_open()) {
		
		double er; // Placeholder for values to go int effMass_err
		
		errData >> er;
		
		for (int t = 0; t < timestepNum; t++) {
				
			if(er == er) {
			
				effMass_err[t] = er;
			
			}
			
			else effMass_err[t] == 0;
				
			errData >> er; // Move to next number in data file
			
			if (errData.eof()) {
				
				break;
				
			}	
		}

		errData.close();
		
	}	
	
	else cout << "Could not open effective mass error file to read data\n";
	
//	Calculate fitted mass from effective masses  \\

	fMass = fitMass(effMass, effMass_err, startTime, endTime);

//	Calculate average fitted mass  \\

	fMass_avg = averageFit(fMass);

//	Calculate fitted mass error  \\

	fMass_err = fittedMassError(fMass);

//	Write file containing average fitted mass and its error  \\

	ofstream avgFittedMassFile;
	
	avgFittedMassFile.open("avgFittedMass.dat");
	
	if(avgFittedMassFile.is_open() ) {
		
		avgFittedMassFile << fMass_avg << " " << fMass_err << endl;
		
		avgFittedMassFile.close();
		
	}
	
	else cout << "Could not open file to write average fitted mass\n";
	
	
	return 0;
	
}
