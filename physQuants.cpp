#include "physQuants.h"

using namespace std;

// Calculates the effective masses from a given 'binNum' by 'timestepNum' matrix
// of resampled two-point functions and puts them in a matrix of the same size.

void effMass( vector< vector<double> > *eMass, vector< vector<double> > *twoPtFuncsJK ) {
	
  int timestepNum = twoPtFuncsJK -> size();
  int binNum = twoPtFuncsJK -> at(0).size();
	
  twoPtFuncsJK -> push_back(twoPtFuncsJK -> at(0)); //set periodic boundary conditions
	
  //Calculate effective masses from two-point functions

  for(int t = 0; t < timestepNum; t++) { // Loop through timesteps
		
    for(int b = 0; b < binNum; b++) { // Loop through bins
			
      eMass -> at(t).at(b) = log( twoPtFuncsJK -> at(t).at(b) / twoPtFuncsJK -> at(t+1).at(b) );
		
    }
  }
	
  return;
}


// Calculates the ratio from two- and three-point functions for each
// timeslice, momentum transfer, and bin.

void calcRatio( vector< vector< vector<double> > > *rat,
	    vector< vector< vector<double> > > *thrPtFunc,
	    vector< vector< vector<double> > > *twoPtFunc ) {

  // thrPtFunc[q][t][b]

  int qNum = thrPtFunc -> size();
  int tSink = ( thrPtFunc -> at(0).size() ) - 1;
  int binNum = thrPtFunc -> at(0).at(0).size();

  // The ratio is split into 3 parts so that the calculation is more
  // readable here
  
  double rat1; // The ratio before the square root
  double rat2; // The numerator of the square root
  double rat3; // The denomenator of the square root
  //cout << endl;
  for ( int q = 0; q < qNum; q++ ) {

    for ( int t = 0; t <= tSink; t++ ) {

      for ( int b = 0; b < binNum; b++ ) {

	//cout << "q=" << q << " t=" << t << " b=" << b << endl;

	rat1 = thrPtFunc -> at(q).at(t).at(b) / twoPtFunc -> at(0).at(tSink).at(b);

	//cout << "rat1=" << thrPtFunc -> at(q).at(t).at(b) << "/";
	//cout <<  twoPtFunc -> at(0).at(tSink).at(b);
	//cout << "=" << rat1 << endl;

	rat2 = sqrt( twoPtFunc -> at(q).at(tSink - t).at(b)

		     * twoPtFunc -> at(0).at(t).at(b)

		     * twoPtFunc -> at(0).at(tSink).at(b) );

	//cout << "rat2=sqrt(" << twoPtFunc -> at(q).at(tSink - t).at(b);
	//cout << "*" << twoPtFunc -> at(0).at(t).at(b);
	//cout << "*" << twoPtFunc -> at(0).at(tSink).at(b) << ")";
	//cout << "=" << rat2 << endl;

	rat3 = sqrt( twoPtFunc -> at(0).at(tSink - t).at(b)

		      * twoPtFunc -> at(q).at(t).at(b)

		      * twoPtFunc -> at(q).at(tSink).at(b) );

	//cout << "rat3=sqrt(" << twoPtFunc -> at(0).at(tSink - t).at(b);
	//cout << "*" << twoPtFunc -> at(q).at(t).at(b);
	//cout << "*" << twoPtFunc -> at(q).at(tSink).at(b) << ")";
	//cout << "=" << rat3 << endl;

	rat -> at(q).at(t).at(b) = rat1 * rat2 / rat3;

	//cout << "ratio=" << rat1 << "*" << rat2 << "/" <<  rat3 << "=";
	//cout << rat -> at(q).at(t).at(b) << endl;

      }
    }
  }

  return;
}


// Calculates the scalar charge from the ratio, the fitted mass, and q squared

void calcScalarFF( vector< vector< vector<double> > > *scalarFF,
		   vector< vector< vector<double> > > *ratio,
		   vector<double> *mass, vector<int> *q2, int latticeDim ) {

  // thrPtFunc[q][t][b]

  // Set factor_p to 2*pi/L
  
  double factor_p = 2 * ( 4 * atan( 1 ) ) / latticeDim;
  int qNum = scalarFF -> size();
  int tSink = scalarFF -> at(0).size();
  int binNum = scalarFF -> at(0).at(0).size();

  double energy; // Energy value for calculation
  double preFactor; // Pre-factor to be multiplied by the ratio
  //cout << endl;
  for ( int b = 0; b < binNum; b++ ) {

    for ( int q = 0; q < qNum; q++ ) {

      //cout << "b=" << b << " q=" << q << endl;
      
      energy = sqrt( pow( factor_p * q2 -> at(q), 2 ) + pow( mass -> at(b), 2 ) );

      //cout << "energy=sqrt((" << factor_p << "*" << q2 -> at(q) << ")^2+";
      //cout << mass -> at(b) << "^2" << "=" << energy << endl; 

      // E = sqrt( ( factor_p * q )^2 + m^2 )

      preFactor = sqrt( pow( energy, 2 ) * ( energy + mass -> at(b) )
			/ ( 2 * pow( mass -> at(b), 3 ) ) );

      //cout << "preFactor=sqrt(" << energy << "^2*(" << energy << "+";
      //cout << mass -> at(b) << ")/2*" << mass -> at(b) << "^3)=" << preFactor;
      //cout << endl;
      
      // preFactor = sqrt( E^2 * ( E + m ) / ( 2 * m^3 ) )
      
      for ( int t = 0; t < tSink; t++ ) {

	//cout << "t=" << t << endl;
	
	scalarFF -> at(q).at(t).at(b) = preFactor * ratio -> at(q).at(t).at(b);

	//cout << "scalarFF=" << preFactor << "*" << ratio -> at(q).at(t).at(b);
	//cout << "=" << scalarFF -> at(q).at(t).at(b) << endl;

      }
    }
  }

  return;
}


// Project the nucleons by multiplying the nucleon 4x4matrix by 1/4( 1 + g4 )
// in the physical basis and taking the trace for each timestep and configuration.

void projNucl( vector< vector<double> > *proj, 
	       vector< vector< vector< vector<double> > > > *nucl ) {

  if ( nucl -> size() != 4 || nucl -> at(0).size() != 4 ) {
    
    cout << "ERROR( projNucl() ): first two dimensions of nucleon tensor ";
    cout << "must be 4x4" << endl;

    return;
  
  }

  int Tsink = nucl -> at(0).at(0).size();
  int configNum = nucl -> at(0).at(0).at(0).size();

  for ( int t = 0; t < Tsink; t++ ) {

    for ( int c = 0; c < configNum; c++ ) {

      // Multiple the sum of nucl[1][1] and nucl[2][2] by 1/4.
      // This is equivalent to multiplying by 1/4( 1 + g4 ) in the physical
      // basis and taking the trace

      proj -> at(t).at(c) = ( nucl -> at(1).at(1).at(t).at(c) 
			      
			      + nucl -> at(2).at(2).at(t).at(c) ) / 4;

    }
  }

  return;

}
