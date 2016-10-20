#ifndef JK_H
#define JK_H

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace std;

int detTimestepNum( char * );

int detConfigNum( char *, int );

void readFifthColumnOfData( vector< vector<double> > *, char *, int, int );

void jackknife( vector< vector<double> > *, vector< vector<double> > * );

void effMass( vector< vector<double> > *, vector< vector<double> > * );

void averageRows( vector<double> *, vector<double> *, vector< vector<double> > * );

#endif
