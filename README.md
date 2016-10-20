# 2PtFunc_JK
Calculates the jackknife average of two-point functions and calculates the effective mass from them.

Contents:

1. jk.cpp: library of functions need for calculating the effective mass
2. calcEM.cpp: program to calculate effective mass
3. calcMassFit.cpp: program to calculate fit for effective masses once calcEM has been run (see file for inputs)
4. 2pt_func.dat: sample data of two-point functions for 64 timesteps and 50 configurations

Input for calcEM.exe:

1. data file of two-point functions to be read                 
2. bin size (make sure numbers of configurations are divided by bin size exactly)    
3. 1 if printouts of the JK averaged two-point functions and effective masses are wanted, otherwise, any integer which is not 1
