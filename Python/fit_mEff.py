import math
import numpy as np
import argparse as argp
from os import listdir as ls
from scipy.optimize import curve_fit
import functions as fncs
import physQuants as pq
import readWrite as rw

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Electromagnetic Form Factor" )

parser.add_argument( "mEff_filename", action='store', type=str )

parser.add_argument( "fit_start", action='store', type=int )

parser.add_argument( "fit_end", action='store', type=int )

parser.add_argument( "-o", "--output", action='store', type=str, default="./mEff_fit.dat" )

args = parser.parse_args()

mEff_filename = args.mEff_filename

fitStart = args.fit_start

fitEnd = args.fit_end

output = args.output

timestepNum, binNum = rw.detTimestepAndConfigNum( mEff_filename )

#############################################
# Fill matrices with values from data files #
#############################################

# mEff[ b, t ]

mEff = rw.readDataFile( mEff_filename, timestepNum, binNum )

print "Read effective masses from data file"

# mEff_err[ t ]

mEff_err = np.std( mEff, axis=0 ) * float( binNum - 1 ) / math.sqrt( float( binNum ) )

########################
# Fit effective masses #
########################

mEff_fit = np.zeros( binNum )

for b in range( binNum ):

    mEff_fit[ b ] = np.polyfit( range( fitStart, fitEnd + 1 ), \
                                mEff[ b, fitStart : fitEnd + 1 ], \
                                0, w=mEff_err[ fitStart : fitEnd + 1 ] )

#####################
# Average over bins #
#####################

mEff_fit_avg = np.average( mEff_fit )

mEff_fit_err = np.std( mEff_fit ) * float( binNum - 1 ) / math.sqrt( float( binNum ) )

print str( mEff_fit_avg ) + "+/-" + str( mEff_fit_err )

######################
# Write output files #
######################

# Fitted effective mass

rw.writeFitDataFile( output, mEff_fit_avg, mEff_fit_err, fitStart, fitEnd )

