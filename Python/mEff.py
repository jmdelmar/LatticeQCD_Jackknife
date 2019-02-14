import math
import numpy as np
import argparse as argp
from os import listdir as ls
import functions as fncs
import readWrite as rw
import physQuants as pq

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Effective Mass" )

parser.add_argument( "twop_dir", action='store', type=str )

parser.add_argument( "filename_template", action='store', type=str )

parser.add_argument( "binSize", action='store', type=int )

parser.add_argument( "-o", "--output", action='store', type=str, default="./*.dat" )

parser.add_argument( "-c", "--config_list", action='store', type=str, default="" )

args = parser.parse_args()

twopDir = args.twop_dir

filename_template = args.filename_template

binSize = args.binSize

output_template = args.output

configList = fncs.getConfigList( args.config_list, twopDir )

configNum = len( configList )

assert configNum % binSize == 0, "Number of configurations " \
    + str( configNum ) + " not evenly divided by bin size " \
    + str( binSize ) + ".\n"

binNum = configNum / binSize

#############################################
# Fill matrices with values from data files #
#############################################

# Get real part of zero-momentum two-point functions
# twop[ c, t ]

twop = rw.getDatasets( twopDir, configList, filename_template, "twop" )[ :, 0, 0, :, 0, 0 ]

print "Read two-point functions from HDF5 files"
            
#############
# Jackknife #
#############

twop_jk = fncs.jackknife( twop, binSize )

twop_avg = np.average( twop_jk, axis=0 )

twop_err = np.std( twop_jk, axis=0 ) * float( binNum - 1 ) / math.sqrt( float( binNum ) )

############################
# Fold two-point functions #
############################

twop_fold = fncs.fold( twop_jk )

##############################
# Calculate effective masses #
##############################

# mEff[ b, t ]

mEff = pq.mEffFromSymTwop( twop_fold )

# mEff_avg[ t ]

mEff_avg = np.average( mEff, axis=0 )

# mEff_err[ t ]

mEff_err = np.std( mEff, axis=0 ) * float( binNum - 1 ) / math.sqrt( float( binNum ) )

######################
# Write output files #
######################

# Two-point functions

outputFilename = output_template.replace( "*", "twop" )

avgOutputFilename = output_template.replace( "*", "twop_avg" )

rw.writeDataFile( outputFilename, twop_jk )

rw.writeAvgDataFile( avgOutputFilename, twop_avg, twop_err )

print "Wrote two-point function files"

# Effective mass

outputFilename = output_template.replace( "*", "mEff" )

avgOutputFilename = output_template.replace( "*", "mEff_avg" )

rw.writeDataFile( outputFilename, mEff )

rw.writeAvgDataFile( avgOutputFilename, mEff_avg, mEff_err )

print "Wrote effective mass files"
            
