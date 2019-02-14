import math
import numpy as np
import argparse as argp
from os import listdir as ls
from scipy.optimize import curve_fit
import functions as fncs
import readWrite as rw
import physQuants as pq

Zv = 0.728

latticeSpacing = 0.098

latticeDim = 32

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Pion Electromagnetic Form Factor" )

parser.add_argument( "threep_dir", action='store', type=str )

parser.add_argument( "threep_template", action='store', type=str )

parser.add_argument( "twop_dir", action='store', type=str )

parser.add_argument( "twop_template", action='store', type=str )

parser.add_argument( "mEff_filename", action='store', type=str )

parser.add_argument( "mEff_fit_start", action='store', type=int )

parser.add_argument( "mEff_fit_end", action='store', type=int )

parser.add_argument( 't_sink', action='store', \
                     help="Comma seperated list of t sink's", \
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "-o", "--output_template", action='store', type=str, default="./*.dat" )

parser.add_argument( "-c", "--config_list", action='store', type=str, default="" )

args = parser.parse_args()

#########
# Setup #
#########

threepDir = args.threep_dir

twopDir = args.twop_dir

threep_template = args.threep_template

twop_template = args.twop_template

mEff_filename = args.mEff_filename

mEff_fitStart = args.mEff_fit_start

mEff_fitEnd = args.mEff_fit_end

tsink = args.t_sink

output_template = args.output_template

# Get configurations from given list or from given threep
# directory if list not given

if args.config_list:

    if args.config_list in ls( "./" ):

        with open( args.config_list, "r" ) as configFile:

            configList = configFile.read().splitlines()

    else:

        print "WARNING: Given configuration list not in this directory. " \
            + "Will use all configurations in configuration directory."

        configList = ls( threepDir )

else:

    configList = ls( threepDir )

    configList = sorted( configList )

    configFilename = "./confs.txt_Fpi"

    with open( configFilename, "w" ) as configFile:

        for config in configList:

            configFile.write( str( config ) + "\n" )

    print "Configuration list written"

configNum = len( configList )

# Set timestep and bin number from effective mass file

timestepNum, binNum = rw.detTimestepAndConfigNum( mEff_filename )

if configNum % binNum != 0:

    print "Number of configurations " + str( configNum ) \
        + " not evenly divided by number of bins " + str( binNum ) \
        + " in effective mass file " + mEff_filename + ".\n"

    exit()

binSize = configNum / binNum

########################
# Fit effective masses #
########################

# mEff[ b, t ]

mEff = rw.readDataFile( mEff_filename, timestepNum, binNum )

# mEff_err[ t ]

mEff_err = np.std( mEff, axis=0 ) * float( binNum - 1 ) / math.sqrt( float( binNum ) )

mEff_fit = np.zeros( binNum )

for b in range( binNum ):

    mEff_fit[ b ] = np.polyfit( range( mEff_fitStart, mEff_fitEnd + 1 ), \
                                mEff[ b, mEff_fitStart : mEff_fitEnd + 1 ], \
                                0, w=mEff_err[ mEff_fitStart : mEff_fitEnd + 1 ] )

print "Fit effective mass"

#####################
# Average over bins #
#####################

# Fitted effective mass
# mEff_fit_avg
    
mEff_fit_avg = np.average( mEff_fit )

mEff_fit_err = np.std( mEff_fit ) * float( binNum - 1 ) / math.sqrt( float( binNum ) )

################
# Momenta list #
################

# Read momenta list from dataset
# momList[ c, Q ]

momList = rw.getDatasets( threepDir, configList, threep_template, "Momenta_list" )[ :, 0, 0, ... ]

# Check that momenta agree across configurations

Qsq, Qsq_start, Qsq_end = fncs.processMomList( momList )

######################
# Convert Q^2 to GeV #
######################

Qsq_GeV = pq.convertQsqToGeV( Qsq, latticeSpacing * mEff_fit_avg, latticeSpacing, latticeDim )

#######################
# Two-point functions #
#######################

# Get the real part of two-point functions
# twop[ c, t, Q ]

twop = rw.getDatasets( twopDir, configList, twop_template, "twop" )[ :, 0, 0, ..., 0 ]

print "Read two-point functions from HDF5 files"

# Average over equal Q^2
# twop_avg[ Q^2, c, t ]

twop_avg = fncs.averageOverQsq( twop, Qsq_start, Qsq_end )

# Jackknife
# twop_jk[ Q^2, b, t ]

twop_jk = []

for q in range( twop_avg.shape[ 0 ] ):

    twop_jk.append( fncs.jackknife( twop_avg[ q, ... ], binSize ) )

twop_jk = np.array( twop_jk )

for ts in tsink:
    
    #########################
    # Three-point functions #
    #########################

    # Get the real part of gamma4 insertion three-point functions
    # threep[ c, t, Q ]

    threep_u = rw.getDatasets( threepDir, configList, threep_template, "tsink_" + str( ts ), "noether", "up", "threep" )[ :, 0, 0, ..., 3, 0 ]

    threep_s = rw.getDatasets( threepDir, configList, threep_template, "tsink_" + str( ts ), "noether", "strange", "threep" )[ :, 0, 0, ..., 3, 0 ]

    print "Read three-point functions from HDF5 files for tsink " + str( ts )

    # Add up and strange part multiplied by their EM charge

    threep = 2.0 / 3.0 * threep_u - 1.0 / 3.0 * threep_s

    # Average over equal Q^2
    # threep_avg[ Q^2, c, t ]
    
    threep_avg = fncs.averageOverQsq( threep, Qsq_start, Qsq_end )

    # Jackknife
    # threep_jk[ Q^2, b, t ]
    
    threep_jk = []

    for q in range( threep_avg.shape[ 0 ] ):

        threep_jk.append( fncs.jackknife( threep_avg[ q, ... ], binSize ) )

    threep_jk = np.array( threep_jk )

    #########################
    # Calculate form factor #
    #########################

    emff = pq.calcEMFF( threep_jk, twop_jk, Qsq, latticSpacing * mEff_fit, ts, latticeDim )

    #####################
    # Average over bins #
    #####################

    # Electromagnetic form factor
    # em_avg[ Q^2, t ]

    emff_avg = np.average( emff, axis=1 )

    emff_err = np.std( emff, axis=1 ) * float( binNum - 1 ) / math.sqrt( float( binNum ) )

    ######################
    # Write output files #
    ######################

    # Form factors for each Q^2
    
    emff_outFilename = output_template.replace( "*", "Fpi_tsink" + str( ts ) )

    rw.writeFormFactorFile( emff_outFilename, emff, Qsq )

    # Form factors for each Q^2 and bin

    emff_avg_outFilename = output_template.replace( "*", "avgFpi_tsink" + str( ts ) )

    rw.writeAvgFormFactorFile( emff_avg_outFilename, emff_avg, emff_err, Qsq )

    # Fitted effective mass

    mEff_outputFilename = output_template.replace( "*", "mEff_fit" )

    rw.writeFitDataFile( mEff_outputFilename, mEff_fit_avg, mEff_fit_err, mEff_fitStart, mEff_fitEnd )

    print "Wrote output files for tsink " + str( ts )

# End loop over tsink
