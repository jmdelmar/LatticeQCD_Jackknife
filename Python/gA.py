import numpy as np
import argparse as argp
from scipy.optimize import curve_fit
import functions as fncs
import readWrite as rw
import physQuants as pq

Z = 0.8

particle_list = [ "pion", "kaon" ]

# TO DO: add support for nulceon

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Calculate axial charge gA" )

parser.add_argument( "threep_dir", action='store', type=str )

parser.add_argument( "threep_template", action='store', type=str )

parser.add_argument( "twop_dir", action='store', type=str )

parser.add_argument( "twop_template", action='store', type=str )

parser.add_argument( "particle", action='store', help="Particle to calculate gA for. Should be pion or kaon.", type=str )

parser.add_argument( 't_sink', action='store', \
                     help="Comma seperated list of t sink's", \
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "binSize", action='store', type=int )

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

particle = args.particle

tsink = args.t_sink

binSize = args.binSize

output_template = args.output_template

# Check particle

assert particle in particle_list, "Error: Particle not supported. " \
    + "Supported particles: " + str( particle_list )

# Get configurations from given list or from given threep
# directory if list not given

configList = fncs.getConfigList( args.config_list, threepDir )

configNum = len( configList )

assert configNum % binSize == 0, "Number of configurations " \
    + str( configNum ) + " not evenly divided by bin size " \
    + str( binSize ) + ".\n"

binNum = configNum / binSize

#######################
# Two-point functions #
#######################

# Get the real part of two-point functions
# twop[ c, t ]

twop = rw.getDatasets( twopDir, configList, twop_template, "twop" )[ :, 0, 0, ..., 0, 0 ]

print "Read two-point functions from HDF5 files"

# Jackknife
# twop_jk[ b, t ]

twop_jk = fncs.jackknife( twop, binSize )

for ts in tsink:
    
    #########################
    # Three-point functions #
    #########################

    # Get the real part of g5gx, g5gy, and g5gz
    # three-point functions at zero-momentum
    # threep[ c, t ]

    threep_g5gx_u = rw.getDatasets( threepDir, configList, threep_template, \
                                  "tsink_" + str( ts ), "ultra_local", "up", \
                                  "threep" )[ :, 0, 0, ..., 0, 6, 1 ]

    threep_g5gy_u = rw.getDatasets( threepDir, configList, threep_template, \
                                  "tsink_" + str( ts ), "ultra_local", "up", \
                                  "threep" )[ :, 0, 0, ..., 0, 7, 1 ]

    threep_g5gz_u = rw.getDatasets( threepDir, configList, threep_template, \
                                  "tsink_" + str( ts ), "ultra_local", "up", \
                                  "threep" )[ :, 0, 0, ..., 0, 8, 1 ]

    # This is for future nucleon support
    """
    threep_g5gx_d = np.array( [] )

    threep_g5gy_d = np.array( [] )

    threep_g5gz_d = np.array( [] )
    """
    threep_g5gx_s = np.array( [] )

    threep_g5gy_s = np.array( [] )

    threep_g5gz_s = np.array( [] )
                    
    if particle == "kaon":

        threep_g5gx_s = rw.getDatasets( threepDir, configList, threep_template, \
                                        "tsink_" + str( ts ), "ultra_local", "strange", \
                                        "threep" )[ :, 0, 0, ..., 0, 6, 1 ]

        threep_g5gy_s = rw.getDatasets( threepDir, configList, threep_template, \
                                        "tsink_" + str( ts ), "ultra_local", "strange", \
                                        "threep" )[ :, 0, 0, ..., 0, 7, 1 ]

        threep_g5gz_s = rw.getDatasets( threepDir, configList, threep_template, \
                                        "tsink_" + str( ts ), "ultra_local", "strange", \
                                        "threep" )[ :, 0, 0, ..., 0, 8, 1 ]

    print "Read three-point functions from HDF5 files for tsink " + str( ts )

    # Calculate the isovector

    threep_g5gx = np.array( [] )

    threep_g5gy = np.array( [] )

    threep_g5gz = np.array( [] )

    if particle == "pion": 

        # u

        #threep_g5gx = #threep_g5gx_u
        
        #threep_g5gy = #threep_g5gy_u

        #threep_g5gz = #threep_g5gz_u

        # u-d=2u (d=-u)

        threep_g5gx = 2 * threep_g5gx_u
        
        threep_g5gy = 2 * threep_g5gy_u

        threep_g5gz = 2 * threep_g5gz_u

    elif particle == "kaon":

        # u

        #threep_g5gx = #threep_g5gx_u

        #threep_g5gy = #threep_g5gy_u

        #threep_g5gz = #threep_g5gz_u

        # s

        #threep_g5gx = #threep_g5gx_s

        #threep_g5gy = #threep_g5gy_s

        #threep_g5gz = #threep_g5gz_s

        # u-s

        threep_g5gx = threep_g5gx_u - threep_g5gx_s

        threep_g5gy = threep_g5gy_u - threep_g5gy_s

        threep_g5gz = threep_g5gz_u - threep_g5gz_s

    # Average over insertion currents

    threep = ( threep_g5gx + threep_g5gy + threep_g5gz ) / 3.0 

    # Jackknife
    # threep_jk[ b, t ]
    
    threep_jk = fncs.jackknife( threep, binSize )

    ######
    # gA #
    ######

    # Calculate
    # gA[ b, t ]

    gA = Z * pq.calcgA( threep_jk, twop_jk[ :, ts ] )

    # Average over bins
    # gA_avg[ t ]

    gA_avg = np.average( gA, axis=0 )

    gA_err = np.std( gA, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

    # Write output files

    gA_outFilename = output_template.replace( "*", "gA_tsink" + str( ts ) )

    rw.writeAvgDataFile( gA_outFilename, gA_avg, gA_err )

    print "Wrote output files for tsink " + str( ts )

# End loop over tsink
