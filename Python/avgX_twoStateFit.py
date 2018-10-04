import numpy as np
import argparse as argp
from scipy.optimize import leastsq
import functions as fncs
import readWrite as rw
import physQuants as pq

Z = 1.0

particle_list = [ "pion", "kaon" ]

# TO DO: add support for nulceon

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Calculate quark momentum fraction <x>" )

parser.add_argument( "threep_dir", action='store', type=str )

parser.add_argument( "threep_template", action='store', type=str )

parser.add_argument( "twop_dir", action='store', type=str )

parser.add_argument( "twop_template", action='store', type=str )

parser.add_argument( "mEff_filename", action='store', type=str )

parser.add_argument( "mEff_fit_start", action='store', type=int )

parser.add_argument( "mEff_fit_end", action='store', type=int )

parser.add_argument( "particle", action='store', help="Particle to calculate gA for. Should be pion or kaon.", type=str )

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

particle = args.particle

tsink = args.t_sink

output_template = args.output_template

# Check particle

assert particle in particle_list, "Error: Particle not supported. " \
    + "Supported particles: " + str( particle_list )

# Get configurations from given list or from given threep
# directory if list not given

configList = fncs.getConfigList( args.config_list, threepDir )

configNum = len( configList )

# Set timestep and bin number from effective mass file

timestepNum, binNum = rw.detTimestepAndConfigNum( mEff_filename )

if configNum % binNum != 0:

    print "Number of configurations " + str( configNum ) \
        + " not evenly divided by number of bins " + str( binNum ) \
        + " in effective mass file " + mEff_filename + ".\n"

    exit()

binSize = configNum / binNum

####################
# Effective masses #
####################

# Read from file
# mEff[ b, t ]

mEff = rw.readDataFile( mEff_filename, timestepNum, binNum )

# Read error from file
# mEff_err[ t ]

mEff_err = np.std( mEff, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

# Fit
# mEff_fit [ b ]

mEff_fit = np.zeros( binNum )

for b in range( binNum ):

    mEff_fit[ b ] = np.polyfit( range( mEff_fitStart, mEff_fitEnd + 1 ), \
                                mEff[ b, mEff_fitStart : mEff_fitEnd + 1 ], \
                                0, w=mEff_err[ mEff_fitStart : mEff_fitEnd + 1 ] )

print "Fit effective mass"

# Average over bins

mEff_fit_avg = np.average( mEff_fit )

mEff_fit_err = np.std( mEff_fit ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

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

twop_avg = np.average( twop_jk, axis=0 )

twop_err = np.std( twop_jk, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

twop_ts = []

threep_jk = []

threep_avg = []

threep_err = []

for ts in tsink:
    
    twop_ts.append( twop_jk[ :, ts ] )

    #########################
    # Three-point functions #
    #########################

    # Get the real part of gxDx, gyDy, gzDz, and gtDt
    # three-point functions at zero-momentum
    # threep[ c, t ]

    threep_gxDx_u = rw.getDatasets( threepDir, configList, threep_template, \
                                    "tsink_" + str( ts ), "oneD", "dir_00", \
                                    "up", "threep" )[ :, 0, 0, ..., 0, 1, 0 ]

    threep_gyDy_u = rw.getDatasets( threepDir, configList, threep_template, \
                                    "tsink_" + str( ts ), "oneD", "dir_01", \
                                    "up", "threep" )[ :, 0, 0, ..., 0, 2, 0 ]
    
    threep_gzDz_u = rw.getDatasets( threepDir, configList, threep_template, \
                                    "tsink_" + str( ts ), "oneD", "dir_02", \
                                    "up", "threep" )[ :, 0, 0, ..., 0, 3, 0 ]

    threep_gtDt_u = rw.getDatasets( threepDir, configList, threep_template, \
                                    "tsink_" + str( ts ), "oneD", "dir_03", \
                                    "up", "threep" )[ :, 0, 0, ..., 0, 4, 0 ]

    threep_g5gx_s = np.array( [] )

    threep_g5gy_s = np.array( [] )

    threep_g5gz_s = np.array( [] )
                    
    if particle == "kaon":

        threep_gxDx_s = rw.getDatasets( threepDir, configList, threep_template, \
                                        "tsink_" + str( ts ), "oneD", "dir_00", \
                                        "strange", "threep" )[ :, 0, 0, ..., 0, 1, 0 ]

        threep_gyDy_s = rw.getDatasets( threepDir, configList, threep_template, \
                                        "tsink_" + str( ts ), "oneD", "dir_01", \
                                        "strange", "threep" )[ :, 0, 0, ..., 0, 2, 0 ]
    
        threep_gzDz_s = rw.getDatasets( threepDir, configList, threep_template, \
                                        "tsink_" + str( ts ), "oneD", "dir_02", \
                                        "strange", "threep" )[ :, 0, 0, ..., 0, 3, 0 ]

        threep_gtDt_s = rw.getDatasets( threepDir, configList, threep_template, \
                                        "tsink_" + str( ts ), "oneD", "dir_03", \
                                        "strange", "threep" )[ :, 0, 0, ..., 0, 4, 0 ]

    print "Read three-point functions from HDF5 files for tsink " + str( ts )

    # Calculate the isovector

    threep_gxDx = np.array( [] )
            
    threep_gyDy = np.array( [] )
    
    threep_gzDz = np.array( [] )
            
    threep_gtDt = np.array( [] )

    if particle == "pion": 

        # u-d=2u (d=-u)

        threep_gxDx = 2 * threep_gxDx_u

        threep_gyDy = 2 * threep_gyDy_u

        threep_gzDz = 2 * threep_gzDz_u

        threep_gtDt = 2 * threep_gtDt_u

    elif particle == "kaon":

        # u

        #threep_gxDx = #threep_gxDx_u
                                   
        #threep_gyDy = #threep_gyDy_u
                                   
        #threep_gzDz = #threep_gzDz_u
                                   
        #threep_gtDt = #threep_gtDt_u

        # s

        #threep_gxDx = #threep_gxDx_s
                                   
        #threep_gyDy = #threep_gyDy_s
                                   
        #threep_gzDz = #threep_gzDz_s
                                   
        #threep_gtDt = #threep_gtDt_s

        # u-s

        threep_gxDx = threep_gxDx_u - threep_gxDx_s
                                                   
        threep_gyDy = threep_gyDy_u - threep_gyDy_s
                                                   
        threep_gzDz = threep_gzDz_u - threep_gzDz_s
                                                   
        threep_gtDt = threep_gtDt_u - threep_gtDt_s

    # Subtract average over directions from gtDt

    threep = threep_gtDt - 0.25 * ( threep_gtDt + threep_gxDx + threep_gyDy + threep_gzDz )

    # Jackknife
    # threep_jk[ ts ][ b, t ]
    
    threep_jk.append( fncs.jackknife( threep, binSize ) )

    threep_jk_cp = fncs.jackknife( threep, binSize )

    threep_avg = np.average( threep_jk_cp, axis=0 )

    threep_err = np.std( threep_jk_cp, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

    avgOutputFilename = output_template.replace( "*", "threep_tsink" + str( ts ) )

    rw.writeAvgDataFile( threep_avg, threep_err, avgOutputFilename )

# End loop over tsink

avgOutputFilename = output_template.replace( "*", "twop_avg" )

rw.writeAvgDataFile( twop_avg, twop_err, avgOutputFilename )

##################
# Two-state Fit  #
##################

fit = fncs.twoStateFit( twop_ts, threep_jk )

# Average over bins

fit_avg = np.average( fit, axis=0 )

fit_err = np.std( fit, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

print fit_avg

print fit_err
