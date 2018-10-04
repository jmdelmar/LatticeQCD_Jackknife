import numpy as np
import argparse as argp
from scipy.optimize import curve_fit
import functions as fncs
import readWrite as rw
import physQuants as pq

Z = 1.0

particle_list = [ "pion", "kaon", "nucleon" ]

format_list = [ "gpu", "cpu" ]

# TO DO: add support nulceon

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

parser.add_argument( "particle", action='store', help="Particle to calculate gA for. Should be 'pion' or 'kaon'.", type=str )

parser.add_argument( 't_sink', action='store', \
                     help="Comma seperated list of t sink's", \
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "-o", "--output_template", action='store', type=str, default="./*.dat" )

parser.add_argument( "-f", "--data_format", action='store', help="Data format. Should be 'gpu' or 'cpu'.", type=str, default="gpu" )

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

dataFormat = args.data_format

# Check inputs

assert particle in particle_list, "Error: Particle not supported. " \
    + "Supported particles: " + str( particle_list )

assert dataFormat in format_list, "Error: Data format not supported. " \
    + "Supported particles: " + str( format_list )

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

for ts in tsink:
    
    #########################
    # Three-point functions #
    #########################

    # Get the real part of gxDx, gyDy, gzDz, and gtDt
    # three-point functions at zero-momentum
    # threep[ c, t ]

    if particle == "nucleon":

        if dataFormat == "cpu":

            filename_u_gxDx = threep_template + str( ts ) + ".up.h5"

            threep_u_gxDx = rw.getDatasets( threepDir, configList, filename_u_gxDx, \
                                            "=der:gxDx:sym=", "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

            filename_u_gyDy = threep_template + str( ts ) + ".up.h5"

            threep_u_gyDy = rw.getDatasets( threepDir, configList, filename_u_gyDy, \
                                            "=der:gyDy:sym=", "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

            filename_u_gzDz = threep_template + str( ts ) + ".up.h5"

            threep_u_gzDz = rw.getDatasets( threepDir, configList, filename_u_gzDz, \
                                            "=der:gzDz:sym=", "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

            filename_u_gtDt = threep_template + str( ts ) + ".up.h5"

            threep_u_gtDt = rw.getDatasets( threepDir, configList, filename_u_gtDt, \
                                            "=der:g0D0:sym=", "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

            filename_d_gxDx = threep_template + str( ts ) + ".dn.h5"

            threep_d_gxDx = rw.getDatasets( threepDir, configList, filename_d_gxDx, \
                                            "=der:gxDx:sym=", "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

            filename_d_gyDy = threep_template + str( ts ) + ".dn.h5"

            threep_d_gyDy = rw.getDatasets( threepDir, configList, filename_d_gyDy, \
                                            "=der:gyDy:sym=", "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

            filename_d_gzDz = threep_template + str( ts ) + ".dn.h5"

            threep_d_gzDz = rw.getDatasets( threepDir, configList, filename_d_gzDz, \
                                            "=der:gzDz:sym=", "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

            filename_d_gtDt = threep_template + str( ts ) + ".dn.h5"

            threep_d_gtDt = rw.getDatasets( threepDir, configList, filename_d_gtDt, \
                                            "=der:g0D0:sym=", "msq0000", "arr" )[ :, 0, 0, :, 0 ].real
            
            threep_gxDx = threep_u_gxDx - threep_d_gxDx

            threep_gyDy = threep_u_gyDy - threep_d_gyDy

            threep_gzDz = threep_u_gzDz - threep_d_gzDz

            threep_gtDt = threep_u_gtDt - threep_d_gtDt

        else:

            print "GPU format not supported for nucleon, yet."

            exit()

    else: # Particle is meson

        if dataFormat == "gpu":

            threep_gxDx = rw.getDatasets( threepDir, configList, threep_template, \
                                          "tsink_" + str( ts ), "oneD", "dir_00", \
                                          "up", "threep" )[ :, 0, 0, ..., 0, 1, 0 ]

            threep_gyDy = rw.getDatasets( threepDir, configList, threep_template, \
                                          "tsink_" + str( ts ), "oneD", "dir_01", \
                                          "up", "threep" )[ :, 0, 0, ..., 0, 2, 0 ]
    
            threep_gzDz = rw.getDatasets( threepDir, configList, threep_template, \
                                          "tsink_" + str( ts ), "oneD", "dir_02", \
                                          "up", "threep" )[ :, 0, 0, ..., 0, 3, 0 ]

            threep_gtDt = rw.getDatasets( threepDir, configList, threep_template, \
                                            "tsink_" + str( ts ), "oneD", "dir_03", \
                                            "up", "threep" )[ :, 0, 0, ..., 0, 4, 0 ]

            threep_s_gxDx = np.array( [] )
            
            threep_s_gyDy = np.array( [] )
        
            threep_s_gzDz = np.array( [] )
    
            threep_s_gtDt = np.array( [] )

            if particle == "kaon":
            
                threep_s_gxDx = rw.getDatasets( threepDir, configList, threep_template, \
                                                "tsink_" + str( ts ), "oneD", "dir_00", \
                                                "strange", "threep" )[ :, 0, 0, ..., 0, 1, 0 ]

                threep_s_gyDy = rw.getDatasets( threepDir, configList, threep_template, \
                                                "tsink_" + str( ts ), "oneD", "dir_01", \
                                                "strange", "threep" )[ :, 0, 0, ..., 0, 2, 0 ]
    
                threep_s_gzDz = rw.getDatasets( threepDir, configList, threep_template, \
                                                "tsink_" + str( ts ), "oneD", "dir_02", \
                                                "strange", "threep" )[ :, 0, 0, ..., 0, 3, 0 ]

                threep_s_gtDt = rw.getDatasets( threepDir, configList, threep_template, \
                                                "tsink_" + str( ts ), "oneD", "dir_03", \
                                                "strange", "threep" )[ :, 0, 0, ..., 0, 4, 0 ]
            
        elif dataFormat == "cpu":

            print "CPU format not supported for mesons, yet."
            
            exit()

    print "Read three-point functions from HDF5 files for tsink " + str( ts )

    # Subtract average over directions from gtDt

    threep = threep_gtDt - 0.25 * ( threep_gtDt + threep_gxDx + threep_gyDy + threep_gzDz )

    # Jackknife
    # threep_jk[ b, t ]
    
    threep_jk = fncs.jackknife( threep, binSize )

    #################
    # Calculate <x> #
    #################

    avgX = Z * pq.calcAvgX( threep_jk, twop_jk[ :, ts ], mEff_fit_avg )

    # Average over bins

    avgX_avg = np.average( avgX, axis=0 )

    avgX_err = np.std( avgX, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

    ######################
    # Write output files #
    ######################

    # <x>
    
    avgX_outFilename = output_template.replace( "*", "avgX_tsink" + str( ts ) )

    rw.writeAvgDataFile( avgX_avg, avgX_err, avgX_outFilename )

    # Fitted effective mass

    mEff_outputFilename = output_template.replace( "*", "mEff_fit" )

    rw.writeFitDatafile( mEff_outputFilename, mEff_fit_avg, mEff_fit_err, mEff_fitStart, mEff_fitEnd )

    ###############
    # Fit plateau #
    ###############

    fitStart = []

    fitEnd = []

    if ts == 12:

        fitStart = [ 5, 4, 3 ]

        fitEnd = [ 7, 8, 9 ]

    elif ts == 14:

        fitStart = [ 6, 5, 4 ]

        fitEnd = [ 8, 9, 10 ]

    elif ts == 16:

        fitStart = [ 7, 6, 5 ]

        fitEnd = [ 9, 10, 11 ]

    elif ts == 18:

        fitStart = [ 8, 7, 6 ]

        fitEnd = [ 10, 11, 12 ]

    elif ts == 20:

        fitStart = [ 9, 8, 7 ]

        fitEnd = [ 11, 12, 13 ]

    else:

        print "Tsink not supported."

        exit()

    # Loop over fit ranges

    for irange in range( len( fitStart ) ):

        avgX_fit = []

        # Fit each bin

        for x in avgX:

            avgX_fit.append( float( np.polyfit( range( fitStart[ irange ], fitEnd[ irange ] + 1 ), \
                                         x[ fitStart[ irange ] : fitEnd[ irange ] + 1 ], \
                                         0, w=avgX_err[ fitStart[ irange ] : fitEnd[ irange ] + 1 ] ) ) )

        avgX_fit = np.array( avgX_fit )

        # Average over bins

        avgX_fit_avg = np.average( avgX_fit )

        avgX_fit_err = np.std( avgX_fit ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

        # Write output files

        avgX_fit_outFilename = ""

        if particle == "nucleon":

            avgX_fit_outFilename = output_template.replace( "*", \
                                                            "avgX_fit_" \
                                                            "tsink" + str( ts ) \
                                                            + "_" + str( fitStart[ irange ] ) \
                                                            + "_" + str( fitEnd[ irange ] ) )

        else:

            avgX_fit_outFilename = output_template.replace( "*", \
                                                            "avgX_u_fit_" \
                                                            "tsink" + str( ts ) \
                                                            + "_" + str( fitStart[ irange ] ) \
                                                            + "_" + str( fitEnd[ irange ] ) )

        rw.writeFitDatafile( avgX_fit_outFilename, avgX_fit_avg, avgX_fit_err, fitStart[ irange ], fitEnd[ irange ] )

    if particle == "kaon":

        ###########################
        # Repeat for strange part #
        ###########################

        # Subtract average over directions from gtDt

        threep_s = threep_s_gtDt - 0.25 * ( threep_s_gtDt + threep_s_gxDx + threep_s_gyDy + threep_s_gzDz )

        # Jackknife
        # threep_s_jk[ b, t ]
    
        threep_s_jk = fncs.jackknife( threep_s, binSize )

        #################
        # Calculate <x> #
        #################

        avgX_s = Z * pq.calcAvgX( threep_s_jk, twop_jk[ :, ts ], mEff_fit_avg )

        # Average over bins

        avgX_s_avg = np.average( avgX_s, axis=0 )

        avgX_s_err = np.std( avgX_s, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

        ######################
        # Write output files #
        ######################

        # <x>
    
        avgX_s_outFilename = output_template.replace( "*", "avgX_s_tsink" + str( ts ) )

        rw.writeAvgDataFile( avgX_s_avg, avgX_s_err, avgX_s_outFilename )

        # Fitted effective mass

        mEff_outputFilename = output_template.replace( "*", "mEff_fit" )

        rw.writeFitDatafile( mEff_outputFilename, mEff_fit_avg, mEff_fit_err, mEff_fitStart, mEff_fitEnd )

        ###############
        # Fit plateau #
        ###############

        # Loop over fit ranges

        for irange in range( len( fitStart ) ):

            avgX_s_fit = []

            # Fit each bin
            
            for x in avgX_s:

                avgX_s_fit.append( float( np.polyfit( range( fitStart[ irange ], fitEnd[ irange ] + 1 ), \
                                                      x[ fitStart[ irange ] : fitEnd[ irange ] + 1 ], \
                                                      0, w=avgX_s_err[ fitStart[ irange ] : fitEnd[ irange ] + 1 ] ) ) )

            avgX_s_fit = np.array( avgX_s_fit )

            # Average over bins

            avgX_s_fit_avg = np.average( avgX_s_fit )

            avgX_s_fit_err = np.std( avgX_s_fit ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

            # Write output files

            avgX_s_fit_outFilename = output_template.replace( "*", \
                                                              "avgX_s_fit_" \
                                                              "tsink" + str( ts ) \
                                                              + "_" + str( fitStart[ irange ] ) \
                                                              + "_" + str( fitEnd[ irange ] ) )

            rw.writeFitDatafile( avgX_s_fit_outFilename, avgX_s_fit_avg, avgX_s_fit_err, fitStart[ irange ], fitEnd[ irange ] )

    print "Wrote output files for tsink " + str( ts )

# End loop over tsink
