import numpy as np
import argparse as argp
import functions as fncs
import readWrite as rw

format_list = [ "gpu", "cpu" ]

particle_list= [ "pion", "kaon" ]

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description= \
                              "Jackknife two- and three-point functions \
                              for <x>" )

parser.add_argument( "threep_dir", action='store', type=str )

parser.add_argument( "threep_tokens", action='store', \
                     type=lambda s: [str(token) for token in s.split(',')], \
                     help="Comma seperated list of filename tokens. CPU: part before tsink, part before momentum boost components. GPU: part before momentum boost components, part after momentum boost; * for configuration number." )

parser.add_argument( "twop_dir", action='store', type=str )

parser.add_argument( "twop_template", action='store', type=str )

parser.add_argument( "particle", action='store', help="Particle to calculate gA for. Should be 'pion' or 'kaon'.", type=str )

parser.add_argument( 't_sink', action='store', \
                     help="Comma seperated list of t sink's", \
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "mom_squared", action='store', type=int )

parser.add_argument( "binSize", action='store', type=int )

parser.add_argument( "-o", "--output_template", action='store', type=str, default="./*.dat" )

parser.add_argument( "-f", "--data_format", action='store', help="Data format. Should be 'gpu' or 'cpu'.", type=str, default="gpu" )

parser.add_argument( "-c", "--config_list", action='store', type=str, default="" )

args = parser.parse_args()

#########
# Setup #
#########

threepDir = args.threep_dir

twopDir = args.twop_dir

threep_tokens = args.threep_tokens

twop_template = args.twop_template

particle = args.particle

tsink = args.t_sink

momSq = args.mom_squared

tsinkNum = len( tsink )

ts_range_str = "tsink" + str(tsink[0]) + "_" + str(tsink[-1])

binSize = args.binSize

output_template = args.output_template

dataFormat = args.data_format

# Check inputs

assert particle in particle_list, "Error: Particle not supported. " \
    + "Supported particles: " + str( particle_list )

assert dataFormat in format_list, "Error: Data format not supported. " \
    + "Supported particles: " + str( format_list )

# Get configurations from given list or from given 
# threep directory if list not given

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

twop = []

if dataFormat == "cpu":

    twop = rw.getDatasets( twopDir, configList, twop_template, \
                                            "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

else:
        
    twop = rw.getDatasets( twopDir, configList, twop_template, "twop" )[ :, 0, 0, ..., 0, 0 ]

print "Read two-point functions from HDF5 files"

# Jackknife
# twop_jk[ b, t ]

twop_jk = fncs.jackknife( twop, binSize )

outputFilename = output_template.replace( "*", "twop" )

rw.writeDataFile( outputFilename, twop_jk )

threep_jk = []

momList = []

if momSq == 0:

    momList = [ [ 0, 0, 0 ] ]

elif momSq == 2:

    momList = [ [ +1, +1,  0], \
              [ +1,  0, +1], \
              [  0, +1, +1], \
              [ +1, -1,  0], \
              [ +1,  0, -1], \
              [  0, +1, -1], \
              [ -1, +1,  0], \
              [ -1,  0, +1], \
              [  0, -1, +1], \
              [ -1, -1,  0], \
              [ -1,  0, -1], \
              [  0, -1, -1] ]

elif momSq == 3:

    momList = [ [ +1, +1, +1], \
              [ -1, +1, +1], \
              [ +1, -1, +1], \
              [ +1, +1, -1], \
              [ +1, -1, -1], \
              [ -1, +1, -1], \
              [ -1, -1, +1], \
              [ -1, -1, -1] ]

else:

    print( "Error: momentum squared " + str(momSq) + " not supported." )

    exit()

momBoostNum = len( momList )
                                                         
for ts in tsink:
    
    threep_mom = [ [] for imom in range( momBoostNum ) ]

    threep_s_mom = [ [] for imom in range( momBoostNum ) ]

    for imom in range( momBoostNum ):

        if dataFormat == "cpu":
        
            threep_template = threep_tokens[0] + str(ts) \
                              + threep_tokens[1] \
                              + fncs.signToString( momList[imom][0] ) \
                              + str(momList[imom][0]) + "_" \
                              + fncs.signToString( momList[imom][1] ) \
                              + str(momList[imom][1]) + "_" \
                              + fncs.signToString( momList[imom][2] ) \
                              + str(momList[imom][2])

        else:

            threep_template = threep_tokens[0] + \
                              fncs.signToString( momList[imom][0] ) \
                              + str(momList[imom][0]) + "_" \
                              + fncs.signToString( momList[imom][1] ) \
                              + str(momList[imom][1]) + "_" \
                              + fncs.signToString( momList[imom][2] ) \
                              + str(momList[imom][2]) \
                              + threep_tokens[1]

        #########################
        # Three-point functions #
        #########################

        # Get the real part of gxDx, gyDy, gzDz, and gtDt
        # three-point functions at zero-momentum
        # threep[ c, t ]

        threeps = rw.readAvgXFile( threepDir, configList, threep_template, \
                                   ts, particle, dataFormat)

        threep_gxDx = threeps[0]
        threep_gyDy = threeps[1]
        threep_gzDz = threeps[2]
        threep_gtDt = threeps[3]
    
        threep_s_gxDx = []
        threep_s_gyDy = []
        threep_s_gzDz = []
        threep_s_gtDt = []

        if particle == "kaon":

            threep_s_gxDx = threeps[4]
            threep_s_gyDy = threeps[5]
            threep_s_gzDz = threeps[6]
            threep_s_gtDt = threeps[7]

        print "Read three-point functions from HDF5 files for tsink " + str( ts )

        # Subtract average over directions from gtDt

        threep_mom[ imom ] = threep_gtDt - \
                             0.25 * ( threep_gtDt \
                                      + threep_gxDx \
                                      + threep_gyDy \
                                      + threep_gzDz )

        if particle == "kaon":

            threep_s_mom[ imom ] = threep_s_gtDt - \
                                   0.25 * ( threep_s_gtDt \
                                            + threep_s_gxDx \
                                            + threep_s_gyDy \
                                            + threep_s_gzDz )

        # End loop over momenta

    threep_mom = np.array( threep_mom )

    threep_s_mom = np.array( threep_s_mom )

    threep = np.average( threep_mom, axis=0 )

    # Jackknife
    # threep_jk[ b, t ]
    
    threep_jk = fncs.jackknife( threep, binSize )

    ######################
    # Write output files #
    ######################

    threep_outFilename = output_template.replace( "*", "threep_u_tsink" + str( ts ) )

    rw.writeDataFile( threep_outFilename, threep_jk )

    if particle == "kaon":

        ###########################
        # Repeat for strange part #
        ###########################

        # Subtract average over directions from gtDt

        threep_s = threep_s_gtDt - 0.25 * ( threep_s_gtDt + threep_s_gxDx + threep_s_gyDy + threep_s_gzDz )

        # Jackknife
        # threep_s_jk[ b, t ]
    
        threep_s_jk = fncs.jackknife( threep_s, binSize )

        ######################
        # Write output files #
        ######################

        threep_s_outFilename = output_template.replace( "*", "threep_s_tsink" + str( ts ) )

        rw.writeDataFile( threep_s_outFilename, threep_s_jk )

        print "Wrote output files for tsink " + str( ts )

# End loop over tsink

