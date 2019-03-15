import numpy as np
import argparse as argp
from scipy.optimize import curve_fit
import functions as fncs
import readWrite as rw
import physQuants as pq

Z = 1.0

L=32

twopFitStart = 10

twopFitEnd = 30

particle_list = [ "pion", "kaon", "nucleon" ]

format_list = [ "gpu", "cpu" ]

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Calculate quark momentum fraction <x>" )

parser.add_argument( "threep_dir", action='store', type=str )

parser.add_argument( "threep_pre_template", action='store', type=str, \
                     help="Part of filename before momentum boost components with '*' as wildcard." )

parser.add_argument( "threep_suf_template", action='store', type=str, \
                     help="Part of filename after momentum boost components with '*' as wildcard." )

parser.add_argument( "twop_dir", action='store', type=str )

parser.add_argument( "twop_template", action='store', type=str )

parser.add_argument( "mEff_filename", action='store', type=str )

parser.add_argument( "mEff_fit_start", action='store', type=int )

parser.add_argument( "mEff_fit_end", action='store', type=int )

parser.add_argument( "particle", action='store', help="Particle to calculate gA for. Should be 'pion' or 'kaon'.", type=str )

parser.add_argument( 't_sink', action='store', \
                     help="Comma seperated list of t sink's", \
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "mom_squared", action='store', type=int )

parser.add_argument( "-o", "--output_template", action='store', type=str, default="./*.dat" )

parser.add_argument( "-f", "--data_format", action='store', help="Data format. Should be 'gpu' or 'cpu'.", type=str, default="gpu" )

parser.add_argument( "-c", "--config_list", action='store', type=str, default="" )

args = parser.parse_args()

#########
# Setup #
#########

threepDir = args.threep_dir

twopDir = args.twop_dir

threep_pre_template = args.threep_pre_template

threep_suf_template = args.threep_suf_template

twop_template = args.twop_template

mEff_filename = args.mEff_filename

mEff_fitStart = args.mEff_fit_start

mEff_fitEnd = args.mEff_fit_end

particle = args.particle

tsink = args.t_sink

momSq = args.mom_squared

tsinkNum = len( tsink )

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

# Write fitted effective mass file

mEff_outputFilename = output_template.replace( "*", "mEff_fit" )

rw.writeFitDataFile( mEff_outputFilename, mEff_fit_avg, mEff_fit_err, mEff_fitStart, mEff_fitEnd )

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

twop_err = np.std( twop_jk, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

threep_jk = []

threep_err = []

momList = []

if momSq == 2:

    momList = [ +1, +1,  0], \
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
              [  0, -1, -1]

elif momSq == 3:

    momList = [ +1, +1, +1], \
              [ -1, +1, +1], \
              [ +1, -1, +1], \
              [ +1, +1, -1], \
              [ +1, -1, -1], \
              [ -1, +1, -1], \
              [ -1, -1, +1], \
              [ -1, -1, -1]

momBoostNum = len( momList )

pSq = (2*np.pi/L)**2 * momSq

energy = np.sqrt( mEff_fit**2 + pSq )                                                                                                                                                              
for ts in tsink:
    
    threep_mom = [ [] for imom in range( momBoostNum ) ]

    threep_s_mom = [ [] for imom in range( momBoostNum ) ]

    for imom in range( momBoostNum ):

        if dataFormat == "cpu":
        
            threep_template = threep_pre_template + \
                              fncs.signToString( momList[imom][0] ) \
                              + str(momList[imom][0]) + "_" \
                              + fncs.signToString( momList[imom][1] ) \
                              + str(momList[imom][1]) + "_" \
                              + fncs.signToString( momList[imom][2] ) \
                              + str(momList[imom][2])

        else:

            threep_template = threep_pre_template + \
                              fncs.signToString( momList[imom][0] ) \
                              + str(momList[imom][0]) + "_" \
                              + fncs.signToString( momList[imom][1] ) \
                              + str(momList[imom][1]) + "_" \
                              + fncs.signToString( momList[imom][2] ) \
                              + str(momList[imom][2]) \
                              + threep_suf_template

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

    threep = fncs.combineMomBoosts( threep_mom, momSq )

    # Jackknife
    # threep_jk[ ts ][ b, t ]
    
    threep_jk.append( fncs.jackknife( threep, binSize ) )

    threep_err.append( np.std( threep_jk[ -1 ], axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) ) )

    #################
    # Calculate <x> #
    #################

    twopFitParams = fncs.twopFit( twop_jk, twopFitStart, twopFitEnd )

    G = np.repeat( twopFitParams[ :, 0 ], ts + 1 ).reshape( binNum, ts + 1 )

    E = np.repeat( twopFitParams[ :, 1 ], ts + 1 ).reshape( binNum, ts + 1 )

    mEff_fit_cp = np.repeat( mEff_fit, \
                             ts + 1 ).reshape( binNum, ts + 1 )

    pSq_cp = np.repeat( pSq, \
                        binNum * (ts + 1) ).reshape( binNum, ts + 1 )

    energy_cp = np.repeat( energy, \
                           binNum * (ts + 1) ).reshape( binNum, ts + 1 )

    preFactor = -2.0/mEff_fit_cp**2 * energy_cp * ( energy_cp + mEff_fit_cp ) \
                / ( 3 * energy_cp**2 + pSq_cp )

    avgX = preFactor * threep_jk[-1] / fncs.twopExp( ts, G, E )

    pSq_cp = np.repeat( pSq, \
                        binNum * (ts + 1) ).reshape( binNum, ts + 1 )
                                                                                                                                                             
    energy_cp = np.repeat( energy, \
                           ts + 1 ).reshape( binNum, ts + 1 )

    preFactor = -2.0/mEff_fit_cp**2 * energy_cp * ( energy_cp + mEff_fit_cp ) \
                / ( 3 * energy_cp**2 + pSq_cp )

    avgX = preFactor * threep_jk[-1][:,:ts+1] / fncs.twopExp( ts, G, E )

    #avgX = -4.0/3.0/mEff_fit_cp * threep_jk[ -1 ] / fncs.twopExp( ts, G, E )

    #avgX = Z * pq.calcAvgX( threep_jk[ -1 ], twop_jk[ :, ts ], mEff_fit_avg )

    # Average over bins

    avgX_avg = np.average( avgX, axis=0 )

    avgX_err = np.std( avgX, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

    ######################
    # Write output files #
    ######################

    # <x>
    
    avgX_outFilename = output_template.replace( "*", "avgX_u_tsink" + str( ts ) )

    rw.writeAvgDataFile( avgX_outFilename, avgX_avg, avgX_err )

    print "Wrote output files for tsink " + str( ts )

    if particle == "kaon":

        ###########################
        # Repeat for strange part #
        ###########################

        threep_s = fncs.combineMomBoosts( threep_s_mom, momSq )

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

        rw.writeAvgDataFile( avgX_s_outFilename, avgX_s_avg, avgX_s_err )

# End loop over tsink

