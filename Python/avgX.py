import numpy as np
import argparse as argp
from scipy.optimize import curve_fit
import functions as fncs
import readWrite as rw
import physQuants as pq
import lqcdjk_fitting as fit

Z = 1.0

#twopFitStart = 10

#twopFitEnd = 30

particle_list = [ "pion", "kaon", "nucleon" ]

format_list = [ "gpu", "cpu" ]

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Calculate quark momentum fraction <x>" )

parser.add_argument( "threep_dir", action='store', type=str )

parser.add_argument( "threep_tokens", action='store', \
                     type=lambda s: [str(token) for token in s.split(',')], \
                     help="Comma seperated list of filename tokens. " \
                     + "CPU: part before tsink, part before momentum " \
                     + "boost components. GPU: part before momentum " \
                     + "boost components, part after momentum boost; " \
                     + "* for configuration number." )

parser.add_argument( "twop_dir", action='store', type=str )

parser.add_argument( "twop_template", action='store', type=str )

parser.add_argument( "mEff_filename", action='store', type=str )

parser.add_argument( "mEff_fit_start", action='store', type=int )

parser.add_argument( "mEff_fit_end", action='store', type=int )

parser.add_argument( "particle", action='store', \
                     help="Particle to calculate gA for. " \
                     + "Should be 'pion' or 'kaon'.", type=str )

parser.add_argument( 't_sink', action='store', \
                     help="Comma seperated list of t sink's", \
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "mom_squared", action='store', type=int )

parser.add_argument( "-o", "--output_template", action='store', \
                     type=str, default="./*.dat" )

parser.add_argument( "-tsf", "--two_state_fit", action='store_true', \
                     help="Performs the two-state fit if supplied" )

parser.add_argument( "-f", "--data_format", action='store', \
                     help="Data format. Should be 'gpu' or 'cpu'.", \
                     type=str, default="gpu" )

parser.add_argument( "-c", "--config_list", action='store', \
                     type=str, default="" )

args = parser.parse_args()

#########
# Setup #
#########

threepDir = args.threep_dir

twopDir = args.twop_dir

threep_tokens = args.threep_tokens

twop_template = args.twop_template

mEff_filename = args.mEff_filename

mEff_fitStart = args.mEff_fit_start

mEff_fitEnd = args.mEff_fit_end

particle = args.particle

tsink = args.t_sink

tsinkNum = len( tsink )

ts_range_str = "tsink" + str(tsink[0]) + "_" + str(tsink[-1])

momSq = args.mom_squared

output_template = args.output_template

dataFormat = args.data_format

# Check inputs

assert particle in particle_list, \
    "Error: Particle not supported. " \
    + "Supported particles: " + str( particle_list )

assert dataFormat in format_list, \
    "Error: Data format not supported. " \
    + "Supported particles: " + str( format_list )

# Get configurations from given list or from given 
# threep directory if list not given

configList = fncs.getConfigList( args.config_list, threepDir )

configNum = len( configList )

# Set timestep and bin number from effective mass file

timestepNum, binNum = rw.detTimestepAndConfigNum( mEff_filename )

assert configNum % binNum == 0, "Number of configurations " \
    + str( configNum ) + " not evenly divided by number of bins " \
    + str( binNum ) + " in effective mass file " \
    + mEff_filename + ".\n"

binSize = int( configNum / binNum )

####################
# Effective masses #
####################

# Read from file
# mEff[ b, t ]

mEff = rw.readDataFile( mEff_filename, timestepNum, binNum )

# Read error from file
# mEff_err[ t ]

mEff_err = np.std( mEff, axis=0 ) \
           * float( binNum - 1 ) \
           / np.sqrt( float( binNum ) )

# Fit
# mEff_fit [ b ]

mEff_fit = np.zeros( binNum )

for b in range( binNum ):

    mEff_fit[ b ] = np.polyfit( range( mEff_fitStart, \
                                       mEff_fitEnd + 1 ), \
                                mEff[ b, \
                                      mEff_fitStart \
                                      : mEff_fitEnd + 1 ], 0, \
                                w=mEff_err[ mEff_fitStart \
                                               : mEff_fitEnd + 1 ] )

print( "Fit effective mass" )

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

print( "Read two-point functions from HDF5 files" )

# Jackknife
# twop_jk[ b, t ]

twop_jk = fncs.jackknife( twop, binSize )

twop_err = np.std( twop_jk, axis=0 ) \
           * float( binNum - 1 ) / np.sqrt( float( binNum ) )

threep_jk = []

threep_err = []

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

        print( "Read three-point functions from HDF5 files for tsink " + str( ts ) )

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

    threep = np.average( threep_mom, axis=0 )

    # Jackknife
    # threep_jk[ ts ][ b, t ]
    
    threep_jk.append( fncs.jackknife( threep, binSize ) )

    threep_err.append( np.std( threep_jk[ -1 ], axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) ) )

    #################
    # Calculate <x> #
    #################

    #twopFitParams = fit.twopFit( twop_jk, twopFitStart, twopFitEnd )

    #G = np.repeat( twopFitParams[ :, 0 ], ts + 1 ).reshape( binNum, ts + 1 )

    #E = np.repeat( twopFitParams[ :, 1 ], ts + 1 ).reshape( binNum, ts + 1 )

    #mEff_fit_cp = np.repeat( mEff_fit, ts + 1 ).reshape( binNum, ts + 1 )

    #avgX = -4.0/3.0/mEff_fit_cp * threep_jk[ -1 ] / fit.twopExp( ts, G, E )

    avgX = Z * pq.calcAvgX( threep_jk[ -1 ], twop_jk[ :, ts ], mEff_fit )

    # Average over bins

    avgX_avg = np.average( avgX, axis=0 )

    avgX_err = np.std( avgX, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

    ######################
    # Write output files #
    ######################

    # <x>
    
    avgX_outFilename = output_template.replace( "*", "avgX_u_tsink" + str( ts ) )

    rw.writeAvgDataFile( avgX_outFilename, avgX_avg, avgX_err )

    ###############
    # Fit plateau #
    ###############

    fitStart = [ ts // 2 - 1, ts // 2 - 2, ts // 2 - 3 ]

    fitEnd = [ ts // 2 + 1, ts // 2 + 2, ts // 2 + 3 ]

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

        avgX_fit_outFilename = output_template.replace( "*", \
                                                        "avgX_u_fit_" \
                                                        "tsink" + str( ts ) \
                                                        + "_" + str( fitStart[ irange ] ) \
                                                        + "_" + str( fitEnd[ irange ] ) )

        rw.writeFitDataFile( avgX_fit_outFilename, avgX_fit_avg, avgX_fit_err, fitStart[ irange ], fitEnd[ irange ] )

    # End loop over fit ranges

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

        rw.writeAvgDataFile( avgX_s_outFilename, avgX_s_avg, avgX_s_err )

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

            rw.writeFitDataFile( avgX_s_fit_outFilename, avgX_s_fit_avg, avgX_s_fit_err, fitStart[ irange ], fitEnd[ irange ] )

        # End loop over fit ranges

    # End if kaon
    
# End loop over tsink

if args.two_state_fit:

    print( "Will perform the two-state fit" )

    ##################
    # Two-state Fit  #
    ##################

    twop_rangeEnd = 20

    goodFit = False

    for twop_rangeStart in range( 1, 4 ):

        # fitParams[ b, param ]

        fitParams, chiSq = fit.twoStateFit_twop( twop_jk, \
                                                 twop_rangeStart, \
                                                 twop_rangeEnd )

        c0 = fitParams[ :, 0 ]
        c1 = fitParams[ :, 1 ]
        E0 = fitParams[ :, 2 ]
        E1 = fitParams[ :, 3 ]

        E0_avg = np.average( E0 )

        E0_err = np.std( E0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

        # Check that the two-state fit is good,
        # i.e. dm < Delta m / 2

        if E0_err < np.abs( mEff_fit_avg - E0_avg ) / 2 \
           or mEff_fit_err < np.abs( mEff_fit_avg - E0_avg ) / 2:
            
            goodFit = True

            break

    # End loop over fit end

    assert goodFit == True, "Error: a good two-state fit which satisfies " \
        "the condition dm < Delta m / 2 was not found."

    # Calculate twop curve

    curve = np.zeros( ( binNum, 50 ) )

    t_s= np.linspace( twop_rangeStart, twop_rangeEnd, 50 )

    for b in range( binNum ):

        for t in range( t_s.shape[ -1 ] ):

            curve[ b, t ] = fit.twoStateTwop( t_s[ t ], c0[ b ], c1[ b ], \
                                              E0[ b ], E1[ b ] )

        # End loop over tsink
    # End loop over bins

    chiSq_avg = np.average( chiSq, axis=0 )

    chiSq_err = np.std( chiSq, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )
                
    curve_avg = np.average( curve, axis=0 )
                
    curve_err = np.std( curve, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

    range_str = "2s" + str( twop_rangeStart ) \
                + ".2e" + str( twop_rangeEnd )

    curveOutputFilename \
        = output_template.replace( "*", \
                                   "twop_twoStateFit_curve_" \
                                   + range_str )

    rw.writeAvgDataFile_wX( curveOutputFilename, t_s, curve_avg, curve_err )

    chiSqOutputFilename \
        = output_template.replace( "*", \
                                   "avgX_twoStateFit_twop_chiSq_" \
                                   + range_str )
            
    rw.writeFitDataFile( chiSqOutputFilename, chiSq_avg, chiSq_err, 0, 0 )

    for threep_neglect in 2,3:

        fitParams, chiSq = fit.twoStateFit_threep( threep_jk, \
                                                   threep_neglect, \
                                                   tsink, E0, E1 )

        a00 = fitParams[ :, 0 ]
        a01 = fitParams[ :, 1 ]
        a11 = fitParams[ :, 2 ]
          
        fitParams = np.stack( ( a00, a01, a11, c0, c1, E0, E1 ), axis=1 )

        # Calculate curve with constant tsink

        curve = np.zeros( ( binNum, tsinkNum, 50 ) )

        avgX = np.zeros( binNum )
        
        t_i= np.zeros( ( tsinkNum, 50 ) )
            
        for b in range( binNum ):

            for ts in range( tsinkNum ):
                    
                t_i[ ts, : ] = np.linspace( threep_neglect, tsink[ ts ] - threep_neglect, 50 )

                for t in range( t_i.shape[ -1 ] ):

                    curve[ b, ts, t ] = -4.0 / 3.0 / mEff_fit[ b ] * Z \
                                        * fit.twoStateThreep( t_i[ ts, t ], tsink[ ts ], \
                                                              a00[ b ], a01[ b ], a11[ b ], \
                                                              E0[ b ], E1[ b ] ) \
                                        / fit.twoStateTwop( tsink[ ts ], c0[ b ], c1[ b ], \
                                                            E0[ b ], E1[ b ] )

                # End loop over insertion time
            # End loop over tsink

            avgX[ b ] = -4.0 / 3.0 / mEff_fit[ b ] * Z * a00[ b ] / c0[ b ]

            # Write curve with constant insertion time = tsink / 2
        
            #for b in range( binNum ):
                    
            #t_s = np.linspace( tsink[ 0 ] - 2, tsink[ -1 ] + 2, 50 )

            #for t in range( t_s.shape[ 0 ] ):
                    
            #curve[ b, t ] = -4.0 / 3.0 / E0[ b ] * Z * \
                #fit.twoStateThreep( t_s[ t ] / 2, t_s[ t ], \
                #a00[ b ], a01[ b ], a11[ b ], \
                #E0[ b ], E1[ b ] ) \
                #/ fit.twoStateTwop( t_s[ t ], c0[ b ], c1[ b ], \
                #E0[ b ], E1[ b] )

        # End loop over bins
        
        # Average over bins
                    
        curve_avg = np.average( curve, axis=0 )
                
        curve_err = np.std( curve, axis=0 ) \
                    * float( binNum - 1 ) \
                    / np.sqrt( float( binNum ) )
                
        fitParams_avg = np.average( fitParams, axis=0 )

        fitParams_err = np.std( fitParams, axis=0 ) \
                        * float( binNum - 1 ) \
                        / np.sqrt( float( binNum ) )

        chiSq_avg = np.average( chiSq, axis=0 )

        chiSq_err = np.std( chiSq, axis=0 ) \
                    * float( binNum - 1 ) \
                    / np.sqrt( float( binNum ) )
                
        avgX_avg = np.average( avgX )

        avgX_err = np.std( avgX ) \
                   * float( binNum - 1 ) \
                   / np.sqrt( float( binNum ) )
    
        # Write output file

        range_str = "2s" + str( twop_rangeStart ) \
                    + ".2e" + str( twop_rangeEnd ) \
                    + ".3n" + str( threep_neglect )

        avgXOutputFilename \
            = output_template.replace( "*", \
                                       "avgX_twoStateFit_" \
                                       + range_str + "_" \
                                       + ts_range_str )
                
        rw.writeFitDataFile( avgXOutputFilename, \
                             avgX_avg, avgX_err, 0, 0 )
            
        chiSqOutputFilename \
            = output_template.replace( "*", \
                                       "avgX_twoStateFit_threep_chiSq_" \
                                       + range_str + "_" \
                                       + ts_range_str )
        
        rw.writeFitDataFile( chiSqOutputFilename, \
                             chiSq_avg, chiSq_err, 0, 0 )

        avgXParamsOutputFilename \
            = output_template.replace( "*", \
                                       "avgX_twoStateFitParams_" \
                                       + range_str + "_" \
                                       + ts_range_str )

        rw.writeTSFParamsFile( avgXParamsOutputFilename, \
                               fitParams_avg, fitParams_err )

        for ts in range( tsinkNum ):

            curveOutputFilename \
                = output_template.replace( "*", \
                                           "avgX_twoStateFit_curve_tsink" \
                                           + str( tsink[ ts ] ) + "_" \
                                           + range_str + "_" \
                                           + ts_range_str )

            rw.writeAvgDataFile_wX( curveOutputFilename, \
                                    t_i[ ts ], curve_avg[ ts ], \
                                    curve_err[ ts ] )

    # End loop over number of neglected three-point functions

# CJL To Do: Add kaon two-state fit
