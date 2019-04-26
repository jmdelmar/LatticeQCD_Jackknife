import numpy as np
import argparse as argp
from scipy.optimize import curve_fit
import functions as fncs
import readWrite as rw
import physQuants as pq
import lqcdjk_fitting as fit

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

parser.add_argument( "threep_tokens", action='store', \
                     type=lambda s: [str(token) for token in s.split(',')], \
                     help="Comma seperated list of filename tokens. CPU: part before tsink, part before momentum boost components. GPU: part before momentum boost components, part after momentum boost; * for configuration number." )

parser.add_argument( "twop_filename", action='store', type=str )

parser.add_argument( "mEff_filename", action='store', type=str )

parser.add_argument( "mEff_fit_start", action='store', type=int )

parser.add_argument( "mEff_fit_end", action='store', type=int )

parser.add_argument( 't_sink', action='store', \
                     help="Comma seperated list of t sink's", \
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "-o", "--output_template", action='store', type=str, default="./*.dat" )

parser.add_argument( "-f", "--data_format", action='store', help="Data format. Should be 'gpu' or 'cpu'.", type=str, default="gpu" )

args = parser.parse_args()

#########
# Setup #
#########

threep_tokens = args.threep_tokens

twop_filename = args.twop_filename

mEff_filename = args.mEff_filename

mEff_fitStart = args.mEff_fit_start

mEff_fitEnd = args.mEff_fit_end

tsink = args.t_sink

tsinkNum = len( tsink )

ts_range_str = "tsink" + str(tsink[0]) + "_" + str(tsink[-1])

output_template = args.output_template

dataFormat = args.data_format

# Check inputs

assert dataFormat in format_list, "Error: Data format not supported. " \
    + "Supported particles: " + str( format_list )

# Set timestep and bin number from effective mass file

timestepNum_twop, binNum_twop = rw.detTimestepAndConfigNum( twop_filename )

for ts in tsink:

    threep_filename = threep_tokens[0] + str(ts) + threep_tokens[-1]

    timestepNum_threep, binNum_threep = rw.detTimestepAndConfigNum( threep_filename )

    assert binNum_threep == binNum_twop, \
        "ERROR: Number of bins does not match for " \
        + "two-, and three-point function files.\n"

    assert timestepNum_threep == timestepNum_twop, \
        "ERROR: Number of timesteps does not match for " \
        + "two-, and three-point function files.\n"



timestepNum_mEff, binNum_mEff = rw.detTimestepAndConfigNum( mEff_filename )

binNum = 0

if binNum_twop == binNum_twop:

    binNum = binNum_threep

else:

    print( "ERROR: Number of bins does not match for effective mass " \
        + " and two-point function files.\n" )

    exit()

###################
# Read data files #
###################

# Read from file
# twop[ b, t ]

twop_jk = rw.readDataFile( twop_filename, timestepNum_twop, binNum )

twop_err = np.std( twop_jk, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

# Read from file
# threep[ts][ b, t ]

threep_jk = []

threep_err = []

for ts in tsink:

    threep_filename = threep_tokens[0] + str(ts) + threep_tokens[-1]

    threep_jk.append( rw.readDataFile( threep_filename, timestepNum_threep, binNum ) )

    threep_err.append( np.std( threep_jk, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) ) )

# Read from file
# mEff[ b, t ]

mEff = rw.readDataFile( mEff_filename, timestepNum_mEff, binNum )

# mEff_err[ t ]

mEff_err = np.std( mEff, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

# Fit
# mEff_fit [ b ]

mEff_fit = np.zeros( binNum )

for b in range( binNum ):

    mEff_fit[ b ] = np.polyfit( range( mEff_fitStart, mEff_fitEnd + 1 ), \
                                mEff[ b, mEff_fitStart : mEff_fitEnd + 1 ], \
                                0, w=mEff_err[ mEff_fitStart : mEff_fitEnd + 1 ] )

print( "Fit effective mass" )

# Average over bins

mEff_fit_avg = np.average( mEff_fit )

mEff_fit_err = np.std( mEff_fit ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

# Write fitted effective mass file

mEff_outputFilename = output_template.replace( "*", "mEff_fit" )

rw.writeFitDataFile( mEff_outputFilename, mEff_fit_avg, mEff_fit_err, mEff_fitStart, mEff_fitEnd )

# End loop over tsink

##################
# Two-state Fit  #
##################

for twop_rangeStart in range( 0, 6 ):

    for twop_rangeEnd in range(10,20):

        # fitParams[ b, param ]
        
        fitParams, chiSq = fit.twoStateFit_twop( twop_jk, \
                                                 twop_rangeStart, \
                                                 twop_rangeEnd )

        c0 = fitParams[ :, 0 ]
        c1 = fitParams[ :, 1 ]
        E0 = fitParams[ :, 2 ]
        E1 = fitParams[ :, 3 ]

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
                
            curve_err = np.std( curve, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )
                
            fitParams_avg = np.average( fitParams, axis=0 )

            fitParams_err = np.std( fitParams, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

            chiSq_avg = np.average( chiSq, axis=0 )

            chiSq_err = np.std( chiSq, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

            avgX_avg = np.average( avgX )

            avgX_err = np.std( avgX ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )
    
            # Write output file

            range_str = "2s" + str( twop_rangeStart ) \
                        + ".2e" + str( twop_rangeEnd ) \
                        + ".3n" + str( threep_neglect )

            print( range_str + ":" )

            check = ""

            if mEff_fit_err < (fitParams_avg[6] - fitParams_avg[5])/2:

                check = "True"

            else: 
                
                check = "False"

            print( "dm < (E1-E0)/2: " + check )

            avgXOutputFilename \
                = output_template.replace( "*", \
                                           "avgX_twoStateFit_" \
                                           + range_str + "_" + ts_range_str )
                
            rw.writeFitDataFile( avgXOutputFilename, avgX_avg, avgX_err, 0, 0 )

            chiSqOutputFilename \
                = output_template.replace( "*", \
                                           "avgX_twoStateFit_chiSq_" \
                                           + range_str + "_" + ts_range_str )
            
            rw.writeFitDataFile( chiSqOutputFilename, chiSq_avg, chiSq_err, 0, 0 )

            avgXParamsOutputFilename \
                = output_template.replace( "*", \
                                           "avgX_twoStateFitParams_" \
                                           + range_str + "_" + ts_range_str )

            rw.writeTSFParamsFile( avgXParamsOutputFilename, fitParams_avg, fitParams_err )

            for ts in range( tsinkNum ):

                curveOutputFilename \
                    = output_template.replace( "*", \
                                               "avgX_twoStateFit_curve_tsink" \
                                               + str( tsink[ ts ] ) + "_" \
                                               + range_str + "_" + ts_range_str )

                rw.writeAvgDataFile_wX( curveOutputFilename, t_i[ ts ], curve_avg[ ts ], curve_err[ ts ] )

"""
for twop_rangeStart in range( 5, 6 ):

    for twop_rangeEnd in range(14,15):

        for threep_neglect in 2,3:

            # fitParams[ b, param ]

            fitParams, chiSq = fit.twoStateFit( twop_jk, twop_err, \
                                                 twop_rangeStart, twop_rangeEnd, \
                                                 threep_jk, threep_err, \
                                                 threep_neglect, tsink )

            a00 = fitParams[ :, 0 ]
            
            a01 = fitParams[ :, 1 ]

            a11 = fitParams[ :, 2 ]
          
            c0 = fitParams[ :, 3 ]
            
            c1 = fitParams[ :, 4 ]
        
            E0 = fitParams[ :, 5 ]
                
            E1 = fitParams[ :, 6 ]

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

                #avgX[ b ] = -4.0 / 3.0 / mEff_fit[ b ] * Z * a00[ b ] / c0[ b ]

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
                
            curve_err = np.std( curve, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )
                
            fitParams_avg = np.average( fitParams, axis=0 )

            fitParams_err = np.std( fitParams, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

            chiSq_avg = np.average( chiSq, axis=0 )

            chiSq_err = np.std( chiSq, axis=0 ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )

            avgX_avg = np.average( avgX )

            avgX_err = np.std( avgX ) * float( binNum - 1 ) / np.sqrt( float( binNum ) )
    
            # Write output file

            range_str = "2s" + str( twop_rangeStart ) \
                        + ".2e" + str( twop_rangeEnd ) \
                        + ".3n" + str( threep_neglect )

            avgXOutputFilename \
                = output_template.replace( "*", \
                                           "avgX_twoStateFit_" \
                                           + range_str + "_" + ts_range_str )
                
            rw.writeFitDataFile( avgXOutputFilename, avgX_avg, avgX_err, 0, 0 )

            chiSqOutputFilename \
                = output_template.replace( "*", \
                                           "avgX_twoStateFit_chiSq_" \
                                           + range_str + "_" + ts_range_str )
            
            rw.writeFitDataFile( chiSqOutputFilename, chiSq_avg, chiSq_err, 0, 0 )

            avgXParamsOutputFilename \
                = output_template.replace( "*", \
                                           "avgX_twoStateFitParams_" \
                                           + range_str + "_" + ts_range_str )

            rw.writeTSFParamsFile( avgXParamsOutputFilename, fitParams_avg, fitParams_err )

            for ts in range( tsinkNum ):

                curveOutputFilename \
                    = output_template.replace( "*", \
                                               "avgX_twoStateFit_curve_tsink" \
                                               + str( tsink[ ts ] ) + "_" \
                                               + range_str + "_" + ts_range_str )

                rw.writeAvgDataFile_wX( curveOutputFilename, t_i[ ts ], curve_avg[ ts ], curve_err[ ts ] )
"""

