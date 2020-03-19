import sys
import time
import numpy as np
import argparse as argp
from scipy.optimize import curve_fit
import functions as fncs
import mpi_functions as mpi_fncs
import readWrite as rw
import physQuants as pq
import lqcdjk_fitting as fit
from mpi4py import MPI

np.set_printoptions(threshold=sys.maxsize)

L = 32.0

particle_list = [ "pion", "kaon", "nucleon" ]

format_list = [ "gpu", "cpu" ]

moment_list = [ 1, 2, 3 ]

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Calculate a mellin moment up to "
                              + " the third order" )

parser.add_argument( "threep_dir", action='store', type=str )

parser.add_argument( "threep_tokens", action='store',
                     type=lambda s: [str(token) for token in s.split(',')],
                     help="Comma seperated list of filename tokens. "
                     + "CPU: part before tsink, part before momentum "
                     + "boost components. GPU: part before momentum "
                     + "boost components, part after momentum boost; "
                     + "* for configuration number." )

parser.add_argument( "twop_dir", action='store', type=str )

parser.add_argument( "twop_template", action='store', type=str )

parser.add_argument( "fit_range_end", action='store', type=int )

parser.add_argument( "particle", action='store',
                     help="Particle to calculate gA for. "
                     + "Should be 'pion' or 'kaon'.", type=str )

parser.add_argument( 't_sink', action='store',
                     help="Comma seperated list of t sink's",
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "mom_squared", action='store', type=int )

parser.add_argument( "moment", action='store', type=int )

parser.add_argument( "binSize", action='store', type=int )

parser.add_argument( "-o", "--output_template", action='store',
                     type=str, default="./*.dat" )

parser.add_argument( "-sn", "--source_number", action='store', type=int,
                     help="Number of sources correlators were averaged "
                     + "over", default=16 )

parser.add_argument( "-tsf", "--two_state_fit", action='store_true',
                     help="Performs the two-state fit if supplied" )

parser.add_argument( "--tsf_fit_start", action='store', type=int,
                     help="If given, will perform two-state fit on effective "
                     + "mass starting at given t value, otherwise, will "
                     + "use lowest t value which satisfies condition." )

parser.add_argument( "--twop_fit_start", action='store', type=int,
                     help="If given, will perform one- or two-state fit on "
                     + "two-point functions starting at given t value, "
                     + "otherwise, will use lowest t value which satisfies "
                     + "condition." )

parser.add_argument( "--plat_fit_start", action='store', type=int,
                     help="If given, will perform plateau fit on effective "
                     + "mass starting at given t value, otherwise, will "
                     + "use lowest t value which satisfies condition." )

parser.add_argument( "-f", "--data_format", action='store',
                     help="Data format. Should be 'gpu' or 'cpu'.",
                     type=str, default="gpu" )

parser.add_argument( "-c", "--config_list", action='store',
                     type=str, default="" )

args = parser.parse_args()

#########
# Setup #
#########

# Set MPI values

mpi_confs_info = mpi_fncs.lqcdjk_mpi_init()

comm = mpi_confs_info[ 'comm' ]
rank = mpi_confs_info[ 'rank' ]

# Input directories and filename templates

threepDir = args.threep_dir

twopDir = args.twop_dir

threep_tokens = args.threep_tokens

twop_template = args.twop_template

# Last point to fit

rangeEnd = args.fit_range_end

# Info on what to analyze

particle = args.particle

tsink = args.t_sink
tsinkNum = len( tsink )
ts_range_str = "tsink" + str(tsink[0]) + "_" + str(tsink[-1])

# Other info

output_template = args.output_template

tsf_fitStart = args.tsf_fit_start
twop_fitStart = args.twop_fit_start
plat_fitStart = args.plat_fit_start

if tsf_fitStart and plat_fitStart:

    checkFit = False

else:

    checkFit = True

srcNum = args.source_number

tsf = args.two_state_fit

dataFormat = args.data_format

momSq = args.mom_squared

moment = args.moment

# Get configurations from given list or from given 
# threep directory if list not given

mpi_confs_info[ 'configList' ] = fncs.getConfigList( args.config_list, 
                                                     threepDir )
mpi_confs_info[ 'configNum' ] = len( mpi_confs_info[ 'configList' ] )
mpi_confs_info[ 'binSize' ] = args.binSize

# Set mpi configuration information

mpi_fncs.lqcdjk_mpi_confs_info( mpi_confs_info )

binNum = mpi_confs_info[ 'binNum_glob' ]
recvCount = mpi_confs_info[ 'recvCount' ]
recvOffset = mpi_confs_info[ 'recvOffset' ]

# Check inputs

assert particle in particle_list, \
    "Error: Particle not supported. " \
    + "Supported particles: " + str( particle_list )

if particle == "pion":

    flavNum = 1

else:

    flavNum = 2

# Set string for up and strange quark.
# If pion, will not access strange.

flav_str = [ "u", "s" ]

assert dataFormat in format_list, \
    "Error: Data format not supported. " \
    + "Supported particles: " + str( format_list ) \

assert moment in moment_list, \
    "Error: Moment order not supported. " \
    + "Supported moments: " + str( moment_list )


if moment == 1:

    Z = 1.123
    moment_str = "avgX"

elif moment == 2:

    Z = 1.0
    moment_str = "avgX2"

elif moment == 3:

    Z = 1.0
    moment_str = "avgX3"

# Read momentum list

momList = rw.readMomentaList( twopDir, twop_template,
                              mpi_confs_info[ 'configList_loc' ][ 0 ], particle,
                              srcNum, momSq, dataFormat, mpi_confs_info )

momBoostNum = len( momList )

############################
# Read Two-point Functions #
############################

# Zero momentum two-point functions
# twop[ c, t ]

twop = rw.readTwopFile_zeroQ( twopDir, mpi_confs_info[ 'configList_loc' ], 
                              mpi_confs_info[ 'configNum' ],
                              twop_template, srcNum, 0, dataFormat,
                              mpi_confs_info )

# Time dimension length

T = twop.shape[ -1 ]

# Time dimension length after fold

T_fold = T // 2 + 1

##########################################
# Jackknife and fold two-point functions #
##########################################

if mpi_confs_info[ 'binNum_loc' ]:

    twop_jk_loc = fncs.jackknifeBinSubset( twop, mpi_confs_info[ 'binSize' ], 
                                           mpi_confs_info[ 'binList_loc' ] )

    # twop_fold[ b, t ]

    twop_fold_loc = fncs.fold( twop_jk_loc )

    # mEff[ b, t ]

    mEff_loc = pq.mEffFromSymTwop( twop_fold_loc )

else:

    twop_jk_loc = np.array( [] )

    twop_fold_loc = np.array( [] )

    mEff_loc = np.array( [] )

##################
# Effective mass #
##################

twop_fold = np.zeros( ( binNum, T_fold ) )
mEff = np.zeros( ( binNum, T_fold ) )

comm.Allgatherv( twop_fold_loc, 
                 [ twop_fold, 
                   recvCount * T_fold,
                   recvOffset * T_fold, 
                   MPI.DOUBLE ] )
comm.Allgatherv( mEff_loc, 
                 [ mEff, recvCount * T_fold,
                   recvOffset * T_fold, 
                   MPI.DOUBLE ] )

if rank == 0:

    # mEff_avg[ t ]

    mEff_avg = np.average( mEff, axis=0 )
    mEff_err = fncs.calcError( mEff, binNum )

    avgOutputFilename = rw.makeFilename( output_template, "mEff_avg" )
    rw.writeAvgDataFile( avgOutputFilename, mEff_avg, mEff_err )

# Fit the effective mass and two-point functions 

try:
    
    fitResults = fit.mEffTwopFit( mEff, twop_fold,
                                  rangeEnd, 0, L, tsf,
                                  mpi_confs_info,
                                  tsf_t_low_range=[tsf_fitStart],
                                  plat_t_low_range=[plat_fitStart],
                                  checkFit=checkFit )
    
except fit.lqcdjk_BadFitError as error:
        
    mpi_fncs.mpiPrintErr( "ERROR (lqcdjk_fitting.mEffTwopFit):"
                          + str( error ), mpi_confs_info )

if rank == 0:

    fitParams = fitResults[ 0 ]
    chiSq = fitResults[ 1 ]
    mEff_fit = fitResults[ 2 ]
    rangeStart = fitResults[ 3 ]
    mEff_rangeStart = fitResults[ 4 ]
    
    mEffFit_str = "2s" + str( rangeStart ) \
                  + ".2e" + str( rangeEnd )

    if tsf:

        c = fitParams[ :, 0 ]
        E0_mEff = fitParams[ :, 1 ]
        E1_mEff = fitParams[ :, 2 ]

        # Calculate fitted curve

        curve, \
            t_s = fit.calcmEffTwoStateCurve( np.ones( binNum ),
                                             c, E0_mEff, E1_mEff, T,
                                             rangeStart,
                                             rangeEnd )

        curveOutputFilename = rw.makeFilename( output_template,
                                               "mEff_2sf_curve_{}",
                                               mEffFit_str )
        
        c_avg = np.average( c, axis=0 )
        c_err = fncs.calcError( c, binNum )

        E0_mEff_avg = np.average( E0_mEff, axis=0 )
        E0_mEff_err = fncs.calcError( E0_mEff, binNum )

        E1_mEff_avg = np.average( E1_mEff, axis=0 )
        E1_mEff_err = fncs.calcError( E1_mEff, binNum )

        mEff_tsf_outputFilename = rw.makeFilename( output_template,
                                                   "mEff_2sf_{}",
                                                   mEffFit_str )
        rw.writeFitDataFile( mEff_tsf_outputFilename, E0_mEff_avg,
                             E0_mEff_err, rangeStart, rangeEnd )

        chiSqOutputFilename = rw.makeFilename( output_template,
                                               "mEff_2sf_chiSq_{}",
                                               mEffFit_str )

    else: # One-state fit

        c = fitParams[ :, 0 ]
        E0_mEff = fitParams[ :, 1 ]

        # Calculate fitted curve

        curve, t_s = fit.calcTwopOneStateCurve( c, E0_mEff, T,
                                                rangeStart, rangeEnd )

        curveOutputFilename = rw.makeFilename( output_template,
                                               "twop_1sf_curve_{}",
                                               mEffFit_str )
        chiSqOutputFilename = rw.makeFilename( output_template,
                                               "twop_oneStateFit_chiSq_{}",
                                               twopFit_str )

    # End if one-state fit

    #curve_avg = np.average( curve, axis=0 )
    #curve_err = fncs.calcError( curve, binNum )
            
    curve_avg = np.average( curve, axis=0 )
    curve_err = fncs.calcError( curve, binNum )

    chiSq_avg = np.average( chiSq, axis=0 )
    chiSq_err = fncs.calcError( chiSq, binNum )
            
    # Write output files

    rw.writeAvgDataFile_wX( curveOutputFilename, t_s,
                            curve_avg, curve_err )
        
    rw.writeFitDataFile( chiSqOutputFilename, chiSq_avg,
                         chiSq_err, rangeStart, rangeEnd )

    mEff_fit_avg = np.average( mEff_fit, axis=0 )
    mEff_fit_err = fncs.calcError( mEff_fit, binNum )

    mEff_range_str = "2s" + str( mEff_rangeStart ) \
                     + ".2e" + str( rangeEnd )

    mEff_outputFilename = rw.makeFilename( output_template,
                                           "mEff_fit_{}",
                                           mEff_range_str )
    rw.writeFitDataFile( mEff_outputFilename, mEff_fit_avg,
                         mEff_fit_err, mEff_rangeStart, rangeEnd )

# End if first process

comm.Barrier()


###############################
# Boosted two-point functions #
###############################


if momSq > 0:

    twop_boost = rw.readTwopFile_zeroQ( twopDir, 
                                        mpi_confs_info[ 'configList_loc' ], 
                                        mpi_confs_info[ 'configNum' ],
                                        twop_template, srcNum, momSq,
                                        dataFormat, mpi_confs_info )

    twop_boost_fold_p = np.zeros( ( momBoostNum, 
                                    binNum, T_fold ) )

# Loop over momenta
for imom in range( momBoostNum ):

    if momSq > 0:

        if mpi_confs_info[ 'binNum_loc' ]:

            twop_boost_jk_loc \
                = fncs.jackknifeBinSubset( twop_boost[ imom ],\
                                           mpi_confs_info[ 'binSize' ],
                                           mpi_confs_info[ 'binList_loc' ] )

            twop_boost_fold_loc = fncs.fold( twop_boost_jk_loc )

        else:

            twop_boost_fold_loc = np.array( [] )

        comm.Allgatherv( twop_boost_fold_loc, 
                         [ twop_boost_fold_p[ imom ],
                           recvCount * T_fold,
                           recvOffset * T_fold,
                           MPI.DOUBLE ] )

    # End if non-zero momentum boost

# End loop over momenta
    

###########################
# Fit two-point functions #
###########################


if momSq > 0: # Boosted two-point functions

    # Average over momenta

    twop_to_fit = twop_boost_fold = np.average( twop_boost_fold_p, axis=0 )
    mEff_to_fit = pq.mEffFromSymTwop( twop_to_fit )

    if np.any( np.isnan( mEff_to_fit ) ):

        rangeEnd = min(np.where(np.isnan(mEff_to_fit))[-1]) - 1

    fitResults_tmp = fit.mEffTwopFit( mEff_to_fit, twop_to_fit,
                                      rangeEnd, momSq, L, tsf,
                                      mpi_confs_info,
                                      tsf_t_low_range=[tsf_fitStart],
                                      plat_t_low_range=[plat_fitStart],
                                      checkFit=checkFit )

    twop_rangeStart = fitResults_tmp[ 3 ]
        
else: # Zero momentum two-point functions

    twop_to_fit = twop_fold
    twop_rangeStart = rangeStart
    
"""
if args.twop_fit_start: # fit starts at given t
    
twop_rangeStart = args.twop_fit_start

else: # fit range starts at same t as was used for mEff

twop_rangeStart = rangeStart

twopFit_str = "2s" + str( twop_rangeStart ) \
+ ".2e" + str( rangeEnd )
"""
#twopFit_str = "2s" + str( twop_rangeStart )
#              + ".2e" + str( rangeEnd )

twop_rangeStart = comm.bcast( twop_rangeStart, root=0 )

if tsf:

    fitParams_twop,chiSq=fit.twoStateFit_twop( twop_to_fit,
                                               twop_rangeStart,
                                               rangeEnd, T,
                                               mpi_confs_info )

else: # One-state fit

    fitParams_twop,chiSq=fit.oneStateFit_twop( twop_to_fit,
                                               twop_rangeStart,
                                               rangeEnd, T )


######################################################
# Write the two-point functions and their fit curves #
######################################################


if rank == 0:

    if tsf:
        
        c0 = fitParams_twop[ :, 0 ]
        c1 = fitParams_twop[ :, 1 ]
        E0 = fitParams_twop[ :, 2 ]
        E1 = fitParams_twop[ :, 3 ]

        twop_curve, ts_twop = fit.calcTwopTwoStateCurve( c0, c1, 
                                                         E0, E1, T,
                                                         twop_rangeStart, 
                                                         rangeEnd )

    else: # One-state fit
        
        c0 = fitParams_twop[ :, 0 ]
        E0 = fitParams_twop[ :, 1 ]
            
        twop_curve, ts_twop = fit.calcTwopOneStateCurve( c0, E0, T,
                                                         twop_rangeStart, 
                                                         rangeEnd )

    # End one-state fit

    # Average over bins
        
    twop_avg = np.average( twop_to_fit, axis=-2 )
    twop_err = fncs.calcError( twop_to_fit, binNum, axis=-2 )

    twop_curve_avg = np.average( twop_curve, axis=-2 )
    twop_curve_err = fncs.calcError( twop_curve, binNum, axis=-2 )

    # Write twop output file for each momentum
    
    twop_outFilename = rw.makeFilename( output_template, "twop" )
    rw.writeAvgDataFile( twop_outFilename, twop_avg, twop_err )

    twop_curve_outFilename = rw.makeFilename( output_template,
                                              "twop_2sf_curve" )
    rw.writeAvgDataFile_wX( twop_curve_outFilename, ts_twop,
                            twop_curve_avg, twop_curve_err )

# End first process


##############################
# Read three-point functions #
##############################


threep_p_jk = np.zeros( ( momBoostNum, flavNum,
                          tsinkNum, binNum, T ) )

# Loop over momenta
for imom in range( momBoostNum ):

    # Loop over tsink
    for ts, its in zip( tsink, range( tsinkNum ) ) :
    
        # threep_p[ iflav, c, t ]

        threep_p = rw.getMellinMomentThreep( threepDir, 
                                             mpi_confs_info[ 'configList_loc' ],
                                             mpi_confs_info[ 'configNum' ], 
                                             threep_tokens,
                                             srcNum, ts, momList[ imom ],
                                             particle, dataFormat, moment,
                                             L, T, mpi_confs_info )

        # Loop over flavor
        for iflav in range( flavNum ):

            # Jackknife
            # threep_p_jk[ iflav, ts, b, t ]

            threep_p_jk_loc \
                = fncs.jackknifeBinSubset( threep_p[ iflav ],
                                           mpi_confs_info[ 'binSize' ],
                                           mpi_confs_info[ 'binList_loc' ] )

            comm.Gatherv( threep_p_jk_loc,
                          [ threep_p_jk[ imom, iflav, its ],
                            recvCount * T,
                            recvOffset * T,
                            MPI.DOUBLE ], root=0 )

        # End loop over flavor
    # End loop over tsink
# End loop over momenta


####################
# Calculate moment #
####################


if rank == 0:

    # mellin[ flav, ts, b, t ]

    mellin = np.zeros( ( flavNum, tsinkNum, binNum, T ) )

    # Average threep over momenta

    threep_jk = np.average( threep_p_jk, axis=0 )

    # Loop over flavor
    for iflav in range( flavNum ):
        # Loop over tsink
        for ts, its in zip( tsink, range( tsinkNum ) ) :
            
            mellin[ iflav,
                    its ] = Z * pq.calcMellin_twopFit( threep_jk[ iflav,
                                                                  its ],
                                                       ts,E0_mEff,momSq,
                                                       L, c0, E0, 
                                                       moment )

        # End loop over tsink
    # End loop over flavor

    # Average over bins
    # mellin_avg[ flav, ts, t ]

    mellin_avg = np.average( mellin, axis=-2 )
    mellin_err = fncs.calcError( mellin, binNum, axis=-2 )

    threep_avg = np.average( threep_jk, axis=-2 )
    threep_err = fncs.calcError( threep_jk, binNum, 
                                 axis=-2 )

    # Loop over flavor
    for iflav in range( flavNum ):
        # Loop over tsink
        for ts, its in zip( tsink, range( tsinkNum ) ) :
            
            # Write threep output file
    
            threep_outFilename = rw.makeFilename( output_template,
                                                  "{}_threep_{}_tsink{}",
                                                  moment_str,
                                                  flav_str[ iflav ], ts )

            rw.writeAvgDataFile( threep_outFilename,
                                 threep_avg[ iflav, its ],
                                 threep_err[ iflav, its ] )

            # Write moment output file

            mellin_outFilename \
                = rw.makeFilename( output_template,
                                   "{}_{}_tsink{}",
                                   moment_str, flav_str[ iflav ],
                                   ts )
            rw.writeAvgDataFile( mellin_outFilename, mellin_avg[ iflav, its ],
                                 mellin_err[ iflav, its ] )

            # Fit plateau

            rangeStart_plat = [ ts // 2 - 1, ts // 2 - 2,
                                ts // 2 - 3, ts // 2 - 4 ]
            
            fitEnd_plat = [ ts // 2 + 1, ts // 2 + 2,
                            ts // 2 + 3, ts // 2 + 4 ]

            # Loop over fit ranges
            for irange in range( len( rangeStart_plat ) ):

                mellin_fit, chiSq = fit.fitPlateau( mellin[ iflav, its ],
                                                    mellin_err[iflav, its ],
                                                    rangeStart_plat[ irange ],
                                                    fitEnd_plat[ irange ] )

                # Average over bins

                mellin_fit_avg = np.average( mellin_fit )
                mellin_fit_err = fncs.calcError( mellin_fit, 
                                                 binNum )
                
                # Write output files

                mellin_fit_outFilename \
                    = rw.makeFilename( output_template,\
                                       "{}_{}_fit_{}_{}_tsink{}",
                                       moment_str,
                                       flav_str[ iflav ],
                                       rangeStart_plat[ irange ],
                                       fitEnd_plat[ irange ],
                                       ts)
                rw.writeFitDataFile( mellin_fit_outFilename, mellin_fit_avg,
                                     mellin_fit_err, rangeStart_plat[ irange ],
                                     fitEnd_plat[ irange ] )
            
            # End loop over fit ranges
        # End loop over tsink
    # End loop over flavor

    exit()

    ##################
    # Two-state Fit  #
    ##################

    if tsf:

        mpi_fncs.mpiPrint( "Will perform the two-state fit", mpi_confs_info )

        # Loop over flavors
        for iflav in range( flavNum ):
            # Loop over number of neglected three-point functions
            for neglect in 2,3:

                ti_to_fit = fncs.initEmptyList( tsinkNum, 1 )

                # Loop over tsinks
                for ts in range( tsinkNum ):

                    ti_to_fit[ ts ] = range( neglect,
                                             tsink[ ts ] - neglect + 1 )
                    """
                    ti_to_fit[ ts ] = np.concatenate( ( range( neglect,
                    tsink[ ts ]
                    - neglect
                    + 1 ),
                    range( tsink[ ts ]
                    + neglect
                    + 5,
                    T -
                    neglect
                    - 5 + 1 ) ) )
                    ti_to_fit[ ts ] = range( tsink[ ts ] + neglect + 5,
                    T - neglect - 5 + 1 )
                    """

                # End loop over tsink

                fitParams = np.zeros( ( momBoostNum, 
                                        binNum, 7 ) )
                mellin = np.zeros( ( momBoostNum, 
                                     binNum ) )
                chiSq = np.zeros( ( momBoostNum, 
                                    binNum ) )

                threep_curve = fncs.initEmptyList( momBoostNum, 1 )
                mellin_curve_const_ts = fncs.initEmptyList( momBoostNum, 1 )
                mellin_curve_const_ti = fncs.initEmptyList( momBoostNum, 1 )
            
                # Loop over momenta
                for imom in range( momBoostNum ):
                
                    fitParams_threep, \
                        chiSq[imom]=fit.twoStateFit_threep(threep_p_jk[imom,
                                                                       iflav],
                                                           ti_to_fit,
                                                           tsink, E0, E1, T )
                    
                    a00 = fitParams_threep[ :, 0 ]
                    a01 = fitParams_threep[ :, 1 ]
                    a11 = fitParams_threep[ :, 2 ]
          
                    # CJL: This needs to be fixed for multi momenta
                    fitParams[ imom ] = np.stack( ( a00, a01, a11,
                                                    c0, c1, E0, E1 ),
                                                  axis=1 )
                
                    # Calculate curve with constant tsink
                    
                    # curve[ p, b, ts, t ]
                    # ti[ ts, t ]

                    threep_curve[ imom ], \
                        ti_threep = fit.calcThreepTwoStateCurve( a00, a01,
                                                                 a11, E0,
                                                                 E1, T,
                                                                 tsink,
                                                                 ti_to_fit,
                                                                 neglect )

                    mellin_curve_const_ts[ imom ], \
                        ti_mellin = fit.calcAvgXTwoStateCurve_const_ts( a00,
                                                                        a01,
                                                                        a11,
                                                                        c0,
                                                                        c1,
                                                                        E0,
                                                                        E1,
                                                                        E0_mEff,
                                                                        momSq,
                                                                        L, T,
                                                                        Z,
                                                                        tsink,
                                                                        ti_to_fit,
                                                                        neglect )

                    mellin_curve_const_ti[ imom ], \
                        ts_mellin = fit.calcAvgXTwoStateCurve_const_ti( a00,
                                                                        a01,
                                                                        a11,
                                                                        c0,
                                                                        c1,
                                                                        E0,
                                                                        E1,
                                                                        E0_mEff,
                                                                        momSq,
                                                                        L, T,
                                                                        Z,
                                                                        tsink[0]-2,
                                                                        tsink[-1]+5 )
                    
                    # mellin[ p, b ]

                    mellin[ imom ] = pq.calcAvgX_twoStateFit( a00, c0,
                                                              E0_mEff,
                                                              momSq,
                                                              L, Z )
                
                # End loop over momenta

                # Average over momenta
                    
                threep_curve = np.average( threep_curve, axis=0 )
            
                mellin_curve_const_ts = np.average( mellin_curve_const_ts,
                                                    axis=0 )
                
                mellin_curve_const_ti = np.average( mellin_curve_const_ti,
                                                    axis=0 )
                
                fitParams = np.average( fitParams, axis=0 )
            
                chiSq = np.average( chiSq, axis=0 )
                
                mellin = np.average( mellin, axis=0 )
    
                # Average over bins
                    
                threep_curve_avg = np.average( threep_curve, axis=0 )
                threep_curve_err = fncs.calcError( threep_curve, 
                                                   binNum )
            
                mellin_curve_const_ts_avg = np.average( mellin_curve_const_ts,
                                                        axis=0 )
                mellin_curve_const_ts_err = fncs.calcError( mellin_curve_const_ts,
                                                            binNum )
                
                mellin_curve_const_ti_avg = np.average( mellin_curve_const_ti,
                                                        axis=0 )
                mellin_curve_const_ti_err = fncs.calcError( mellin_curve_const_ti,
                                                            binNum )
                
                fitParams_avg = np.average( fitParams, axis=0 )
                fitParams_err = fncs.calcError( fitParams, 
                                                binNum )
                
                chiSq_avg = np.average( chiSq, axis=0 )
                chiSq_err = fncs.calcError( chiSq, binNum )
                
                mellin_avg = np.average( mellin )
                mellin_err = fncs.calcError( mellin, binNum )
    
                # Write output file

                tsf_threep_range_str = mEffFit_str + ".3n" + str( neglect )

                mellinOutputFilename \
                    = rw.makeFilename( output_template,
                                       "{}_{}_2sf_{}_{}",
                                       moment_str,
                                       flav_str[ iflav ],
                                       tsf_threep_range_str,
                                       ts_range_str )
                
                rw.writeFitDataFile( mellinOutputFilename,
                                     mellin_avg, mellin_err, 0, 0 )
                
                chiSqOutputFilename \
                    = rw.makeFilename( output_template,
                                       "{}_{}_2sf_threep_chiSq_{}_{}",
                                       moment_str,
                                       flav_str[ iflav ],
                                       tsf_threep_range_str,
                                       ts_range_str )
        
                rw.writeFitDataFile( chiSqOutputFilename,
                                     chiSq_avg, chiSq_err, 0, 0 )
                
                mellinParamsOutputFilename \
                    = rw.makeFilename( output_template,
                                       "{}_{}_2sf_params_{}_{}",
                                       moment_str,
                                       flav_str[ iflav ],
                                       tsf_threep_range_str,
                                       ts_range_str )

                rw.writeTSFParamsFile( mellinParamsOutputFilename,
                                       fitParams_avg, fitParams_err )
            
                curveOutputFilename \
                    = rw.makeFilename( output_template,
                                       "{}_{}_2sf_curve_{}_{}",
                                       moment_str,
                                       flav_str[ iflav ],
                                       tsf_threep_range_str,
                                       ts_range_str )
                rw.writeAvgDataFile_wX( curveOutputFilename,
                                        ts_mellin,
                                        mellin_curve_const_ti_avg,
                                        mellin_curve_const_ti_err )
                
                for ts in range( tsinkNum ):
            
                    threep_curveOutputFilename \
                        = rw.makeFilename( output_template,
                                           "{}_threep_{}_2sf_curve_tsink{}_{}_{}",
                                           moment_str,
                                           flav_str[ iflav ],
                                           tsink[ ts ],
                                           tsf_threep_range_str,
                                           ts_range_str )
                    rw.writeAvgDataFile_wX( threep_curveOutputFilename,
                                            ti_threep[ ts ],
                                            threep_curve_avg[ ts ],
                                            threep_curve_err[ ts ] )
                    
                    curveOutputFilename \
                        = rw.makeFilename( output_template,
                                           "{}_{}_2sf_curve_tsink{}_{}_{}",
                                           moment_str,
                                           flav_str[ iflav ],
                                           tsink[ ts ],
                                           tsf_threep_range_str,
                                           ts_range_str )
                    rw.writeAvgDataFile_wX( curveOutputFilename,
                                            ti_mellin[ ts ],
                                            mellin_curve_const_ts_avg[ ts ],
                                            mellin_curve_const_ts_err[ ts ] )
            
                # End loop over tsink
            # End loop over number of neglected three-point functions
        # End loop over flavors
    # End if two-state fit
# End if first process
