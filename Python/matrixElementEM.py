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

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Calculate GE(0)" )

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

parser.add_argument( "fit_range_end", action='store', type=int )

parser.add_argument( "particle", action='store', \
                     help="Particle to calculate gA for. " \
                     + "Should be 'pion' or 'kaon'.", type=str )

parser.add_argument( 't_sink', action='store', \
                     help="Comma seperated list of t sink's", \
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "mom_squared", action='store', type=int )

parser.add_argument( "binSize", action='store', type=int )

parser.add_argument( "-o", "--output_template", action='store', \
                     type=str, default="./*.dat" )

parser.add_argument( "-sn", "--source_number", action='store', type=int, \
                     help="Number of sources correlators were averaged " \
                     + "over", default=16 )

parser.add_argument( "-tsf", "--two_state_fit", action='store_true', \
                     help="Performs the two-state fit if supplied" )

parser.add_argument( "--tsf_fit_start", action='store', type=int, \
                     help="If given, will perform two-state fit on effective " \
                     + "mass starting at given t value, otherwise, will " \
                     + "use lowest t value which satisfies condition." )

parser.add_argument( "--twop_fit_start", action='store', type=int, \
                     help="If given, will perform one- or two-state fit on " \
                     + "two-point functions starting at given t value, " \
                     + "otherwise, will use lowest t value which satisfies " \
                     + "condition." )

parser.add_argument( "--plat_fit_start", action='store', type=int, \
                     help="If given, will perform plateau fit on effective " \
                     + "mass starting at given t value, otherwise, will " \
                     + "use lowest t value which satisfies condition." )

parser.add_argument( "-f", "--data_format", action='store', \
                     help="Data format. Should be 'gpu' or 'cpu'.", \
                     type=str, default="gpu" )

parser.add_argument( "-i", "--insertion_type", action='store', \
                     help="Type of insertion current. Should be 'local' or 'noether'.", \
                     type=str, default="noether" )

parser.add_argument( "-c", "--config_list", action='store', \
                     type=str, default="" )

args = parser.parse_args()

#########
# Setup #
#########

# Set MPI values

mpi_confs_info = mpi_fncs.lqcdjk_mpi_init()

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
plat_fitStart = args.plat_fit_start

if tsf_fitStart and plat_fitStart:

    checkFit = False

else:

    checkFit = True

srcNum = args.source_number

tsf = args.two_state_fit

dataFormat = args.data_format

momSq = args.mom_squared

# Get configurations from given list or from given 
# threep directory if list not given

mpi_confs_info[ 'configList' ] = fncs.getConfigList( args.config_list, 
                                                     threepDir )
mpi_confs_info[ 'configNum' ] = len( mpi_confs_info[ 'configList' ] )
mpi_confs_info[ 'binSize' ] = args.binSize

# Set mpi configuration information

mpi_fncs.lqcdjk_mpi_confs_info( mpi_confs_info )

binNum = mpi_confs_info[ 'binNum_glob' ]

# Check inputs

insType = args.insertion_type

assert insType in [ "local", "noether" ], \
    "Error: insertion type not supported. " \
    "Should be 'local' or 'noether'."

if insType == "local":

    Zv = 0.715

else:

    Zv = 1.0

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
    + "Supported particles: " + str( format_list )

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

if mpi_confs_info[ 'rank' ] == 0:

    twop_fold = np.zeros( ( binNum, T_fold ) )
    mEff = np.zeros( ( binNum, T_fold ) )

else:

    twop_fold = None
    mEff = None

mpi_confs_info[ 'comm' ].Gatherv( twop_fold_loc, 
                            [ twop_fold, mpi_confs_info[ 'recvCount' ] * T_fold,
                              mpi_confs_info[ 'recvOffset' ] * T_fold, 
                              MPI.DOUBLE ], root=0 )
mpi_confs_info[ 'comm' ].Gatherv( mEff_loc, 
                            [ mEff, mpi_confs_info[ 'recvCount' ] * T_fold,
                              mpi_confs_info[ 'recvOffset' ] * T_fold, 
                              MPI.DOUBLE ], root=0 )

if mpi_confs_info[ 'rank' ] == 0:

    # mEff_avg[ t ]

    mEff_avg = np.average( mEff, axis=0 )
    mEff_err = fncs.calcError( mEff, binNum )

    avgOutputFilename = rw.makeFilename( output_template, "mEff_avg" )
    rw.writeAvgDataFile( avgOutputFilename, mEff_avg, mEff_err )

    # Fit the effective mass and two-point functions 

    try:
    
        fitResults = fit.mEffTwopFit( mEff, twop_fold, \
                                      rangeEnd, 0, L, tsf, \
                                      tsf_t_low_range=[tsf_fitStart], \
                                      plat_t_low_range=[plat_fitStart], \
                                      checkFit=checkFit )
    
    except fit.lqcdjk_BadFitError as error:
        
        mpi_fncs.mpiPrintErr( "ERROR (lqcdjk_fitting.mEffTwopFit):" \
                              + str( error ), comm )

    fitParams = fitResults[ 0 ]
    chiSq = fitResults[ 1 ]
    mEff_fit = fitResults[ 2 ]
    rangeStart = fitResults[ 3 ]
    mEff_rangeStart = fitResults[ 4 ]

    twopFit_str = "2s" + str( rangeStart ) \
                  + ".2e" + str( rangeEnd )

    if tsf:

        c = fitParams[ :, 0 ]
        E0_mEff = fitParams[ :, 1 ]
        E1_mEff = fitParams[ :, 2 ]

        # Calculate fitted curve

        mEff_curve, \
            t_s = fit.calcmEffTwoStateCurve( np.ones( binNum ), \
                                             c, E0_mEff, E1_mEff, T, \
                                             rangeStart, \
                                             rangeEnd )
                                
        mEff_curveOutputFilename = rw.makeFilename( output_template, \
                                                    "mEff_2sf_curve_{}", \
                                                    twopFit_str )

        c_avg = np.average( c, axis=0 )
        c_err = fncs.calcError( c, binNum )

        E0_mEff_avg = np.average( E0_mEff, axis=0 )
        E0_mEff_err = fncs.calcError( E0_mEff, binNum )

        E1_mEff_avg = np.average( E1_mEff, axis=0 )
        E1_mEff_err = fncs.calcError( E1_mEff, binNum )

        mEff_tsf_outputFilename = rw.makeFilename( output_template, \
                                                   "mEff_2sf_{}", \
                                                   twopFit_str )
        rw.writeFitDataFile( mEff_tsf_outputFilename, E0_mEff_avg, \
                             E0_mEff_err, rangeStart, rangeEnd )

        chiSqOutputFilename = rw.makeFilename( output_template, \
                                               "mEff_2sf_chiSq_{}", \
                                               twopFit_str )

    else: # One-state fit

        c0 = fitParams[ :, 0 ]
        E0 = fitParams[ :, 1 ]

        # Calculate fitted curve

        curve, t_s = fit.calcTwopOneStateCurve( c0, E0, T, \
                                                rangeStart, rangeEnd )

        curveOutputFilename = rw.makeFilename( output_template, \
                                               "twop_1sf_curve_{}", \
                                               twopFit_str )
        chiSqOutputFilename = rw.makeFilename( output_template, \
                                               "twop_1sf_chiSq_{}", \
                                               twopFit_str )

    # End if one-state fit

    #curve_avg = np.average( curve, axis=0 )
    #curve_err = fncs.calcError( curve, binNum )
            
    mEff_curve_avg = np.average( mEff_curve, axis=0 )
    mEff_curve_err = fncs.calcError( mEff_curve, binNum )

    chiSq_avg = np.average( chiSq, axis=0 )
    chiSq_err = fncs.calcError( chiSq, binNum )
            
    # Write output files

    rw.writeAvgDataFile_wX( mEff_curveOutputFilename, t_s, \
                            mEff_curve_avg, mEff_curve_err )
        
    rw.writeFitDataFile( chiSqOutputFilename, chiSq_avg, \
                         chiSq_err, rangeStart, rangeEnd )

    mEff_fit_avg = np.average( mEff_fit, axis=0 )
    mEff_fit_err = fncs.calcError( mEff_fit, binNum )

    mEff_range_str = "2s" + str( mEff_rangeStart ) \
                     + ".2e" + str( rangeEnd )

    mEff_outputFilename = rw.makeFilename( output_template, \
                                           "mEff_fit_{}", \
                                           mEff_range_str )
    rw.writeFitDataFile( mEff_outputFilename, mEff_fit_avg, \
                         mEff_fit_err, mEff_rangeStart, rangeEnd )

# End if first process

mpi_confs_info[ 'comm' ].Barrier()

# Boosted two-point functions
# twop_boost[ mom, c, t ]

if momSq > 0:

    twop_boost = rw.readTwopFile_zeroQ( twopDir, 
                                        mpi_confs_info[ 'configList_loc' ], 
                                        mpi_confs_info[ 'configNum' ],
                                        twop_template, srcNum, momSq,
                                        dataFormat, mpi_confs_info )

else:

    twop_boost_loc = np.array( [] )

if rank == 0:
        
    twop_boost_fold_p = np.zeros( ( momBoostNum, binNum, T_fold ) )
    threep_p_jk = np.zeros( ( momBoostNum, flavNum, \
                              tsinkNum, binNum, T ) )

else:

    twop_boost_fold_p = np.array( [ None for imom in range( momBoostNum ) ] )
    threep_p_jk = np.array( [ [ [ None for ts in tsink ] \
                                for f in flav_str ] \
                              for imom in range( momBoostNum ) ] )

# Loop over momenta
for imom in range( momBoostNum ):

    if momSq > 0:

        #########################################
        # Jackknife boosted two-point functions #
        #########################################

        if mpi_confs_info[ 'binNum_loc' ]:

            twop_boost_jk_loc \
                = fncs.jackknifeBinSubset( twop_boost[ imom ],\
                                           mpi_confs_info[ 'binSize' ],
                                           mpi_confs_info[ 'binList_loc' ] )

            twop_boost_fold_loc = fncs.fold( twop_boost_jk_loc )

        else:

            twop_boost_fold_loc = np.array( [] )

        mpi_confs_info[ 'comm' ].Gatherv( twop_boost_fold_loc, 
                                          [ twop_boost_fold_p[ imom ],
                                            mpi_confs_info[ 'recvCount' ] 
                                            * T_fold,
                                            mpi_confs_info[ 'recvOffset' ] 
                                            * T_fold,
                                            MPI.DOUBLE ], root=0 )

    # End if non-zero momentum boost
    
    ##############################
    # Read three-point functions #
    ##############################

    # Loop over tsink
    for ts, its in zip( tsink, range( tsinkNum ) ) :
    
        # Get the real part of gxDx, gyDy, gzDz, and gtDt
        # three-point functions at zero-momentum
        # threep[ c, t ]

        threeps = rw.readEMFile( threepDir, mpi_confs_info[ 'configList_loc' ],
                                 threep_tokens, srcNum, ts, momList[ imom ], \
                                 particle, dataFormat, insType, T, comm )

        threep_loc = threeps[ 0 ]

        threep = np.zeros( ( flavNum, configNum, T ) )

        comm.Allgather( threep_loc, threep[ 0 ] )

        if particle == "kaon":

            threep_loc = threeps[ 1 ]

            comm.Allgather( threep_loc, threep[ 1 ] )

        # Loop over flavor
        for iflav in range( flavNum ):

            # Jackknife
            # threep_p_jk[ p, iflav, ts, b, t ]

            threep_p_jk_loc \
                = fncs.jackknifeBinSubset( threep[ iflav ],
                                           mpi_confs_info[ 'binSize' ],
                                           mpi_confs_info[ 'binList_loc' ] )

            mpi_confs_info[ 'comm' ].Gatherv( threep_jk_loc,
                                              [ threep_p_jk[ imom, iflav, its ],
                                                mpi_confs_info[ 'recvCount' ] 
                                                * T,
                                                mpi_confs_info[ 'recvOffset' ] 
                                                * T,
                                                MPI.DOUBLE ], root=0 )
            """
            if rank == 0:

                threep_avg = np.average( threep_jk[ imom, iflav, its ], axis=-2 )
                threep_err = fncs.calcError( threep_jk[ imom, iflav, its ], \
                                             binNum, \
                                             axis=-2 )

                threep_output_template = "threep_{0}_tsink{1}_{2:+}_{3:+}_{4:+}".format( flav_str[iflav], ts, momList[imom][0], momList[imom][1], momList[imom][2] )

                threep_outFilename = output_template.replace( "*", \
                                                              threep_output_template )

                rw.writeAvgDataFile( threep_outFilename, \
                                     threep_avg, \
                                     threep_err )
            """
        # End loop over flavor
    # End loop over tsink
# End loop over momenta


###################
# Calculate GE(0) #
###################


if mpi_confs_info[ 'rank' ] == 0:
        
    ratio_p = np.zeros( ( momBoostNum, flavNum, tsinkNum, \
                       binNum, T ) )
    ratio_avgBeforeRatio = np.zeros( ( flavNum, tsinkNum, \
                                       binNum, T ) )

    c0_p = np.zeros( ( momBoostNum, binNum ) )
    E0_p = np.zeros( ( momBoostNum, binNum ) )

    if tsf:

        c1_p = np.zeros( ( momBoostNum, binNum ) )
        E1_p = np.zeros( ( momBoostNum, binNum ) )

    # ratio_p[ p, flav, ts, b, t ]

    # Loop over momenta
    for imom in range( momBoostNum ):

        # Fit the two-point functions

        if momSq > 0: # Boosted two-point functions
            
            twop_to_fit = twop_boost_fold_p[ imom ]
            mEff_to_fit = pq.mEffFromSymTwop( twop_boost_fold_p[ imom ] )

        else: # Zero momentum two-point functions

            twop_to_fit = twop_fold
            mEff_to_fit = mEff

        if np.any( np.isnan( mEff_to_fit ) ):

            rangeEnd = min(np.where(np.isnan(mEff_to_fit))[-1]) - 1
        """
        if args.twop_fit_start: # fit starts at given t

            twop_rangeStart = args.twop_fit_start

        else: # fit range starts at same t as was used for mEff

            twop_rangeStart = rangeStart
        """
        fitResults_p = fit.mEffTwopFit( mEff_to_fit, twop_to_fit,
                                          rangeEnd, momSq, L, tsf,
                                          tsf_t_low_range=[tsf_fitStart],
                                          plat_t_low_range=[plat_fitStart],
                                          checkFit=checkFit )

        twop_rangeStart = fitResults_p[ 3 ]
        
        #print(twop_rangeStart,fitResults_p[4])

        if tsf:

            fitParams_twop,chiSq=fit.twoStateFit_twop(twop_to_fit, \
                                                      twop_rangeStart, \
                                                      rangeEnd, T )
            
            c0_p[ imom ] = fitParams_twop[ :, 0 ]
            c1_p[ imom ] = fitParams_twop[ :, 1 ]
            E0_p[ imom ] = fitParams_twop[ :, 2 ]
            E1_p[ imom ] = fitParams_twop[ :, 3 ]

        else:

            fitParams_twop,chiSq=fit.oneStateFit_twop(twop_to_fit, \
                                                      twop_rangeStart, \
                                                      rangeEnd, T )

            c0_p[ imom ] = fitParams_twop[ :, 0 ]
            E0_p[ imom ] = fitParams_twop[ :, 1 ]

        fitParamsOutputFilename \
            = rw.makeFilename( output_template,
                               "twop_2sf_params_{0:+}_{1:+}_{2:+}", 
                               momList[ imom ][ 0 ],
                               momList[ imom ][ 1 ],
                               momList[ imom ][ 2 ] )
                         
        rw.writeDataFile( fitParamsOutputFilename, fitParams_twop )

        # Loop over flavor
        for iflav in range( flavNum ):
            # Loop over tsink
            for ts, its in zip( tsink, range( tsinkNum ) ) :

                ratio_p[imom, \
                        iflav, \
                        its]=Zv*pq.calcMatrixElemEM_twopFit(threep_p_jk[imom, \
                                                                        iflav, \
                                                                        its], \
                                                            ts, c0_p[ imom ], \
                                                            E0_p[ imom ] )

                                                            #c1_p[ imom ], \
                                                            #E1_p[ imom ] )
                
            # End loop over tsink
        # End loop over flavor
    # End loop over momenta

    # Average over momenta
    # ratio[ flav, ts, b, t ]

    ratio = np.average( ratio_p, axis=0 )
    threep_jk = np.average( threep_p_jk, axis=0 )

    if momSq > 0:

        twop_boost_fold = np.average( twop_boost_fold_p, axis=0 )

    c0 = np.average( c0_p, axis=0 )
    c1 = np.average( c1_p, axis=0 )
    E0 = np.average( E0_p, axis=0 )
    E1 = np.average( E1_p, axis=0 )
    
    # Calculate moment from averaged twop and threep

    # Loop over flavor
    for iflav in range( flavNum ):
        # Loop over tsink
        for ts, its in zip( tsink, range( tsinkNum ) ) :
            
            ratio_avgBeforeRatio[iflav, \
                                 its]=Zv*pq.calcMatrixElemEM_twopFit(threep_jk[iflav, \
                                                                               its], \
                                                                     ts, c0, E0 ) 

            # c1, \E1 )

    # Average over bins
    # ratio_avg[ flav, ts, t ]

    ratio_avg = np.average( ratio, axis=-2 )
    ratio_err = fncs.calcError( ratio, binNum, axis=-2 )

    ratio_avgBeforeRatio_avg = np.average( ratio_avgBeforeRatio, axis=-2 )
    ratio_avgBeforeRatio_err = fncs.calcError( ratio_avgBeforeRatio, \
                                               binNum, axis=-2 )

    threep_avg = np.average( threep_jk, axis=-2 )
    threep_err = fncs.calcError( threep_jk, binNum, axis=-2 )

    if momSq > 0: # Boosted two-point functions
            
        twop_avg = np.average( twop_boost_fold, axis=-2 )
        twop_err = fncs.calcError( twop_boost_fold, binNum, axis=-2 )

    else: # Zero momentum two-point functions

        twop_avg = np.average( twop_fold, axis=-2 )
        twop_err = fncs.calcError( twop_fold, binNum, axis=-2 )

    # Write twop output file

    twop_outFilename = rw.makeFilename( output_template, "twop" )

    rw.writeAvgDataFile( twop_outFilename, \
                         twop_avg, \
                         twop_err )

    # Loop over flavor
    for iflav in range( flavNum ):
        # Loop over tsink
        for ts, its in zip( tsink, range( tsinkNum ) ) :
            
            # Write threep output files
    
            threep_outFilename = rw.makeFilename( output_template, \
                                                  "threep_{}_tsink{}", \
                                                  flav_str[ iflav ], ts )
            rw.writeAvgDataFile( threep_outFilename, \
                                 threep_avg[ iflav, its ], \
                                 threep_err[ iflav, its ] )

            # Write GE(0) output files
    
            ratio_outFilename = rw.makeFilename( output_template, \
                                                 "matrixElemEM_{}_tsink{}", \
                                                 flav_str[ iflav ], ts )

            rw.writeAvgDataFile( ratio_outFilename, ratio_avg[ iflav, its ], \
                                 ratio_err[ iflav, its ] )

            ratio_outFilename \
                = rw.makeFilename( output_template, \
                                   "matrixElemEM_{}_tsink{}_avgBeforeRatio", \
                                   flav_str[ iflav ], ts )
            rw.writeAvgDataFile( ratio_outFilename, \
                                 ratio_avgBeforeRatio_avg[ iflav, its ], \
                                 ratio_avgBeforeRatio_err[ iflav, its ] )

            ###############
            # Fit plateau #
            ###############

            fitStart = [ ts // 2 - 1, ts // 2 - 2, \
                         ts // 2 - 3, ts // 2 - 4 ]
            
            fitEnd = [ ts // 2 + 1, ts // 2 + 2, \
                       ts // 2 + 3, ts // 2 + 4 ]

            # Loop over fit ranges
            for irange in range( len( fitStart ) ):

                #for x in ratio[ iflav, its ]:

                ratio_fit, chiSq = fit.fitPlateau( ratio[ iflav, \
                                                        its ], \
                                                   ratio_err[iflav, \
                                                            its ], \
                                                   fitStart[irange], \
                                                   fitEnd[irange] )

                # Average over bins

                ratio_fit_avg = np.average( ratio_fit )
                
                ratio_fit_err = fncs.calcError( ratio_fit, binNum )
                
                # Write output files

                ratio_fit_outFilename \
                    = rw.makeFilename( output_template, \
                                       "matrixElemEM_{}_fit_tsink{}_{}_{}", \
                                       flav_str[iflav], ts, \
                                       fitStart[irange], fitEnd[irange] )

                rw.writeFitDataFile( ratio_fit_outFilename, \
                                     ratio_fit_avg, ratio_fit_err, \
                                     fitStart[ irange ], fitEnd[ irange ] )
            
            # End loop over fit ranges
        # End loop over tsink
    # End loop over flavor
# End if first process
