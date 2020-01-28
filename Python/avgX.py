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

ZvD1 = 1.123

L = 32.0

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

parser.add_argument( "--tsf_fit_start", action='store', type=int, \
                     help="If given, will perform two-state fit on effective " \
                     + "mass starting at given t value, otherwise, will " \
                     + "use lowest t value which satisfies condition." )

parser.add_argument( "--plat_fit_start", action='store', type=int, \
                     help="If given, will perform plateau fit on effective " \
                     + "mass starting at given t value, otherwise, will " \
                     + "use lowest t value which satisfies condition." )

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

# Set MPI values

comm = MPI.COMM_WORLD
procNum = comm.Get_size()
rank = comm.Get_rank()

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

binSize = args.binSize

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

configList = np.array( fncs.getConfigList( args.config_list, threepDir ) )
configNum = len( configList )

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
    + "Supported particles: " + str( format_list )

assert configNum % binSize == 0, "Number of configurations " \
    + str( configNum ) + " not evenly divided by bin size " \
    + str( binSize ) + "."

assert configNum % procNum == 0, "Number of configurations " \
    + str( configNum ) + " not evenly divided by number of processes " \
    + str( procNum ) + "."

# Number of configurations on each process

procSize = configNum // procNum

# Total number of bins across processes

binNum_glob = configNum // binSize

# Global index of confs for each process

iconf = np.array( [ np.array( [ r * procSize + cl \
                                for cl in range( procSize ) ], dtype=int )
                    for r in range( procNum ) ] )

# List of configurations on this process

configList_loc = configList[ iconf[ rank ] ]

# Global index of first conf of bins for each process

binStart = np.array( [ np.array( [ cl for cl in iconf[ r ] \
                                   if cl % binSize == 0 ], dtype=int )
                       for r in range( procNum ) ] )

# Global bin index for each process

bin_glob = binStart // binSize

# Number of bins for each process

binNum = [ len( binStart[ r ] ) for r in range( procNum ) ]

# Number of bins for this process

binNum_loc = binNum[ rank ]

recvCount, recvOffset = mpi_fncs.recvCountOffset( procNum, binNum )

# Read momentum list

momList = rw.readMomentaList( twopDir, twop_template, \
                              configList_loc[ 0 ], particle, \
                              srcNum, momSq, dataFormat, comm )

momBoostNum = len( momList )

############################
# Read Two-point Functions #
############################

# Zero momentum two-point functions
# twop[ c, t ]

twop = rw.readTwopFile_zeroQ( twopDir, configList_loc, configNum, \
                              twop_template, srcNum, 0, particle, dataFormat, \
                              comm )

# Time dimension length

T = twop.shape[ -1 ]

# Time dimension length after fold

T_fold = T // 2 + 1

##########################################
# Jackknife and fold two-point functions #
##########################################

if binNum_loc:

    twop_jk_loc = fncs.jackknifeBinSubset( twop, binSize, bin_glob[ rank ] )

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

if rank == 0:

    twop_fold = np.zeros( ( binNum_glob, T_fold ) )
    mEff = np.zeros( ( binNum_glob, T_fold ) )

else:

    twop_fold = None
    mEff = None

comm.Gatherv( twop_fold_loc, [ twop_fold, recvCount * T_fold, \
                             recvOffset * T_fold, MPI.DOUBLE ], root=0 )
comm.Gatherv( mEff_loc, [ mEff, recvCount * T_fold, \
                          recvOffset * T_fold, MPI.DOUBLE ], root=0 )

if rank == 0:

    # mEff_avg[ t ]

    mEff_avg = np.average( mEff, axis=0 )
    mEff_err = fncs.calcError( mEff, binNum_glob )

    avgOutputFilename = output_template.replace( "*", "mEff_avg" )
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
            t_s = fit.calcmEffTwoStateCurve( np.ones( binNum_glob ), \
                                             c, E0_mEff, E1_mEff, T, \
                                             rangeStart, \
                                             rangeEnd )
                                
        mEff_curveOutputFilename \
            = output_template.replace( "*", \
                                       "mEff_twoStateFit_curve_" \
                                       + twopFit_str )

        c_avg = np.average( c, axis=0 )
        c_err = fncs.calcError( c, binNum_glob )

        E0_mEff_avg = np.average( E0_mEff, axis=0 )
        E0_mEff_err = fncs.calcError( E0_mEff, binNum_glob )

        E1_mEff_avg = np.average( E1_mEff, axis=0 )
        E1_mEff_err = fncs.calcError( E1_mEff, binNum_glob )

        mEff_tsf_outputFilename \
            = output_template.replace( "*", \
                                       "mEff_twoStateFit_" \
                                       + twopFit_str )
        rw.writeFitDataFile( mEff_tsf_outputFilename, E0_mEff_avg, \
                             E0_mEff_err, rangeStart, rangeEnd )

        chiSqOutputFilename \
            = output_template.replace( "*", \
                                       "mEff_twoStateFit_chiSq_" \
                                       + twopFit_str )
        fitParams_twop, chiSq_twop = fit.twoStateFit_twop( twop_fold, \
                                                           rangeStart, \
                                                           rangeEnd, T )

        c0 = fitParams_twop[ :, 0 ]
        c1 = fitParams_twop[ :, 1 ]
        E0 = fitParams_twop[ :, 2 ]
        E1 = fitParams_twop[ :, 3 ]

    else: # One-state fit

        G = fitParams[ :, 0 ]
        E = fitParams[ :, 1 ]

        # Calculate fitted curve

        curve, t_s = fit.calcTwopOneStateCurve( G, E, T, \
                                                rangeStart, rangeEnd )

        curveOutputFilename \
            = output_template.replace( "*", \
                                       "twop_oneStateFit_curve_" \
                                       + twopFit_str )
        chiSqOutputFilename \
            = output_template.replace( "*", \
                                       "twop_oneStateFit_chiSq_" \
                                       + twopFit_str )

    # End if not two-state fit

    #curve_avg = np.average( curve, axis=0 )
    #curve_err = fncs.calcError( curve, binNum_glob )
            
    mEff_curve_avg = np.average( mEff_curve, axis=0 )
    mEff_curve_err = fncs.calcError( mEff_curve, binNum_glob )

    chiSq_avg = np.average( chiSq, axis=0 )
    chiSq_err = fncs.calcError( chiSq, binNum_glob )
            
    # Write output files

    rw.writeAvgDataFile_wX( mEff_curveOutputFilename, t_s, \
                            mEff_curve_avg, mEff_curve_err )
        
    rw.writeFitDataFile( chiSqOutputFilename, chiSq_avg, \
                         chiSq_err, rangeStart, rangeEnd )

    mEff_fit_avg = np.average( mEff_fit, axis=0 )
    mEff_fit_err = fncs.calcError( mEff_fit, binNum_glob )

    mEff_range_str = "2s" + str( mEff_rangeStart ) \
                     + ".2e" + str( rangeEnd )

    mEff_outputFilename = output_template.replace( "*", "mEff_fit_" + mEff_range_str )
    rw.writeFitDataFile( mEff_outputFilename, mEff_fit_avg, \
                         mEff_fit_err, mEff_rangeStart, rangeEnd )

# End if first process

comm.Barrier()

# Boosted two-point functions
# twop_boost[ mom, c, t ]

if momSq > 0:

    twop_boost = rw.readTwopFile_zeroQ( twopDir, configList_loc, configNum, \
                                        twop_template, srcNum, momSq, particle, \
                                        dataFormat, comm )

else:

    twop_boost = np.array( [] )

if rank == 0:
        
    twop_boost_fold = np.zeros( ( momBoostNum, binNum_glob, T_fold ) )
    threep_jk = np.zeros( ( momBoostNum, flavNum, \
                            tsinkNum, binNum_glob, T ) )
    avgX = np.zeros( ( momBoostNum, flavNum, \
                       tsinkNum, binNum_glob, T ) )

else:

    twop_boost_fold = np.array( [ None for imom in range( momBoostNum ) ] )
    threep_jk = np.array( [ [ [ None for ts in tsink ] \
                              for f in flav_str ] \
                            for imom in range( momBoostNum ) ] )

# Loop over momenta
for imom in range( momBoostNum ):

    if momSq > 0:

        #########################################
        # Jackknife boosted two-point functions #
        #########################################

        if binNum_loc:

            twop_boost_jk_loc = fncs.jackknifeBinSubset( twop_boost[ imom ],\
                                                         binSize, \
                                                         bin_glob[ rank ] )

            twop_boost_fold_loc = fncs.fold( twop_boost_jk_loc )

        else:

            twop_boost_fold_loc = np.array( [] )

        comm.Gatherv( twop_boost_fold_loc, [ twop_boost_fold[ imom ], \
                                             recvCount * T_fold, \
                                             recvOffset * T_fold, \
                                             MPI.DOUBLE ], root=0 )

    # End if non-zero momentum boost
    
    ##############################
    # Read three-point functions #
    ##############################

    t0 = time.time()

    # Loop over tsink
    for ts, its in zip( tsink, range( tsinkNum ) ) :
    
        t0_ts = time.time()

        # Get the real part of gxDx, gyDy, gzDz, and gtDt
        # three-point functions at zero-momentum
        # threep[ c, t ]

        threeps = rw.readAvgXFile( threepDir, configList_loc, \
                                   threep_tokens, srcNum, ts, momList[ imom ], \
                                   particle, dataFormat, T, comm )

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

        mpi_fncs.mpiPrint( "Read three-point functions from HDF5 files " \
                           + "for tsink {} in {:.4}".format( ts, \
                                                             time.time() \
                                                             - t0_ts ) \
                           + " seconds.", rank )

        # Subtract average over directions from gtDt

        threep_loc = threep_gtDt - \
                     0.25 * ( threep_gtDt \
                              + threep_gxDx \
                              + threep_gyDy \
                              + threep_gzDz )

        threep = np.zeros( ( flavNum, configNum, T ) )

        comm.Allgather( threep_loc, threep[ 0 ] )

        if particle == "kaon":

            threep_loc = threep_s_gtDt - \
                              0.25 * ( threep_s_gtDt \
                                       + threep_s_gxDx \
                                       + threep_s_gyDy \
                                       + threep_s_gzDz )

            comm.Allgather( threep_loc, threep[ 1 ] )

        # Loop over flavor
        for iflav in range( flavNum ):

            # Jackknife
            # threep_jk[ iflav, ts, b, t ]

            threep_jk_loc = fncs.jackknifeBinSubset( threep[ iflav ], \
                                                     binSize, \
                                                     bin_glob[ rank ] )

            comm.Gatherv( threep_jk_loc, \
                          [ threep_jk[ imom, iflav, its ], \
                            recvCount * T, \
                            recvOffset * T, \
                            MPI.DOUBLE ], root=0 )
            """
            if rank == 0:

                threep_avg = np.average( threep_jk[ imom, iflav, its ], axis=-2 )
                threep_err = fncs.calcError( threep_jk[ imom, iflav, its ], \
                                             binNum_glob, \
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

"""
if rank == 0 and momSq > 0:

    # Loop over momenta
    for imom in range( momBoostNum ):

        try:
    
            fitResults = fit.mEffTwopFit( mEff, twop_boost_fold[ imom ], \
                                          rangeEnd, momSq, L, tsf, \
                                          tsf_t_low_range=[tsf_fitStart], \
                                          plat_t_low_range=[plat_fitStart], \
                                          checkFit=checkFit, fitType="twop" )
    
        except fit.lqcdjk_BadFitError as error:
        
            mpi_fncs.mpiPrintErr( "ERROR (lqcdjk_fitting.mEffTwopFit):" \
            + str( error ), comm )

        fitParams = fitResults[ 0 ]
        chiSq = fitResults[ 1 ]
        mEff_fit = fitResults[ 2 ]
        rangeStart = fitResults[ 3 ]
        mEff_rangeStart = fitResults[ 4 ]

        if tsf:

            c0_boost = fitParams[ :, 0 ]
            E0_boost = fitParams[ :, 2 ]

        else:

            G = fit_boost[ :, 0 ]
            E = fit_boost[ :, 1 ]
"""
#################
# Calculate <x> #
#################

if rank == 0:
        
    avgX = np.zeros( ( momBoostNum, flavNum, tsinkNum, \
                       binNum_glob, T ) )

else:

    avgX = np.array( [ [ [ None for ts in tsink ] \
                         for f in flav_str ] \
                       for p in momList ] )

if rank == 0:

    # avgX[ p, flav, ts, b, t ]

    # Loop over momenta
    for imom in range( momBoostNum ):

        if momSq > 0:
            
            # Fit the boosted two point functions

            if tsf:

                fitParams,chiSq=fit.twoStateFit_twop(twop_boost_fold[imom], \
                                                     rangeStart, \
                                                     rangeEnd, T )

                c0_boost = fitParams[ :, 0 ]
                E0_boost = fitParams[ :, 2 ]

            else:

                fitParams,chiSq=fit.oneStateFit_twop(twop_boost_fold[imom], \
                                                     rangeStart, \
                                                     rangeEnd, T )

                c0_boost = fitParams[ :, 0 ]
                E0_boost = fitParams[ :, 1 ]

            #print(rangeStart,rangeEnd)
            #print("{:+} {:+} {:+}: {}".format(momList[imom][0],momList[imom][1],momList[imom][2],fitParams))

        # Loop over flavor
        for iflav in range( flavNum ):
            # Loop over tsink
            for ts, its in zip( tsink, range( tsinkNum ) ) :
                
                if momSq > 0:

                    """
                    avgX[imom, \
                    iflav, \
                    its ]=ZvD1*pq.calcAvgX_momBoost(threep_jk[imom, \
                    iflav, \
                    its ], \
                    twop_boost_fold[imom, \
                    :, \
                    ts], \
                    E0_mEff, momSq, \
                    L )
                    """
                    avgX[imom, \
                         iflav, \
                         its]=ZvD1*pq.calcAvgX_twopFit( threep_jk[imom, \
                                                                  iflav, \
                                                                  its ], \
                                                        ts,E0_mEff,momSq,\
                                                        L, c0_boost, \
                                                        E0_boost )

                else:

                    if tsf:

                        avgX[imom, \
                             iflav, \
                             its]=ZvD1*pq.calcAvgX_twopFit( threep_jk[imom, \
                                                                      iflav, \
                                                                      its ], \
                                                            ts, E0_mEff, momSq, \
                                                            L, c0, E0 )

                    else:

                        avgX[imom, \
                             iflav, \
                             its]=ZvD1*pq.calcAvgX_twopFit( threep_jk[imom, \
                                                                      iflav, \
                                                                      its ], \
                                                            ts, E, momSq, \
                                                            L, G, E )

                threep_avg = np.average( threep_jk[imom,iflav,its], \
                                         axis=-2 )
                threep_err = fncs.calcError( threep_jk[imom,iflav,its], \
                                             binNum_glob, axis=-2 )

                avgX_avg = np.average( avgX[imom,iflav,its], \
                                       axis=-2 )
                avgX_err = fncs.calcError( avgX[imom,iflav,its], \
                                           binNum_glob, axis=-2 )

                # Write <x> output files
    
                threep_output_template = "threep_{0}_tsink{1}_{2:+}_{3:+}_{4:+}".format( flav_str[iflav], \
                                                                                         ts, \
                                                                                         momList[imom][0], \
                                                                                         momList[imom][1], \
                                                                                         momList[imom][2] )

                threep_outFilename = output_template.replace( "*", \
                                                              threep_output_template )

                rw.writeAvgDataFile( threep_outFilename, \
                                     threep_avg, \
                                     threep_err )
                    
                avgX_output_template = "avgX_{0}_tsink{1}_{2:+}_{3:+}_{4:+}".format( flav_str[iflav], \
                                                                                     ts, \
                                                                                     momList[imom][0], \
                                                                                     momList[imom][1], \
                                                                                     momList[imom][2] )
                avgX_outFilename = output_template.replace( "*", \
                                                            avgX_output_template )
                rw.writeAvgDataFile( avgX_outFilename, \
                                     avgX_avg, \
                                     avgX_err )

            # End loop over tsink
        # End loop over flavor
    # End loop over momenta

    # Average over momenta

    threep_jk = np.average( threep_jk, axis=0 )
    avgX = np.average( avgX, axis=0 )

    # Average over bins

    # avgX_avg[ flav, ts, t ]

    threep_avg = np.average( threep_jk, axis=-2 )
    threep_err = fncs.calcError( threep_jk, binNum_glob, axis=-2 )

    avgX_avg = np.average( avgX, axis=-2 )
    avgX_err = fncs.calcError( avgX, binNum_glob, axis=-2 )

    # Loop over flavor
    for iflav in range( flavNum ):
        # Loop over tsink
        for ts, its in zip( tsink, range( tsinkNum ) ) :
            
            # Write <x> output files
    
            threep_outFilename = output_template.replace( "*", "threep_" \
                                                        + flav_str[ iflav ] \
                                                        + "_tsink" \
                                                        + str( ts ) )

            rw.writeAvgDataFile( threep_outFilename, \
                                 threep_avg[ iflav, its ], \
                                 threep_err[ iflav, its ] )

            avgX_outFilename = output_template.replace( "*", "avgX_" \
                                                        + flav_str[ iflav ] \
                                                        + "_tsink" \
                                                        + str( ts ) )
            rw.writeAvgDataFile( avgX_outFilename, avgX_avg[ iflav, its ], \
                                 avgX_err[ iflav, its ] )

            ###############
            # Fit plateau #
            ###############

            fitStart = [ ts // 2 - 1, ts // 2 - 2, \
                         ts // 2 - 3, ts // 2 - 4 ]
            
            fitEnd = [ ts // 2 + 1, ts // 2 + 2, \
                       ts // 2 + 3, ts // 2 + 4 ]

            # Loop over fit ranges
            for irange in range( len( fitStart ) ):

                # Fit plateau

                avgX_fit, chiSq = fit.fitPlateau( avgX[ iflav, its ], \
                                                  avgX_err[iflav, its ], \
                                                  fitStart[ irange ], \
                                                  fitEnd[ irange ] )

                # Average over bins

                avgX_fit_avg = np.average( avgX_fit )
                avgX_fit_err = fncs.calcError( avgX_fit, binNum_glob )
                
                # Write output files

                avgX_fit_outFilename=output_template.replace("*", \
                                                             "avgX_" \
                                                             + flav_str[iflav]\
                                                             + "_fit_" \
                                                             "tsink" \
                                                             + str( ts ) \
                                                             + "_" \
                                                             + str(fitStart[irange]) \
                                                             + "_" \
                                                             + str(fitEnd[irange]))

                rw.writeFitDataFile( avgX_fit_outFilename, avgX_fit_avg, avgX_fit_err, fitStart[ irange ], fitEnd[ irange ] )
            
            # End loop over fit ranges
        # End loop over tsink
    # End loop over flavor
# End if first process

##################
# Two-state Fit  #
##################

if tsf and rank == 0:

    mpi_fncs.mpiPrint( "Will perform the two-state fit", rank )

    for iflav in range( flavNum ):

        for neglect in 2,3:

            ti_to_fit = fncs.initEmptyList( tsinkNum, 1 )

            for ts in range( tsinkNum ):

                ti_to_fit[ ts ] = range( neglect, \
                                         tsink[ ts ] - neglect + 1 )
                """
                ti_to_fit[ ts ] = np.concatenate( ( range( neglect, \
                                                           tsink[ ts ] \
                                                           - neglect \
                                                           + 1 ), \
                                                    range( tsink[ ts ] \
                                                           + neglect \
                                                           + 5, \
                                                           T - \
                                                           neglect \
                                                           - 5 + 1 ) ) )
                ti_to_fit[ ts ] = range( tsink[ ts ] + neglect + 5, \
                                         T - neglect - 5 + 1 )
                """

            fitParams, chiSq = fit.twoStateFit_threep( threep_jk[iflav], \
                                                       ti_to_fit, \
                                                       tsink, E0, E1, T )

            a00 = fitParams[ :, 0 ]
            a01 = fitParams[ :, 1 ]
            a11 = fitParams[ :, 2 ]
          
            fitParams = np.stack( ( a00, a01, a11, c0, c1, E0, E1 ), axis=1 )

            # Calculate curve with constant tsink

            # curve[ b, ts, t ]
            # ti[ ts, t ]

            threep_curve, ti_threep = fit.calcThreepTwoStateCurve( a00, a01, \
                                                                   a11, E0, \
                                                                   E1, T, \
                                                                   tsink, \
                                                                   ti_to_fit,\
                                                                   neglect )

            avgX_curve_const_ts, \
                ti_avgX = fit.calcAvgXTwoStateCurve_const_ts( a00, \
                                                              a01, \
                                                              a11, \
                                                              c0, \
                                                              c1, \
                                                              E0, \
                                                              E1, \
                                                              E0_mEff, \
                                                              momSq, \
                                                              L, T, \
                                                              ZvD1, \
                                                              tsink, \
                                                              ti_to_fit, \
                                                              neglect )

            avgX_curve_const_ti, \
                ts_avgX = fit.calcAvgXTwoStateCurve_const_ti( a00, \
                                                              a01, \
                                                              a11, \
                                                              c0, \
                                                              c1, \
                                                              E0, \
                                                              E1, \
                                                              E0_mEff, \
                                                              momSq, \
                                                              L, T, \
                                                              ZvD1, \
                                                              tsink[0]-2, \
                                                              tsink[-1]+5 )



            # avgX[ b ]
            
            avgX = pq.calcAvgX_twoStateFit( a00, c0, E0_mEff, momSq, L, ZvD1 )
        
            # Average over bins
                    
            threep_curve_avg = np.average( threep_curve, axis=0 )
            threep_curve_err = fncs.calcError( threep_curve, binNum_glob )
            
            avgX_curve_const_ts_avg = np.average( avgX_curve_const_ts, \
                                                  axis=0 )
            avgX_curve_const_ts_err = fncs.calcError( avgX_curve_const_ts, \
                                                      binNum_glob )
                
            avgX_curve_const_ti_avg = np.average( avgX_curve_const_ti, \
                                                  axis=0 )
            avgX_curve_const_ti_err = fncs.calcError( avgX_curve_const_ti, \
                                                      binNum_glob )
                
            fitParams_avg = np.average( fitParams, axis=0 )
            fitParams_err = fncs.calcError( fitParams, binNum_glob )

            chiSq_avg = np.average( chiSq, axis=0 )
            chiSq_err = fncs.calcError( chiSq, binNum_glob )
                
            avgX_avg = np.average( avgX )
            avgX_err = fncs.calcError( avgX, binNum_glob )
    
            # Write output file

            tsf_threep_range_str = twopFit_str + ".3n" + str( neglect )

            avgXOutputFilename \
                = output_template.replace( "*", \
                                           "avgX_" \
                                           + flav_str[ iflav ] \
                                           + "_twoStateFit_" \
                                           + tsf_threep_range_str + "_" \
                                           + ts_range_str )
                
            rw.writeFitDataFile( avgXOutputFilename, \
                                 avgX_avg, avgX_err, 0, 0 )
            
            chiSqOutputFilename \
                = output_template.replace( "*", \
                                           "avgX_" \
                                           + flav_str[ iflav ] \
                                           + "_twoStateFit_threep_chiSq_" \
                                           + tsf_threep_range_str + "_" \
                                           + ts_range_str )
        
            rw.writeFitDataFile( chiSqOutputFilename, \
                                 chiSq_avg, chiSq_err, 0, 0 )
            
            avgXParamsOutputFilename \
                = output_template.replace( "*", \
                                           "avgX_" \
                                           + flav_str[ iflav ] \
                                           + "_twoStateFitParams_" \
                                           + tsf_threep_range_str + "_" \
                                           + ts_range_str )

            rw.writeTSFParamsFile( avgXParamsOutputFilename, \
                                   fitParams_avg, fitParams_err )
            
            curveOutputFilename \
                = output_template.replace( "*", \
                                           "avgX_" \
                                           + flav_str[ iflav ] \
                                           + "_twoStateFit_curve_" \
                                           + tsf_threep_range_str + "_" \
                                           + ts_range_str )
            rw.writeAvgDataFile_wX( curveOutputFilename, \
                                    ts_avgX, \
                                    avgX_curve_const_ti_avg, \
                                    avgX_curve_const_ti_err )

            for ts in range( tsinkNum ):
            
                threep_curveOutputFilename \
                    = output_template.replace( "*", \
                                               "threep_" \
                                               + flav_str[ iflav ] \
                                               + "_twoStateFit_curve_tsink" \
                                               + str( tsink[ ts ] ) + "_" \
                                               + tsf_threep_range_str + "_" \
                                               + ts_range_str )
                rw.writeAvgDataFile_wX( threep_curveOutputFilename, \
                                        ti_threep[ ts ], \
                                        threep_curve_avg[ ts ], \
                                        threep_curve_err[ ts ] )

                curveOutputFilename \
                    = output_template.replace( "*", \
                                               "avgX_" \
                                               + flav_str[ iflav ] \
                                               + "_twoStateFit_curve_tsink" \
                                               + str( tsink[ ts ] ) + "_" \
                                               + tsf_threep_range_str + "_" \
                                               + ts_range_str )
                rw.writeAvgDataFile_wX( curveOutputFilename, \
                                        ti_avgX[ ts ], \
                                        avgX_curve_const_ts_avg[ ts ], \
                                        avgX_curve_const_ts_err[ ts ] )
            
            # End loop over tsink
        # End loop over number of neglected three-point functions
    # End loop over flavors
# End if two-state fit
