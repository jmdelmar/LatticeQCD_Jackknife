from sys import stderr
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

#tsf = True

Z = 1.0

rangeEnd = 25

L = 64.0

particle_list = fncs.particleList()

format_list = fncs.dataFormatList()

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Perform SVD for GE and GM" )

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

parser.add_argument( "mEff_fit_range_start", action='store', type=int )

parser.add_argument( "mEff_fit_range_end", action='store', type=int )

parser.add_argument( "ratio_fit_range_start", action='store', type=int )

parser.add_argument( "ratio_fit_range_end", action='store', type=int )

parser.add_argument( "particle", action='store', \
                     help="Particle to calculate GE/GM for. " \
                     + "Should be 'pion', 'kaon', or 'nucleon'.", \
                     type=str )

parser.add_argument( 't_sink', action='store', \
                     help="Comma seperated list of t sink's", \
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "threep_final_momentum_squared", \
                     action='store', type=int )

parser.add_argument( "binSize", action='store', type=int )

parser.add_argument( "-o", "--output_template", action='store', \
                     type=str, default="./*.dat" )

oparser.add_argument( "-sn", "--source_number", action='store', type=int, \
                     help="Number of sources correlators were averaged " \
                     + "over", default=16 )

parser.add_argument( "-f", "--data_format", action='store', \
                     help="Data format. Should be 'gpu', " \
                     + "'cpu', or 'ASCII'.", \
                     type=str, default="gpu" )

parser.add_argument( "-c", "--config_list", action='store', \
                     type=str, default="" )

parser.add_argument( "-m", "--momentum_transfer_list", action='store', \
                     type=str, default="" )

args = parser.parse_args()

#########
# Setup #
#########

# Set MPI values

mpi_info = mpi_fncs_init()

comm = MPI.COMM_WORLD
procNum = comm.Get_size()
rank = comm.Get_rank()

# Input directories and filename templates

threepDir = args.threep_dir

twopDir = args.twop_dir

threep_tokens = args.threep_tokens

twop_template = args.twop_template

# First and last points to fit ratios

rangeStart_mEff = args.mEff_fit_range_start
rangeEnd_mEff = args.mEff_fit_range_end

# First and last points to fit ratios

rangeStart_ratio = args.ratio_fit_range_start
rangeEnd_ratio = args.ratio_fit_range_end

# Info on what to analyze

particle = args.particle

tsink = args.t_sink
tsinkNum = len( tsink )
ts_range_str = "tsink" + str(tsink[0]) + "_" + str(tsink[-1])

# Other info

binSize = args.binSize

output_template = args.output_template

srcNum = args.source_number

dataFormat = args.data_format

momSq = args.threep_final_momentum_squared

# Get configurations from given list or from given 
# threep directory if list not given

configList = np.array( fncs.getConfigList( args.config_list, threepDir ) )
configNum = len( configList )

# Check inputs

assert particle in particle_list, \
    "Error: Particle not supported. " \
    + "Supported particles: " + str( particle_list )

assert dataFormat in format_list, \
    "Error: Data format not supported. " \
    + "Supported particles: " + str( format_list )

assert configNum % binSize == 0, "Number of configurations " \
    + str( configNum ) + " not evenly divided by bin size " \
    + str( binSize ) + "."

assert configNum % procNum == 0, "Number of configurations " \
    + str( configNum ) + " not evenly divided by number of processes " \
    + str( procNum ) + "."

if particle == "pion":

    projector = [ "" ]

    flavNum = 1
    flav_str = [ "u" ]

    ratioNum = 10

elif particle == "kaon":

    projector = [ "" ]

    flavNum = 2
    flav_str = [ "u", "s" ]

    ratioNum = 10

elif particle == "nucleon":

    projector = [ "0", "4", "5", "6" ]

    flavNum = 2
    flav_str = [ "IV", "IS" ]

    ratioNum = 10

projNum = len( projector )

# Momentum list

if args.momentum_transfer_list:

    Q = rw.readTxtFile( args.momentum_transfer_list, dtype=int )

    if dataFormat == "ASCII":
        
        Q = -1.0 * Q

else:

    if dataFormat == "gpu":

        mpi_fncs.mpiPrint( "No momentum list given, will read momentum " \
                           + "from three-point function files", rank )

        Q = rw.getDatasets( threepDir, configList, threep_template, \
                                  "Momenta_list" )[ :, 0, 0, ... ]
    elif dataFormat == "ASCII":

        mpi_fncs.mpiPrintError( "ERROR: ASCII format requires a " \
                                + "momentum list to be given.", mpi_info )

    else: # dataFormat == gpu

        mpi_fncs.mpiPrintError( "ERROR: No momentum list given. " \
                                + "I'll fix this later. -CJL", mpi_info )

QNum = len( Q )

Qsq, Qsq_start, Qsq_end = fncs.processMomList( Q )

QsqNum = len( Qsq )

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

# Read final momentum list

finalMomList = rw.readMomentaList( twopDir, twop_template, \
                                   configList_loc[ 0 ], particle, \
                                   srcNum, momSq, dataFormat, mpi_info )

finalMomNum = len( finalMomList )

############################
# Read Two-point Functions #
############################

# Zero momentum two-point functions
# twop[ c, q, t ]

t0 = time.time()

if dataFormat == "cpu":

    twop_loc = rw.getDatasets( twopDir, configList_loc, twop_template, \
                               "msq0000", "arr" )[ :, 0, 0, :, 0 ].real
    # CJL: add non-zero Q functionality

elif dataFormat == "ASCII":

    # Determine length of time dimension.
    # 2nd output is not really configuration number because
    # files are not formatted like that.

    T, dummy = rw.detTimestepAndConfigNum( twopDir + \
                                           twop_template.replace( "*", \
                                                                  configList_loc[0] ) )

    # Get 5th column of two-point files for each configuration

    twop_loc = rw.getTxtData( twopDir, \
                              configList_loc, \
                              twop_template, \
                              dtype=float).reshape( len( configList_loc ), \
                                                    QNum, T, 6 )[ ..., 4 ]

else:
        
    twop_loc = rw.getDatasets( twopDir, configList_loc, \
                               twop_template, \
                               "twop" )[ :, 0, 0, ..., 0, 0 ]
    # CJL: Have to check which axis Q is on


twop_loc = np.asarray( twop_loc, order='c' )

mpi_fncs.mpiPrint( "Read two-point functions from files " \
                   + "in {:.3} seconds".format( time.time() - t0 ), rank )

T = twop_loc.shape[ -1 ]

# Gather two-point functions

twop = np.zeros( ( configNum, QNum, T ), dtype=float )

mpi_info[ 'comm' ].Allgather( twop_loc, twop )

#################################
# Jackknife two-point functions #
#################################

if binNum_loc:

    twop_jk_loc = np.zeros( ( binNum_loc, QNum, T ) )

    for q in range( QNum ):

        twop_jk_loc[:,q,:] = fncs.jackknifeBinSubset( twop[:,q,:], \
                                                      binSize, \
                                                      bin_glob[ rank ] )
    mEff_loc = pq.mEff( twop_jk_loc[:,0,:] )

else:

    twop_jk_loc = np.array( [] )

    mEff_loc = np.array( [] )

##################
# Effective mass #
##################

if rank == 0:

    twop_jk = np.zeros( ( binNum_glob, QNum, T ) )
    mEff = np.zeros( ( binNum_glob, T ) )

else:

    twop_jk = None
    mEff = None

recvCount, recvOffset = mpi_fncs.recvCountOffset( procNum, binNum )

#mpi_info[ 'comm' ].Gatherv( twop_jk_loc, [ twop_jk, recvCount * QNum * T, \
#                             recvOffset * QNum * T, MPI.DOUBLE ], root=0 )
mpi_info[ 'comm' ].Gatherv( mEff_loc, [ mEff, recvCount * T, \
                          recvOffset * T, MPI.DOUBLE ], root=0 )

if rank == 0:

    # mEff_avg[ t ]

    mEff_avg = np.average( mEff, axis=0 )
    mEff_err = fncs.calcError( mEff, binNum_glob )

    avgOutputFilename = output_template.replace( "*", "mEff_avg" )
    rw.writeAvgDataFile( avgOutputFilename, mEff_avg, mEff_err )

    # Fit the effective mass and two-point functions 

    mEff_fit = np.zeros( binNum_glob )

    # Perform the plateau fit

    mEff_fit, chiSq = fit.fitPlateau( mEff, mEff_err, \
                                      rangeStart_mEff, \
                                      rangeEnd )

    # End loop over bins
    """
    try:
    
        fitResults = fit.mEffTwopFit( mEff, twop_jk[ :, 0, : ], \
                                      rangeEnd_mEff, 0, L, tsf )
    
    except fit.lqcdjk_BadFitError as error:
        
        mpi_fncs.mpiPrintErr( "ERROR (lqcdjk_fitting.mEffTwopFit):" \
                              + str( error ), mpi_info )

    fitParams = fitResults[ 0 ]
    chiSq = fitResults[ 1 ]
    mEff_fit = fitResults[ 2 ]
    rangeStart = fitResults[ 3 ]
    rangeStart_mEff = fitResults[ 4 ]

    curve = np.zeros( ( binNum_glob, 50 ) )

    t_s = np.concatenate( ( np.linspace( rangeStart, \
                                         rangeEnd_mEff, 25 ), \
                            np.linspace( T - rangeEnd_mEff, \
                                         T- rangeStart, 25 ) ) )
                    
    twopFit_str = "2s" + str( rangeStart ) \
                  + ".2e" + str( rangeEnd_mEff )

    if tsf:

        c0 = fitParams[ :, 0 ]
        c1 = fitParams[ :, 1 ]
        E0 = fitParams[ :, 2 ]
        E1 = fitParams[ :, 3 ]

        # Calculate fitted curve

        for b in range( binNum_glob ):

            for t in range( t_s.shape[ -1 ] ):
                
                curve[ b, t ] = fit.twoStateTwop( t_s[ t ], T, \
                                                  c0[ b ], c1[ b ], \
                                                  E0[ b ], E1[ b ] )
                        
            # End loop over tsink
        # End loop over bins

        curveOutputFilename \
            = output_template.replace( "*", \
                                       "twop_twoStateFit_curve_" \
                                       + twopFit_str )
        chiSqOutputFilename \
            = output_template.replace( "*", \
                                       "twop_twoStateFit_chiSq_" \
                                       + twopFit_str )

    else: # One-state fit

        G = fitParams[ :, 0 ]
        E = fitParams[ :, 1 ]

        # Calculate fitted curve

        for b in range( binNum_glob ):

            for t in range( t_s.shape[ -1 ] ):
                
                curve[ b, t ] = fit.oneStateTwop( t_s[ t ], T, \
                                                  G[ b ], E[ b ] )
                        
            # End loop over tsink
        # End loop over bins

        curveOutputFilename \
            = output_template.replace( "*", \
                                       "twop_oneStateFit_curve_" \
                                       + twopFit_str )
        chiSqOutputFilename \
            = output_template.replace( "*", \
                                       "twop_oneStateFit_chiSq_" \
                                       + twopFit_str )

    # End if not two-state fit

    curve_avg = np.average( curve, axis=0 )
    curve_err = fncs.calcError( curve, binNum_glob )
            
    chiSq_avg = np.average( chiSq, axis=0 )
    chiSq_err = fncs.calcError( chiSq, binNum_glob )
            
    # Write output files

    rw.writeAvgDataFile_wX( curveOutputFilename, t_s, curve_avg, curve_err )
        
    rw.writeFitDataFile( chiSqOutputFilename, chiSq_avg, \
                         chiSq_err, rangeStart, rangeEnd_mEff )
    """
    # mEff_fit_avg

    mEff_fit_avg = np.average( mEff_fit, axis=0 )
    mEff_fit_err = fncs.calcError( mEff_fit, binNum_glob )

    mEff_range_str = "2s" + str( rangeStart_mEff ) \
                     + ".2e" + str( rangeEnd_mEff )

    mEff_outputFilename = output_template.replace( "*", "mEff_fit_" \
                                                   + mEff_range_str )
    rw.writeFitDataFile( mEff_outputFilename, mEff_fit_avg, \
                         mEff_fit_err, rangeStart_mEff, rangeEnd_mEff )

# End if first process
"""
if momSq > 0:

    twop_boost = np.zeros( ( finalMomNum, configNum, T ) )

    mpi_info[ 'comm' ].Allgather( twop_boost_loc, twop_boost )

else:
    
    twop_boost = np.array( [] )

if rank == 0:
        
    twop_boost_jk = np.zeros( ( finalMomNum, binNum_glob, T ) )

else:

    twop_boost_jk = np.array( [ None for imom in range( finalMomNum ) ] )
    #avgX = np.array( [ [ [ None for ts in tsink ] \
    #                     for f in flav_str ] \
    #                   for imom in range( finalMomNum ) ] )
"""
"""
if momSq > 0:

    
    twop_boost = rw.readTwopFile_zeroQ( twopDir, configList_loc, configNum, \
                                        twop_template, srcNum, momSq, particle, \
                                        dataFormat, mpi_info )

    # Loop over final momenta
    for imom in range( finalMomNum ):

        #########################################
        # Jackknife boosted two-point functions #
        #########################################

        if binNum_loc:

            twop_boost_jk = fncs.jackknifeBinSubset( twop_boost[ imom ], \
                                                         binSize, \
                                                         bin_glob[ rank ] )

        else:

            twop_boost_jk = np.array( [] )

        mpi_info[ 'comm' ].Gatherv( twop_boost_jk, [ twop_boost_jk[ imom ], \
                                           recvCount * T, recvOffset * T, \
                                           MPI.DOUBLE ], root=0 )

    # End loop over final momenta

else:

    twop_boost_jk = np.array( [ twop_jk_loc ] )
"""
##############################
# Read three-point functions #
##############################

"""
# threep_jk[ ts ][ p, flav, b, Q, ratio, t ]

if rank == 0:

    threep_jk = fncs.initEmptyList( tsinkNum, 1 )

else:

    threep_jk = np.array( [ [ None for ts in tsink ]
                            for imom in range( finalMomNum ) ] )
"""
# Loop over tsink
for ts, its in zip( tsink, range( tsinkNum ) ) :

    if particle == "nucleon":

        threepTimeNum = ts + 1

    else:

        threepTimeNum = T

    # Loop over final momenta
    for ip in range( finalMomNum ):

        t0_ts = time.time()

        # threep_loc[ flav, proj, conf, Q, curr, t ]

        threep_loc = rw.readEMFormFactorFile( threepDir, configList_loc, \
                                              threep_tokens, srcNum, Qsq, QNum, \
                                              ts, projector, \
                                              finalMomList[ ip ], \
                                              particle, dataFormat )

        mpi_fncs.mpiPrint( "Read three-point functions from files " \
                           + "for tsink {} in {:.4}".format( ts, \
                                                             time.time() \
                                                             - t0_ts ) \
                           + " seconds.", rank )

        if particle == "nucleon":

            # Calculate isovector and isoscalar

            threep_tmp = np.copy( threep_loc )

            threep_loc[ 0 ] = 0.5 * ( threep_tmp[ 0 ] \
                                      - threep_tmp[ 1 ] )
            threep_loc[ 1 ] = 0.5 * ( threep_tmp[ 0 ] \
                                      + threep_tmp[ 1 ] )

        #mpi_fncs.mpiPrintAllRanks(threep_loc.shape, mpi_info)

        # Get the projection and insertion combinations we want
        # threep_loc[ flav, proj, conf, Q, curr, t ]
        # -> threep_loc[ flav, conf, Q, ratio, t ]

        # ratio   ProjInsertion
        # 0       P0g0
        # 1       P0g1
        # 2       P0g2
        # 3       P0g3
        # 4       P4g2
        # 5       P4g3
        # 6       P5g1
        # 7       P5g3
        # 8       P6g1
        # 9       P6g2
        # CJL: this might be different for mesons

        threep_loc = np.stack ( [ threep_loc[ :, 0, :, :, 0, : ].real, \
                                  threep_loc[ :, 0, :, :, 1, : ].imag, \
                                  threep_loc[ :, 0, :, :, 2, : ].imag, \
                                  threep_loc[ :, 0, :, :, 3, : ].imag, \
                                  threep_loc[ :, 1, :, :, 2, : ].real, \
                                  threep_loc[ :, 1, :, :, 3, : ].real, \
                                  threep_loc[ :, 2, :, :, 1, : ].real, \
                                  threep_loc[ :, 2, :, :, 3, : ].real, \
                                  threep_loc[ :, 3, :, :, 1, : ].real, \
                                  threep_loc[ :, 3, :, :, 2, : ].real ], \
                                axis=3 )
        
        threep = np.zeros( ( flavNum, configNum, QNum, \
                             ratioNum, threepTimeNum ) )

        # Allgather threep for each flavor so that 
        # everything is in the correct order

        # Loop over flavor
        for iflav in range( flavNum ):

            mpi_info[ 'comm' ].Allgather( threep_loc[ iflav ], \
                            threep[ iflav ] )

        # End loop over flavor

        # Loop over flavor
        for iflav in range( flavNum ):
            # If bin on this process
            if binNum_loc:

                ratio_loc = np.zeros( ( binNum_loc, QNum, \
                                        ratioNum, threepTimeNum ) )

                # Loop over ratio
                for ir in range( ratioNum ):

                    # Jackknife
                    # threep_jk[ b, Q, t ]

                    threep_jk = np.zeros( ( binNum_loc, QNum, \
                                            threepTimeNum ) )

                    # Loop over Q
                    for iq in range( QNum ):

                        threep_jk[:, \
                                  iq, \
                                  :]=fncs.jackknifeBinSubset(threep[iflav, \
                                                                    :, iq, \
                                                                    ir, \
                                                                    : ], \
                                                             binSize, \
                                                             bin_glob[rank])

                    # End loop over Q

                    #mpi_fncs.mpiPrint(threep_jk,rank)

                    ####################
                    # Calculate ratios #
                    ####################

                    ratio_loc[...,ir,:] = pq.calcRatio_Q( threep_jk, \
                                                          twop_boost_jk[ ip ], \
                                                          ts )

                # End loop over ratio
            # End if bin on process
            else:

                ratio_loc = np.array( [] )
        
            ratio = np.zeros( ( binNum_glob, QNum, \
                                ratioNum, threepTimeNum ) )

            mpi_info[ 'comm' ].Allgatherv( ratio_loc, \
                             [ ratio, \
                               recvCount * QNum \
                               * ratioNum * threepTimeNum, \
                               recvOffset * QNum \
                               * ratioNum * threepTimeNum, \
                               MPI.DOUBLE ] )

            ratio_err = fncs.calcError( ratio, binNum_glob )

            #mpi_fncs.mpiPrintAllRanks(ratio_err,mpi_info)

            if binNum_loc:

                # ratio_fit[ b, Q, ratio ]
             
                ratio_fit_loc=fit.fitGenFormFactor(ratio_loc, \
                                                   ratio_err, \
                                                   rangeStart_ratio, \
                                                   rangeEnd_ratio)

            else: 

                ratio_fit_loc = []

            ratio_fit = np.zeros( ( binNum_glob, QNum, ratioNum ) )

            mpi_info[ 'comm' ].Gatherv( ratio_fit_loc, \
                          [ ratio_fit, \
                            recvCount * QNum * ratioNum, \
                            recvOffset * QNum * ratioNum, \
                            MPI.DOUBLE ], root=0 )

            # ratio_fit_err[ Q, ratio ]

            ratio_fit_err = fncs.calcError( ratio_fit, binNum_glob )        
            
            if rank == 0:

                ###############################
                # Calculate kinematic factors #
                ###############################

                # kineFacter[ b, Q, r, [ GE, GM ] ]
                
                #print("ratio_fit_err:")
                #print(ratio_fit_err)
                #print("mEff_fit")                
                #print(mEff_fit[0])

                kineFactor = pq.kineFactor_GE_GM( ratio_fit_err, \
                                                  mEff_fit, Q, L )

                #print(kineFactor[0])

                GE = np.zeros( ( QsqNum, binNum_glob ) )
                GM = np.zeros( ( QsqNum, binNum_glob ) )

                gE = np.zeros( ( QsqNum, binNum_glob ) )
                gM = np.zeros( ( QsqNum, binNum_glob ) )

                for qsq in range( QsqNum ):

                    ###############
                    # Perform SVD #
                    ###############

                    kineFactor_Qsq \
                        = kineFactor[ :, \
                                      Qsq_start[ qsq ]:Qsq_end[ qsq ] + 1, \
                                      ... ].reshape( binNum_glob, \
                                                     ( Qsq_start[ qsq ] \
                                                       - Qsq_end[ qsq ] \
                                                       + 1 ) \
                                                     * ratioNum, 2 )

                    u, s, vT = np.linalg.svd( kineFactor_Qsq, \
                                              full_matrices=False )

                    #print("U")
                    #print(u[0])
                    #print("S")
                    #print(s[0])
                    #print("V^T")
                    #print(vT[0])

                    ##############################
                    # Calculate ( v s^-1 u^T )^T #
                    ##############################

                    uT = np.transpose( u, ( 0, 2, 1 ) )
                    v = np.transpose( vT, ( 0, 2, 1 ) )

                    #print("U^T")
                    #print(uT[0])
                    #print("V")
                    #print(v[0])

                    smat = np.zeros( ( u.shape[-1], vT.shape[-2] ) )
                    smat_inv = np.zeros( ( binNum_glob, ) \
                                         + np.transpose( smat ).shape )

                    for b in range( binNum_glob ):

                        smat[ :vT.shape[ -2 ], \
                              :vT.shape[ -2 ] ] = np.diag( s[ b ] )

                        smat_inv[ b ] = np.linalg.pinv( smat )

                    # End loop over bins

                    #print("S")
                    #print(smat)
                    #print("S^-1")
                    #print(smat_inv[0])

                    # decomp[ b, Q, ratio, [ GE, GM ] ]

                    decomp=np.transpose(v@smat_inv@uT, \
                                         (0,2,1))
                    
                    decomp = decomp.reshape(binNum_glob, \
                                            Qsq_end[ qsq ] \
                                            - Qsq_start[ qsq ] \
                                            + 1, \
                                            ratioNum,2)

                    gE[ qsq ], gM[ qsq ]  = pq.calc_gE_gM( decomp, \
                                                           ratio_fit, \
                                                           ratio_fit_err, \
                                                           Qsq_start[ qsq ],\
                                                           Qsq_end[ qsq ] )
                    
                # End loop over Q^2

                # Average over bins

                gE_avg = np.average( gE, axis=-1 )
                gE_err = fncs.calcError( gE, binNum_glob, axis=-1 )

                gM_avg = np.average( gM, axis=-1 )
                gM_err = fncs.calcError( gM, binNum_glob, axis=-1 )

                ################
                # Write output #
                ################

                output_filename = output_template.replace( "*", \
                                                           particle + "_" \
                                                           + flav_str[iflav] \
                                                           + "_GE_tsink" \
                                                           + str( ts ) )
                rw.writeAvgDataFile_wX( output_filename, Qsq, \
                                           gE_avg, gE_err )

                output_filename = output_template.replace( "*", \
                                                           particle + "_" \
                                                           + flav_str[iflav] \
                                                           + "_GM_tsink" \
                                                           + str( ts ) )
                rw.writeAvgDataFile_wX( output_filename, Qsq, \
                                           gM_avg, gM_err )

            # End if first process
        # End loop over flavor
    # End loop over p
# End loop over tsink
