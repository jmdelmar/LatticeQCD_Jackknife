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
                                + "momentum list to be given.", comm )

    else: # dataFormat == gpu

        mpi_fncs.mpiPrintError( "ERROR: No momentum list given. " \
                                + "I'll fix this later. -CJL", comm )

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

# Read momentum list

finalMomList = []

if dataFormat == "cpu":

    finalMomList = np.array ( rw.getDatasets( twopDir, \
                                              [ configList_loc[0] ], \
                                              twop_template, \
                                              "twop_".format( particle ), \
                                              "ave16", \
                                              "msq{:0>4}".format( momSq ), \
                                              "mvec" )[ 0, 0, 0, ... ].real, \
                              dtype = int )

else:

    if momSq == 0:

        finalMomList = np.array( [ [ 0, 0, 0 ] ] )

    else:

        mpi_fncs.mpiPrintErr( "ERROR: nonzero momenta boost not yet " \
               + "supported for gpu or ASCII format", comm )
        
# Multiply finalMomList by -1 because three-point functions are named
# opposite their sign (sign of phase negative because adjoint taken of
# sequential propagator)

finalMomList = -1 * finalMomList

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

comm.Allgather( twop_loc, twop )

##########################################
# Jackknife and fold two-point functions #
##########################################

# Time dimension length after fold

T_fold = T // 2 + 1

if binNum_loc:

    twop_jk_loc = np.zeros( ( binNum_loc, QNum, T ) )

    for q in range( QNum ):

        twop_jk_loc[:,q,:] = fncs.jackknifeBinSubset( twop[:,q,:], \
                                                      binSize, \
                                                      bin_glob[ rank ] )
    """
    # twop_fold[ b, t ]

    twop_fold = fncs.fold( twop_jk_loc )

    # mEff[ b, t ]

    mEff_loc = pq.mEffFromSymTwop( twop_fold )
    """
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

comm.Gatherv( twop_jk_loc, [ twop_jk, recvCount * QNum * T, \
                             recvOffset * QNum * T, MPI.DOUBLE ], root=0 )
comm.Gatherv( mEff_loc, [ mEff, recvCount * T, \
                          recvOffset * T, MPI.DOUBLE ], root=0 )

if rank == 0:

    # Fit the effective mass and two-point functions 

    mEff_err = fncs.calcError( mEff, binNum_glob )

    mEff_fit = np.zeros( binNum_glob )

    # Loop over bins
    for b in range( binNum_glob ):

        # Perform the plateau fit

        mEff_fit[ b ] = np.polyfit( range( rangeStart_mEff, \
                                           rangeEnd + 1 ), \
                                    mEff[ b, \
                                          rangeStart_mEff \
                                          : rangeEnd + 1 ], 0, \
                                    w=mEff_err[ rangeStart_mEff \
                                                : rangeEnd + 1 ] )

    # End loop over bins
    """
    try:
    
        fitResults = fit.mEffTwopFit( mEff, twop_jk[ :, 0, : ], \
                                      rangeEnd_mEff, 0, L, tsf )
    
    except fit.lqcdjk_BadFitError as error:
        
        mpi_fncs.mpiPrintErr( "ERROR (lqcdjk_fitting.mEffTwopFit):" \
                              + str( error ), comm )

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
    # mEff_avg[ t ]

    mEff_avg = np.average( mEff, axis=0 )
    mEff_err = fncs.calcError( mEff, binNum_glob )

    avgOutputFilename = output_template.replace( "*", "mEff_avg" )
    rw.writeAvgDataFile( avgOutputFilename, mEff_avg, mEff_err )

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

if momSq > 0:

    twop_boost = np.zeros( ( finalMomNum, configNum, T ) )

    comm.Allgather( twop_boost_loc, twop_boost )

else:
    
    twop_boost = np.array( [] )

if rank == 0:
        
    twop_boost_jk = np.zeros( ( finalMomNum, binNum_glob, T ) )

else:

    twop_boost_jk = np.array( [ None for imom in range( finalMomNum ) ] )
    #avgX = np.array( [ [ [ None for ts in tsink ] \
    #                     for f in flav_str ] \
    #                   for imom in range( finalMomNum ) ] )

# Loop over final momenta
for imom in range( finalMomNum ):

    if momSq > 0:

        #########################################
        # Jackknife boosted two-point functions #
        #########################################

        if binNum_loc:

            twop_boost_jk_loc = fncs.jackknifeBinSubset( twop_boost[ imom ], \
                                                         binSize, \
                                                         bin_glob[ rank ] )

        else:

            twop_boost_jk_loc = np.array( [] )

        comm.Gatherv( twop_boost_jk_loc, [ twop_boost_jk[ imom ], \
                                           recvCount * T, recvOffset * T, \
                                           MPI.DOUBLE ], root=0 )

    else:

        twop_boost_jk = np.array( [ twop_jk ] )

# End loop over final momenta

##############################
# Read three-point functions #
##############################

t0 = time.time()

# threep_jk[ ts ][ p, flav, b, Q, ratio, t ]

if rank == 0:

    threep_jk = fncs.initEmptyList( tsinkNum, 1 )

else:

    threep_jk = np.array( [ [ None for ts in tsink ]
                            for imom in range( finalMomNum ) ] )

# Loop over tsink
for ts, its in zip( tsink, range( tsinkNum ) ) :

    if particle == "nucleon":

        threepTimeNum = ts + 1

    else:

        threepTimeNum = T

    if rank == 0:

        threep_jk[ its ] = np.zeros( ( finalMomNum, flavNum, \
                                       binNum_glob, QNum, \
                                       ratioNum, threepTimeNum ) )
    
    # Loop over final momenta
    for ip in range( finalMomNum ):

        t0_ts = time.time()

        # threep_loc[ flav, proj, conf, Q, curr, t ]

        threep_loc = rw.readEMFormFactorFile( threepDir, configList_loc, \
                                              threep_tokens, Qsq, QNum, \
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

            threep_loc[ 0 ] = threep_tmp[ 0 ] - threep_tmp[ 1 ]
            threep_loc[ 1 ] = threep_tmp[ 0 ] + threep_tmp[ 1 ]

        #mpi_fncs.mpiPrintAllRanks(threep_loc.shape, comm)

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

        threep_loc = np.stack ( [ threep_loc[ :, 0, :, :, 0, : ], \
                                  threep_loc[ :, 0, :, :, 1, : ], \
                                  threep_loc[ :, 0, :, :, 2, : ], \
                                  threep_loc[ :, 0, :, :, 3, : ], \
                                  threep_loc[ :, 1, :, :, 2, : ], \
                                  threep_loc[ :, 1, :, :, 3, : ], \
                                  threep_loc[ :, 2, :, :, 1, : ], \
                                  threep_loc[ :, 2, :, :, 3, : ], \
                                  threep_loc[ :, 3, :, :, 1, : ], \
                                  threep_loc[ :, 3, :, :, 2, : ] ], \
                                axis=3 )

        threep = np.zeros( ( flavNum, configNum, QNum, \
                             ratioNum, threepTimeNum ) )

        comm.Allgather( threep_loc, threep )

        # Jackknife
        # threep_jk_loc[ flav, b, Q, ratio, t ]

        if binNum_loc:

            threep_jk_loc = np.zeros( ( flavNum, binNum_loc, \
                                        QNum, ratioNum, \
                                        threepTimeNum ) )

            # Loop over flavor
            for iflav in range( flavNum ):
                # Loop over Q
                for iq in range( QNum ):
                    # Loop over ratio
                    for ir in range( ratioNum ):

                        threep_jk_loc[ iflav, :, iq, ir, : ] = fncs.jackknifeBinSubset( threep[ iflav, :, iq, ir, : ], \
                                                                                        binSize, \
                                                                                        bin_glob[ rank ] )

            # End loop over flavor

        else:

            threep_jk_loc = np.array( [] )
            
        size_WO_bin = flavNum * QNum * ratioNum * threepTimeNum

        comm.Gatherv( threep_jk_loc, \
                      [ threep_jk[ its ][ ip ], \
                        recvCount * size_WO_bin, \
                        recvOffset * size_WO_bin, \
                        MPI.DOUBLE ], root=0 )

    # End loop over p
# End loop over tsink

####################
# Calculate ratios #
####################

mEff_fit = np.random.rand( binNum_glob )
twop_jk = np.random.rand( binNum_glob, QNum, 128 )
threep_jk = fncs.initEmptyList( tsinkNum, 1 )

if rank == 0:

    # Loop over tsink
    for ts, its in zip( tsink, range( tsinkNum ) ) :

        threep_jk[ its ] = np.random.rand( finalMomNum, flavNum, \
                                           binNum_glob, QNum, \
                                           ratioNum, ts+1 )

        # ratio_fit[ p, flav, b, Q, ratio ]

        ratio_fit = np.zeros( ( finalMomNum, flavNum, binNum_glob, \
                                QNum, ratioNum ) )

        # Loop over p
        for ip in range( finalMomNum ) :
            # Loop over flavor
            for iflav in range( flavNum ):

                # ratio[ b, Q, ratio, t ]

                ratio = np.zeros( ( threep_jk[ its ][ ip, iflav ].shape ) )

                # Loop over ratio
                for ir in range( ratioNum ):
                    # Loop over time
                    for t in range( ratio.shape[ -1 ] ):

                        ratio[ :, :, ir, t ] = threep_jk[ its ][ ip, \
                                                                 iflav, \
                                                                 :, :, \
                                                                 ir, t ] \
                            / twop_jk[ ..., ts ]

                ratio_err = fncs.calcError( ratio, binNum_glob )

                ratio_fit[ip, iflav]=fit.fitGenFormFactor(ratio, \
                                                          ratio_err, \
                                                          rangeStart_ratio, \
                                                          rangeEnd_ratio)
            # End loop over flavor
        # End loop over final momentum
    
        # ratio_fit_err[ p, flav, Q, ratio ]

        ratio_fit_err = fncs.calcError( ratio_fit, \
                                        binNum_glob, axis=-3 )

        # Loop over p
        for ip in range( finalMomNum ) :
            # Loop over flavor
            for iflav in range( flavNum ):

                ###############################
                # Calculate kinematic factors #
                ###############################

                # kineFacter[ b, Q, r, [ GE, GM ] ]

                kineFactor = pq.kineFactor_GE_GM( ratio_fit_err[ ip, \
                                                                 iflav ], \
                                                  mEff_fit, Q, L )

                GE = np.zeros( ( QsqNum, binNum_glob ) )
                GM = np.zeros( ( QsqNum, binNum_glob ) )

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

                    ##############################
                    # Calculate ( v s^-1 u^T )^T #
                    ##############################

                    uT = np.transpose( u, ( 0, 2, 1 ) )
                    v = np.transpose( vT, ( 0, 2, 1 ) )

                    smat = np.zeros( ( u.shape[-1], vT.shape[-2] ) )
                    smat_inv = np.zeros( ( binNum_glob, ) \
                                         + np.transpose( smat ).shape )

                    for b in range( binNum_glob ):

                        smat[ :vT.shape[ -2 ], \
                              :vT.shape[ -2 ] ] = np.diag( s[ b ] )

                        smat_inv[ b ] = np.linalg.pinv( smat )

                    # End loop over bins

                    # decomp[ b, Q, ratio, [ GE, GM ] ]

                    decomp=np.transpose(v@smat_inv@uT, \
                                         (0,2,1)).reshape(binNum_glob, \
                                                          Qsq_end[ qsq ] \
                                                          - Qsq_start[ qsq ] \
                                                          + 1, \
                                                          ratioNum,2)

                    decomp_err = fncs.calcError( decomp, binNum_glob )
                    print( "Decomp error:" )
                    print( decomp_err )

                    gE = np.zeros( ( binNum_glob ) )
                    gM = np.zeros( ( binNum_glob ) )

                    gE, gM  = pq.calc_gE_gM( decomp, \
                                             ratio_fit[ ip, \
                                                        iflav ], \
                                             ratio_fit_err[ ip, \
                                                            iflav ], \
                                             Qsq_start[ qsq ], \
                                             Qsq_end[ qsq ] )
                    
                    gE_err = fncs.calcError( gE, binNum_glob )
                    print( "gE error:" )
                    print( gE_err )

                    gM_err = fncs.calcError( gM, binNum_glob )
                    print( "gM error:" )
                    print( gM_err )

                    #print( gE, gM )
                    #exit()

                    GE[ qsq ], GM[ qsq ] = pq.calc_GE_GM( gE, gM, mEff_fit, \
                                                          Qsq[ qsq ], L )

                # End loop over Q^2

                # Average over bins

                GE_avg = np.average( GE, axis=-1 )
                GE_err = fncs.calcError( GE, binNum_glob, axis=-1 )

                GM_avg = np.average( GM, axis=-1 )
                GM_err = fncs.calcError( GM, binNum_glob, axis=-1 )

                ################
                # Write output #
                ################

                output_filename = output_template.replace( "*", \
                                                           particle + "_" \
                                                           + flav_str[iflav] \
                                                           + "_GE_tsink" \
                                                           + str( ts ) )
                rw.writeAvgDataFile_wX( output_filename, Qsq, \
                                           GE_avg, GE_err )

                output_filename = output_template.replace( "*", \
                                                           particle + "_" \
                                                           + flav_str[iflav] \
                                                           + "_GM_tsink" \
                                                           + str( ts ) )
                rw.writeAvgDataFile_wX( output_filename, Qsq, \
                                           GM_avg, GM_err )

            # End loop over flavor
        # End loop over final momentum
    # End loop over tsink
# End if first process
