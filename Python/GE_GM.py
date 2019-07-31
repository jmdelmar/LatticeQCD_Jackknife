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

tsf = True

rangeEnd = 25

L = 64.0
ratioNum = 10

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

parser.add_argument( "twop_fit_range_end", action='store', type=int )

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

# Last point to fit two-point functions

rangeEnd_twop = args.twop_fit_range_end

# First and last points to fit ratios

rangeStart_ratio = args.ratio_fit_range_end
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

elif particle == "kaon":

    projector = [ "" ]

    flavNum = 2
    flav_str = [ "u", "s" ]

else: # particle == nulceon

    projector = [ "0", "4", "5", "6" ]

    flavNum = 2
    flav_str = [ "u", "d" ]

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

mpi_fncs.mpiPrint( "Read two-point functions from HDF5 files " \
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

    # mEff_avg[ t ]

    mEff_avg = np.average( mEff, axis=0 )
    mEff_err = fncs.calcError( mEff, binNum_glob )

    avgOutputFilename = output_template.replace( "*", "mEff_avg" )
    rw.writeAvgDataFile( avgOutputFilename, mEff_avg, mEff_err )

    # Fit the effective mass and two-point functions 

    try:
    
        fitResults = fit.mEffTwopFit( mEff, twop_jk[ :, 0, : ], \
                                      rangeEnd_twop, 0, L, tsf )
    
    except fit.lqcdjk_BadFitError as error:
        
        mpi_fncs.mpiPrintErr( "ERROR (lqcdjk_fitting.mEffTwopFit):" \
                              + str( error ), comm )

    fitParams = fitResults[ 0 ]
    chiSq = fitResults[ 1 ]
    mEff_fit = fitResults[ 2 ]
    rangeStart = fitResults[ 3 ]
    mEff_rangeStart = fitResults[ 4 ]

    curve = np.zeros( ( binNum_glob, 50 ) )

    t_s = np.concatenate( ( np.linspace( rangeStart, \
                                         rangeEnd_twop, 25 ), \
                            np.linspace( T - rangeEnd_twop, \
                                         T- rangeStart, 25 ) ) )
                    
    twopFit_str = "2s" + str( rangeStart ) \
                  + ".2e" + str( rangeEnd_twop )

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
                         chiSq_err, rangeStart, rangeEnd_twop )

    mEff_fit_avg = np.average( mEff_fit, axis=0 )
    mEff_fit_err = fncs.calcError( mEff_fit, binNum_glob )

    mEff_range_str = "2s" + str( mEff_rangeStart ) \
                     + ".2e" + str( rangeEnd_twop )

    mEff_outputFilename = output_template.replace( "*", "mEff_fit_" \
                                                   + mEff_range_str )
    rw.writeFitDataFile( mEff_outputFilename, mEff_fit_avg, \
                         mEff_fit_err, mEff_rangeStart, rangeEnd_twop )

# End if first process

if momSq > 0:

    twop_boost = np.zeros( ( finalMomNum, configNum, T ) )

    comm.Allgather( twop_boost_loc, twop_boost )

else:
    
    twop_boost = np.array( [] )

if rank == 0:
        
    twop_boost_jk = np.zeros( ( finalMomNum, binNum_glob, T ) )
    threep_jk = np.zeros( ( finalMomNum, flavNum, \
                            tsinkNum, binNum_glob, T ) )

else:

    twop_boost_jk = np.array( [ None for imom in range( finalMomNum ) ] )
    threep_jk = np.array( [ [ [ None for ts in tsink ] \
                              for f in flav_str ] \
                            for imom in range( finalMomNum ) ] )
    #avgX = np.array( [ [ [ None for ts in tsink ] \
    #                     for f in flav_str ] \
    #                   for imom in range( finalMomNum ) ] )

comm.Barrier()

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

# threep[ ts ][ p, flav, proj, conf, Qsq, curr, t ]

threep = fncs.initEmptyList( tsinkNum, 1 )

# Loop over tsink
for ts, its in zip( tsink, range( tsinkNum ) ) :

    if particle == "nucleon":

        threep[ its ] = np.zeros( ( finalMomNum, flavNum, \
                                    projNum, configNum, \
                                    QNum, 4, ts + 1 ) )

    else:

        threep[ its ] = np.zeros( ( finalMomNum, flavNum, \
                                    projNum, configNum, \
                                    QNum, 4, T ) )
    
    # Loop over final momenta
    for imom in range( finalMomNum ):

        t0_ts = time.time()

        # threep_loc[ flav, proj, conf, Qsq, curr, t ]

        threep_loc = rw.readEMFormFactorFile( threepDir, configList_loc, \
                                              threep_tokens, Qsq, QNum, \
                                              ts, projector, \
                                              finalMomList[ imom ], \
                                              particle, dataFormat )

        mpi_fncs.mpiPrint( "Read three-point functions from HDF5 files " \
                           + "for tsink {} in {:.4}".format( ts, \
                                                             time.time() \
                                                             - t0_ts ) \
                           + " seconds.", rank )

        mpi_fncs.mpiPrintAllRanks(threep_loc.shape, comm)

        comm.Allgather( threep_loc, threep[ its ][ imom ] )

        """
        # Loop over flavor
        for iflav in range( flavNum ):

            # Jackknife
            # threep_jk[ iflav, ts, b, t ]

            threep_jk_loc = fncs.jackknifeBinSubset( threep[ iflav ], \
                                                     binSize, \
                                                     bin_glob[ rank ] )

            comm.Gatherv( threep_jk_loc, [ threep_jk[ imom, iflav, its ], \
                                           recvCount * T, recvOffset * T, \
                                           MPI.DOUBLE ], root=0 )
        """
        # End loop over flavor
    # End loop over tsink
# End loop over momenta

exit()
if rank == 0:
        
    avgX = np.zeros( ( flavNum, \
                       tsinkNum, binNum_glob, T ) )

else:

    avgX = np.array( [ [ None for ts in tsink ] \
                         for f in flav_str ] )

# CJL: Just calculate the average

#################
# Calculate <x> #
#################

if rank == 0:

    # Average over momenta

    threep_jk = np.average( threep_jk, axis=0 )
    twop_boost_jk = np.average( twop_boost_jk, axis=0 )
    
    # avgX[ flav, ts, b, t ]

    # Loop over tsink
    for ts, its in zip( tsink, range( tsinkNum ) ) :
        # Loop over flavor
        for iflav in range( flavNum ):

            if momSq > 0:

                avgX[iflav,its]=ZvD1*pq.calcAvgX_momBoost(threep_jk[iflav, \
                                                                    its], \
                                                          twop_boost_jk[:, \
                                                                        ts], \
                                                          mEff_fit, \
                                                          momSq, L )
            else:

                preFactor = np.repeat( -ZvD1 * 8.0 / 3.0 / mEff_fit, \
                                       T ).reshape( binNum_glob, T )

                if tsf:

                    c0_cp = np.repeat( c0, T ).reshape( binNum_glob, T )
                    c1_cp = np.repeat( c1, T ).reshape( binNum_glob, T )
                    E0_cp = np.repeat( E0, T ).reshape( binNum_glob, T )
                    E1_cp = np.repeat( E1, T ).reshape( binNum_glob, T )

                    avgX[ iflav, its ] = preFactor * threep_jk[ iflav, its ] \
                                         / fit.twoStateTwop( ts, T, \
                                                             c0_cp, c1_cp, \
                                                             E0_cp, E1_cp )
        
                else:

                    G_cp = np.repeat( G, T ).reshape( binNum_glob, T )
                    E_cp = np.repeat( E, T ).reshape( binNum_glob, T )

                    avgX[ iflav, its ] = preFactor * threep_jk[ iflav, its ] \
                                         / fit.oneStateTwop( ts, T, G_cp, E_cp )
                    
            #avgX = Z * pq.calcAvgX( threep_jk[ -1 ], \
            #                        twop_jk[ :, ts ], mEff_fit )

    # Average over bins

    # avgX_avg[ flav, ts, t ]

    avgX_avg = np.average( avgX, axis=-2 )
    avgX_err = fncs.calcError( avgX, binNum_glob, axis=-2 )

    # Loop over tsink
    for ts, its in zip( tsink, range( tsinkNum ) ) :
        # Loop over flavor
        for iflav in range( flavNum ):

            # Write <x> output files
    
            avgX_outFilename = output_template.replace( "*", "avgX_" \
                                                        + flav_str[ iflav ] \
                                                        + "_tsink" \
                                                        + str( ts ) )

            rw.writeAvgDataFile( avgX_outFilename, avgX_avg[ iflav, its ], \
                                 avgX_err[ iflav, its ] )

            ###############
            # Fit plateau #
            ###############

            fitStart = [ ts // 2 - 1, ts // 2 - 2, ts // 2 - 3 ]
            
            fitEnd = [ ts // 2 + 1, ts // 2 + 2, ts // 2 + 3 ]

            # Loop over fit ranges
            for irange in range( len( fitStart ) ):

                avgX_fit = []
        
                # Fit each bin

                for x in avgX[ iflav, its ]:

                    avgX_fit.append(float(np.polyfit(range(fitStart[irange], \
                                                           fitEnd[ irange ] \
                                                           + 1 ), \
                                                     x[ fitStart[ irange ] \
                                                        : fitEnd[ irange ] \
                                                        + 1 ], \
                                                     0, \
                                                     w=avgX_err[iflav, its, \
                                                                fitStart[irange] \
                                                                :fitEnd[irange] \
                                                                + 1 ] ) ) )

                avgX_fit = np.array( avgX_fit )

                # Average over bins

                avgX_fit_avg = np.average( avgX_fit )
                
                avgX_fit_err = fncs.calcError( avgX_fit, binNum_glob )
                
                # Write output files

                avgX_fit_outFilename = ""

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
        # End loop over flavor
    # End loop over tsink
# End if first process
comm.Barrier()
