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

Z = 1.0

L = 32

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

momList = []

if dataFormat == "cpu":

    momList = np.array ( rw.getDatasets( twopDir, [ configList_loc[0] ], \
                                         twop_template, \
                                         "twop_".format( particle ), \
                                         "ave16", \
                                         "msq{:0>4}".format( momSq ), \
                                         "mvec" )[ 0, 0, 0, ... ].real, \
                         dtype = int )

else:

    if momSq == 0:

        momList = np.array( [ [ 0, 0, 0 ] ] )

    else:

        mpi_fncs.mpiPrintErr( "ERROR: nonzero momenta boost not yet " \
               + "supported for gpu format" )
        
        exit()

# Multiply momList by -1 because three-point functions are named
# opposite their sign (sign of phase negative because adjoint taken of
# sequential propagator)

momList = -1 * momList

momBoostNum = len( momList )

############################
# Read Two-point Functions #
############################

# Zero momentum two-point functions
# twop[ c, t ]

t0 = time.time()

if dataFormat == "cpu":

    twop_loc = rw.getDatasets( twopDir, configList_loc, twop_template, \
                               "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

else:
        
    twop_loc = rw.getDatasets( twopDir, configList_loc, \
                               twop_template, \
                               "twop" )[ :, 0, 0, ..., 0, 0 ]

twop_loc = np.asarray( twop_loc, order='c' )

mpi_fncs.mpiPrint( "Read two-point functions from HDF5 files " \
                   + "in {:.3} seconds".format( time.time() - t0 ), rank )

# Boosted two-point functions
# twop_boost[ mom, c, t ]

if momSq > 0:

    t0 = time.time()

    if dataFormat == "cpu":

        twop_boost_loc = rw.getDatasets( twopDir, configList_loc, \
                                         twop_template, \
                                         "msq{:0>4}".format( momSq ), \
                                         "arr" )[ :, 0, 0, ... ].real

    else:
        
        twop_boost_loc = rw.getDatasets( twopDir, configList_loc, \
                                         twop_template, \
                                         "twop" )[ :, 0, 0, ..., 0, 0 ]

    # twop_boost_loc[ c, t, mom ] -> twop_boost_loc [mom, c, t ]
    
    twop_boost_loc = np.moveaxis( twop_boost_loc, -1, 0 )

    twop_boost_loc = np.asarray( twop_boost_loc, order='c' )

    mpi_fncs.mpiPrint( "Read boosted two-point functions from HDF5 files " \
                       + "in {:.3} seconds".format( time.time() - t0 ), \
                       rank )

else:

    twop_boost_loc = np.array( [] )

T = twop_loc.shape[ -1 ]

# Time dimension length after fold

T_fold = T // 2 + 1

# Gather two-point functions

twop = np.zeros( ( configNum, T ) )

comm.Allgather( twop_loc, twop )

##########################################
# Jackknife and fold two-point functions #
##########################################

# Time dimension length after fold

T_fold = T // 2 + 1

if binNum_loc:

    twop_jk_loc = fncs.jackknifeBinSubset( twop, binSize, bin_glob[ rank ] )

    # twop_fold[ b, t ]

    twop_fold = fncs.fold( twop_jk_loc )

    # mEff[ b, t ]

    mEff_loc = pq.mEffFromSymTwop( twop_fold )

else:

    twop_jk_loc = np.array( [] )

    mEff_loc = np.array( [] )

recvCount, recvOffset = mpi_fncs.recvCountOffset( procNum, binNum )

##################
# Effective mass #
##################

if rank == 0:

    twop_jk = np.zeros( ( binNum_glob, T ) )
    mEff = np.zeros( ( binNum_glob, T_fold ) )

else:

    twop_jk = None
    mEff = None

comm.Gatherv( twop_jk_loc, [ twop_jk, recvCount * T, \
                             recvOffset * T, MPI.DOUBLE ], root=0 )
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
    
        fitResults = fit.mEffTwopFit( mEff, twop_jk, \
                                      rangeEnd, tsf )
    
    except fit.lqcdjk_BadFitError as error:

        mpi_fncs.mpiPrintErr( error, rank )

        exit()

    fitParams = fitResults[ 0 ]
    chiSq = fitResults[ 1 ]
    mEff_fit = fitResults[ 2 ]
    rangeStart = fitResults[ 3 ]
    mEff_rangeStart = fitResults[ 4 ]

    if tsf:

        c0 = fitParams[ :, 0 ]
        c1 = fitParams[ :, 1 ]
        E0 = fitParams[ :, 2 ]
        E1 = fitParams[ :, 3 ]

        # Calculate fitted curve

        curve = np.zeros( ( binNum_glob, 50 ) )

        t_s = np.concatenate( ( np.linspace( rangeStart, \
                                             rangeEnd, 25 ), \
                                np.linspace( T - rangeEnd, \
                                             T- rangeStart, 25 ) ) )
                    
        for b in range( binNum_glob ):

            for t in range( t_s.shape[ -1 ] ):
                
                curve[ b, t ] = fit.twoStateTwop( t_s[ t ], T, c0[ b ], c1[ b ], \
                                                  E0[ b ], E1[ b ] )
                        
            # End loop over tsink
        # End loop over bins

        curve_avg = np.average( curve, axis=0 )
        curve_err = fncs.calcError( curve, binNum_glob )
                
        chiSq_avg = np.average( chiSq, axis=0 )
        chiSq_err = fncs.calcError( chiSq, binNum_glob )
    
        c0_avg = np.average( c0 )
        c0_err = fncs.calcError( c0, binNum_glob )
                
        c1_avg = np.average( c1 )
        c1_err = fncs.calcError( c1, binNum_glob )
            
        E1_avg = np.average( E1 )
        E1_err = fncs.calcError( E1, binNum_glob )
        
        # Write output files

        tsf_range_str = "2s" + str( rangeStart ) \
                        + ".2e" + str( rangeEnd )

        curveOutputFilename \
            = output_template.replace( "*", \
                                       "twop_twoStateFit_curve_" \
                                       + tsf_range_str )
        rw.writeAvgDataFile_wX( curveOutputFilename, \
                                t_s, curve_avg, curve_err )

        chiSqOutputFilename \
            = output_template.replace( "*", \
                                       "twop_twoStateFit_chiSq_" \
                                       + tsf_range_str )
        rw.writeFitDataFile( chiSqOutputFilename, \
                             chiSq_avg, chiSq_err, rangeStart, rangeEnd )

    else: # One-state fit

        G = fitParams[ :, 0 ]
        E = fitParams[ :, 1 ]

        # Calculate fitted curve

        curve = np.zeros( ( binNum_glob, 50 ) )

        t_s = np.concatenate( ( np.linspace( rangeStart, \
                                             rangeEnd, 25 ), \
                                np.linspace( T - rangeEnd, \
                                             T- rangeStart, 25 ) ) )

        for b in range( binNum_glob ):

            for t in range( t_s.shape[ -1 ] ):
                
                curve[ b, t ] = fit.oneStateTwop( t_s[ t ], T, \
                                                  G[ b ], E[ b ] )
                        
            # End loop over tsink
        # End loop over bins

        curve_avg = np.average( curve, axis=0 )
        curve_err = fncs.calcError( curve, binNum_glob )
            
        chiSq_avg = np.average( chiSq, axis=0 )
        chiSq_err = fncs.calcError( chiSq, binNum_glob )
            
        G_avg = np.average( G )
        G_err = fncs.calcError( G, binNum_glob )
                
        # Write output files

        twopFit_str = "2s" + str( rangeStart ) \
                      + ".2e" + str( rangeEnd )

        curveOutputFilename \
            = output_template.replace( "*", \
                                       "twop_fit_curve_" \
                                       + twopFit_str )
        rw.writeAvgDataFile_wX( curveOutputFilename, t_s, curve_avg, curve_err )
        
        chiSqOutputFilename \
            = output_template.replace( "*", \
                                       "twop_fit_chiSq_" \
                                       + twopFit_str )
        rw.writeFitDataFile( chiSqOutputFilename, chiSq_avg, chiSq_err, rangeStart, rangeEnd )

    # End if not two-state fit

    mEff_fit_avg = np.average( mEff_fit, axis=0 )
    mEff_fit_err = fncs.calcError( mEff_fit, binNum_glob )

    mEff_range_str = "2s" + str( mEff_rangeStart ) \
                     + ".2e" + str( rangeEnd )

    mEff_outputFilename = output_template.replace( "*", "mEff_fit_" + mEff_range_str )
    rw.writeFitDataFile( mEff_outputFilename, mEff_fit_avg, \
                         mEff_fit_err, mEff_rangeStart, rangeEnd )

# End if first process

if momSq > 0:

    twop_boost = np.zeros( ( momBoostNum, configNum, T ) )

    comm.Allgather( twop_boost_loc, twop_boost )

else:
    
    twop_boost = np.array( [] )

if rank == 0:
        
    twop_boost_jk = np.zeros( ( momBoostNum, binNum_glob, T ) )
    threep_jk = np.zeros( ( momBoostNum, flavNum, \
                            tsinkNum, binNum_glob, T ) )
    avgX = np.zeros( ( momBoostNum, flavNum, \
                       tsinkNum, binNum_glob, T ) )

else:

    twop_boost_jk = np.array( [ None for imom in range( momBoostNum ) ] )
    threep_jk = np.array( [ [ [ None for ts in tsink ] \
                              for f in flav_str ] \
                            for imom in range( momBoostNum ) ] )
    avgX = np.array( [ [ [ None for ts in tsink ] \
                         for f in flav_str ] \
                       for imom in range( momBoostNum ) ] )

# Loop over momenta
for imom in range( momBoostNum ):

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
                                   threep_tokens, ts, momList[ imom ], \
                                   particle, dataFormat )

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

        threep_loc = np.zeros( ( flavNum, binNum_loc, T ) )
            
        # Subtract average over directions from gtDt

        threep_loc = threep_gtDt - \
                     0.25 * ( threep_gtDt \
                              + threep_gxDx \
                              + threep_gyDy \
                              + threep_gzDz )

        threep_loc = np.asarray( threep_loc, order='c' )
    
        threep = np.zeros( ( flavNum, configNum, T ) )

        comm.Allgather( threep_loc, threep[ 0 ] )

        if particle == "kaon":

            threep_loc = threep_s_gtDt - \
                              0.25 * ( threep_s_gtDt \
                                       + threep_s_gxDx \
                                       + threep_s_gyDy \
                                       + threep_s_gzDz )

            threep_loc = np.asarray( threep_loc, order='c' )
    
            comm.Allgather( threep_loc, threep[ 1 ] )

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
            
            if rank == 0:

                """
                threep_avg = np.average( threep_jk[ iflav, its ], axis = 0 )
                threep_err = fncs.calcError( threep_jk[ iflav, its ], \
                                             binNum_glob )
        
                avgOutputFilename = output_template.replace( "*", \
                                                             "threep_" \
                                                             + flav_str[ iflav ] \
                                                             + "_tsink" \
                                                             + str( ts ) )

                rw.writeAvgDataFile( avgOutputFilename, threep_avg, threep_err )
                """
                #################
                # Calculate <x> #
                #################

                # avgX[ b, t ]

                avgX[imom,iflav,its]=pq.calcAvgX_momBoost(threep_jk[imom, \
                                                                    iflav, \
                                                                    its], \
                                                          twop_boost_jk[imom,\
                                                                        :, \
                                                                        ts], \
                                                          mEff_fit_avg, \
                                                          momSq, L )
                """
                if tsf:

                    c0_cp = np.repeat( c0, T ).reshape( binNum_glob, T )
                    c1_cp = np.repeat( c1, T ).reshape( binNum_glob, T )
                    E0_cp = np.repeat( E0, T ).reshape( binNum_glob, T )
                    E1_cp = np.repeat( E1, T ).reshape( binNum_glob, T )

                    avgX = preFactor * threep_jk[ iflav, its ] \
                           / fit.twoStateTwop( ts, T, c0_cp, c1_cp, E0_cp, E1_cp )
        
                else:

                    if its == 0:

                        G = np.repeat( G, T ).reshape( binNum_glob, T )
                        E = np.repeat( E, T ).reshape( binNum_glob, T )

                    avgX = preFactor * threep_jk[ iflav, its ] \
                           / fit.oneStateTwop( ts, T, G, E )
                    
                    #avgX = Z * pq.calcAvgX( threep_jk[ -1 ], \
                    #                        twop_jk[ :, ts ], mEff_fit )
                """
            # End loop over flavor
        # End if first process
    # End loop over tsink
# End loop over momenta

if rank == 0:

    # Average over momenta

    avgX = np.average( avgX, axis=0 )

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

##################
# Two-state Fit  #
##################

if tsf and rank == 0:

    mpi_fncs.mpiPrint( "Will perform the two-state fit", rank )

    #twop_rangeEnd = 20

    for iflav in range( flavNum ):

        for threep_neglect in 2,3:

            ti_to_fit = fncs.initEmptyList( tsinkNum, 1 )

            for ts in range( tsinkNum ):

                ti_to_fit[ ts ] = np.concatenate( ( range( threep_neglect, \
                                                           tsink[ ts ] \
                                                           - threep_neglect \
                                                           + 1 ), \
                                                    range( tsink[ ts ] \
                                                           + threep_neglect \
                                                           + 5, \
                                                           T - \
                                                           threep_neglect \
                                                           - 5 + 1 ) ) )
                """
                ti_to_fit[ ts ] = range( tsink[ ts ] + threep_neglect + 5, \
                T - threep_neglect - 5 + 1 )
                ti_to_fit[ ts ] = range( threep_neglect, \
                tsink[ ts ] - threep_neglect + 1 )
                """

            fitParams, chiSq = fit.twoStateFit_threep( threep_jk[iflav], \
                                                       ti_to_fit, \
                                                       tsink, E0, E1, T )

            a00 = fitParams[ :, 0 ]
            a01 = fitParams[ :, 1 ]
            a11 = fitParams[ :, 2 ]
          
            fitParams = np.stack( ( a00, a01, a11, c0, c1, E0, E1 ), axis=1 )

            # Calculate curve with constant tsink

            threep_curve = np.zeros( ( binNum_glob, tsinkNum, 100 ) )
            curve = np.zeros( ( binNum_glob, tsinkNum, 100 ) )

            avgX = np.zeros( binNum_glob )
        
            ti_curve = np.zeros( ( tsinkNum, 100 ) )
            
            for b in range( binNum_glob ):

                for ts in range( tsinkNum ):

                    ti_curve[ts]=np.concatenate((np.linspace(ti_to_fit[ts][0],\
                                                             ti_to_fit[ts][tsink[ts] - 2*threep_neglect], \
                                                             num = 50 ), \
                                                 np.linspace( ti_to_fit[ ts ][ tsink[ts] - 2*threep_neglect+1  ], \
                                                              ti_to_fit[ ts ][ -1 ], \
                                                              num = 50 ) ) )
                    """
                    ti_curve[ ts ] = np.linspace( ti_to_fit[ ts ][ 0 ], \
                    ti_to_fit[ ts ][ -1 ], \
                    num = 100 )
                    """
                    for t in range( len( ti_curve[ ts ] ) ):
                        
                        threep_curve[b,ts,t] = fit.twoStateThreep( ti_curve[ ts, t ], \
                                                                   tsink[ ts ], \
                                                                   T, \
                                                                   a00[ b ], \
                                                                   a01[ b ], \
                                                                   a11[ b ], \
                                                                   E0[ b ], \
                                                                   E1[ b ] )

                        curve[ b, ts, t ] = -4.0 / 3.0 / mEff_fit[ b ] * Z \
                                            * fit.twoStateThreep( ti_curve[ ts, t ], \
                                                                  tsink[ ts ], \
                                                                  T, \
                                                                  a00[ b ], \
                                                                  a01[ b ], \
                                                                  a11[ b ], \
                                                                  E0[ b ], \
                                                                  E1[ b ] ) \
                                            / fit.twoStateTwop( tsink[ ts ], \
                                                                T, \
                                                                c0[ b ], \
                                                                c1[ b ], \
                                                                E0[ b ], \
                                                                E1[ b ] )

                    # End loop over insertion time
                # End loop over tsink
            
                avgX[ b ] = -4.0 / 3.0 / mEff_fit[ b ] * Z \
                            * a00[ b ] / c0[ b ]

                # Write curve with constant insertion time = tsink / 2
        
                #for b in range( binNum_glob ):
                    
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
                    
            threep_curve_avg = np.average( threep_curve, axis=0 )
            threep_curve_err = fncs.calcError( threep_curve, binNum_glob )
            
            curve_avg = np.average( curve, axis=0 )
            curve_err = fncs.calcError( curve, binNum_glob )
                
            fitParams_avg = np.average( fitParams, axis=0 )
            fitParams_err = fncs.calcError( fitParams, binNum_glob )

            chiSq_avg = np.average( chiSq, axis=0 )
            chiSq_err = fncs.calcError( chiSq, binNum_glob )
                
            avgX_avg = np.average( avgX )
            avgX_err = fncs.calcError( avgX, binNum_glob )
    
            # Write output file

            tsf_threep_range_str = tsf_range_str + ".3n" + str( threep_neglect )

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
                                        ti_curve[ ts ], \
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
                                        ti_curve[ ts ], curve_avg[ ts ], \
                                        curve_err[ ts ] )
            
            # End loop over tsink
        # End loop over number of neglected three-point functions
    # End loop over flavors
# End if two-state fit

