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

# Momenta list

momSq = args.mom_squared

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

    mpi_fncs.mpiPrint( "Error: momentum squared " + str(momSq) + " not supported.", rank )

    exit()

momBoostNum = len( momList )
                                                         
# Other info

binSize = args.binSize

output_template = args.output_template

tsf = args.two_state_fit

dataFormat = args.data_format

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

#######################
# Two-point functions #
#######################

# Get the real part of two-point functions
# twop[ c, t ]

twop_loc = []

t0 = time.time()

if dataFormat == "cpu":

    twop_loc = rw.getDatasets( twopDir, configList_loc, twop_template, \
                                            "msq0000", "arr" )[ :, 0, 0, :, 0 ].real

else:
        
    twop_loc = rw.getDatasets( twopDir, configList_loc, \
                               twop_template, "twop" )[ :, 0, 0, ..., 0, 0 ]

twop_loc = np.asarray( twop_loc, order='c' )

T = twop_loc.shape[ -1 ]

twop = np.zeros( ( configNum, T ) )

comm.Allgather( twop_loc, twop )

mpi_fncs.mpiPrint( "Read two-point functions from HDF5 files " \
                   + "in {:.3} seconds".format( time.time() - t0 ), rank )

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

#########################
# Three-point functions #
#########################

t0 = time.time()

if rank == 0:
        
    threep_jk = np.zeros( ( tsinkNum, binNum_glob, T ) )

else:

    threep_jk = [ None for ts in tsink ]

for ts, its in zip( tsink, range( tsinkNum ) ) :
    
    t0_ts =time.time()

    threep_mom = np.zeros( ( procSize, T, momBoostNum ) )
    threep_s_mom = np.zeros( ( procSize, T, momBoostNum ) )

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

        # Get the real part of gxDx, gyDy, gzDz, and gtDt
        # three-point functions at zero-momentum
        # threep[ c, t ]

        threeps = rw.readAvgXFile( threepDir, configList_loc, \
                                   threep_template, ts, particle, dataFormat)

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

        threep_mom[ ..., imom ] = threep_gtDt - \
                                  0.25 * ( threep_gtDt \
                                           + threep_gxDx \
                                           + threep_gyDy \
                                           + threep_gzDz )

        """
        if particle == "kaon":

            threep_s_mom[ imom ] = threep_s_gtDt - \
                                   0.25 * ( threep_s_gtDt \
                                            + threep_s_gxDx \
                                            + threep_s_gyDy \
                                            + threep_s_gzDz )
        """
    # End loop over momenta

    threep_loc = np.average( threep_mom, axis=-1 )

    threep_loc = np.asarray( threep_loc, order='c' )

    threep = np.zeros( ( configNum, T ) )

    comm.Allgather( threep_loc, threep )

    # Jackknife
    # threep_jk[ ts, b, t ]
    
    threep_jk_loc = fncs.jackknifeBinSubset( threep, \
                                             binSize, bin_glob[ rank ] )

    comm.Gatherv( threep_jk_loc, [ threep_jk[ its ], recvCount * T, \
                                   recvOffset * T, MPI.DOUBLE ], root=0 )

    if rank == 0:

        threep_avg = np.average( threep_jk[ its ], axis = 0 )
        threep_err = fncs.calcError( threep_jk[ its ], binNum_glob )
        
        avgOutputFilename = output_template.replace( "*", "threep_u_tsink" + str( ts ) )
        rw.writeAvgDataFile( avgOutputFilename, threep_avg, threep_err )

        #################
        # Calculate <x> #
        #################

        # avgX[ b, t ]

        preFactor = np.repeat( -4.0 / 3.0 / mEff_fit * Z, \
                               T ).reshape( binNum_glob, T )

        if tsf:

            c0_cp = np.repeat( c0, T ).reshape( binNum_glob, T )
            c1_cp = np.repeat( c1, T ).reshape( binNum_glob, T )
            E0_cp = np.repeat( E0, T ).reshape( binNum_glob, T )
            E1_cp = np.repeat( E1, T ).reshape( binNum_glob, T )

            avgX = preFactor * threep_jk[ its ] \
                   / fit.twoStateTwop( ts, T, c0_cp, c1_cp, E0_cp, E1_cp )
        
        else:

            if its == 0:

                G = np.repeat( G, T ).reshape( binNum_glob, T )
                E = np.repeat( E, T ).reshape( binNum_glob, T )

            avgX = preFactor * threep_jk[ its ] \
                   / fit.oneStateTwop( ts, T, G, E )

            #avgX = Z * pq.calcAvgX( threep_jk[ -1 ], twop_jk[ :, ts ], mEff_fit )

        # Average over bins

        avgX_avg = np.average( avgX, axis=0 )

        avgX_err = fncs.calcError( avgX, binNum_glob )

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

            avgX_fit_err = fncs.calcError( avgX_fit, binNum_glob )
                
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

            avgX_s_err = fncs.calcError( avgX_s, binNum_glob )
                
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

                avgX_s_fit_err = fncs.calcError( avgX_s_fit, binNum_glob )
                
                # Write output files
                
                avgX_s_fit_outFilename = output_template.replace( "*", \
                                                                  "avgX_s_fit_" \
                                                                  "tsink" + str( ts ) \
                                                                  + "_" + str( fitStart[ irange ] ) \
                                                                  + "_" + str( fitEnd[ irange ] ) )

                rw.writeFitDataFile( avgX_s_fit_outFilename, avgX_s_fit_avg, avgX_s_fit_err, fitStart[ irange ], fitEnd[ irange ] )

            # End loop over fit ranges
        # End if kaon
    # End if first rank
# End loop over tsink
    
##################
# Two-state Fit  #
##################

if tsf and rank == 0:

    mpi_fncs.mpiPrint( "Will perform the two-state fit", rank )

    #twop_rangeEnd = 20

    for threep_neglect in 2,3:

        ti_to_fit = fncs.initEmptyList( tsinkNum, 1 )

        for ts in range( tsinkNum ):

            ti_to_fit[ ts ] = np.concatenate( ( range( threep_neglect, \
                                                       tsink[ ts ] - threep_neglect + 1 ), \
                                                range( tsink[ ts ] + threep_neglect + 5, \
                                                       T - threep_neglect - 5 + 1 ) ) )
            """
            ti_to_fit[ ts ] = range( tsink[ ts ] + threep_neglect + 5, \
                                                       T - threep_neglect - 5 + 1 )
            ti_to_fit[ ts ] = range( threep_neglect, \
                                     tsink[ ts ] - threep_neglect + 1 )
            """

        fitParams, chiSq = fit.twoStateFit_threep( threep_jk, \
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


                ti_curve[ ts ] = np.concatenate( ( np.linspace( ti_to_fit[ ts ][ 0 ], \
                                                                ti_to_fit[ ts ][ tsink[ ts ] - 2*threep_neglect ], \
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
            
            avgX[ b ] = -4.0 / 3.0 / mEff_fit[ b ] * Z * a00[ b ] / c0[ b ]

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

        tsf_range_str += ".3n" + str( threep_neglect )

        avgXOutputFilename \
            = output_template.replace( "*", \
                                       "avgX_twoStateFit_" \
                                       + tsf_range_str + "_" \
                                       + ts_range_str )
                
        rw.writeFitDataFile( avgXOutputFilename, \
                             avgX_avg, avgX_err, 0, 0 )
            
        chiSqOutputFilename \
            = output_template.replace( "*", \
                                       "avgX_twoStateFit_threep_chiSq_" \
                                       + tsf_range_str + "_" \
                                       + ts_range_str )
        
        rw.writeFitDataFile( chiSqOutputFilename, \
                             chiSq_avg, chiSq_err, 0, 0 )
            
        avgXParamsOutputFilename \
            = output_template.replace( "*", \
                                       "avgX_twoStateFitParams_" \
                                       + tsf_range_str + "_" \
                                       + ts_range_str )

        rw.writeTSFParamsFile( avgXParamsOutputFilename, \
                               fitParams_avg, fitParams_err )
            
        for ts in range( tsinkNum ):
            
            threep_curveOutputFilename \
                = output_template.replace( "*", \
                                           "threep_twoStateFit_curve_tsink" \
                                           + str( tsink[ ts ] ) + "_" \
                                           + tsf_range_str + "_" \
                                           + ts_range_str )
            rw.writeAvgDataFile_wX( threep_curveOutputFilename, \
                                    ti_curve[ ts ], threep_curve_avg[ ts ], \
                                    threep_curve_err[ ts ] )

            curveOutputFilename \
                = output_template.replace( "*", \
                                           "avgX_twoStateFit_curve_tsink" \
                                           + str( tsink[ ts ] ) + "_" \
                                           + tsf_range_str + "_" \
                                           + ts_range_str )
            rw.writeAvgDataFile_wX( curveOutputFilename, \
                                    ti_curve[ ts ], curve_avg[ ts ], \
                                    curve_err[ ts ] )
            
        # End loop over tsink
    # End loop over number of neglected three-point functions
# End if two-state fit

# CJL To Do: Add kaon two-state fit
