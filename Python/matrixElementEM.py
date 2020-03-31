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
configNum = len( mpi_confs_info[ 'configList' ] )
mpi_confs_info[ 'configNum' ] = configNum

binSize = args.binSize
mpi_confs_info[ 'binSize' ] = binSize

# Set mpi configuration information

mpi_fncs.lqcdjk_mpi_confs_info( mpi_confs_info )

binNum = mpi_confs_info[ 'binNum_glob' ]
binNum_loc = mpi_confs_info[ 'binNum_loc' ]
binList_loc = mpi_confs_info[ 'binList_loc' ]
configList_loc = mpi_confs_info[ 'configList_loc' ]
recvCount = mpi_confs_info[ 'recvCount' ]
recvOffset = mpi_confs_info[ 'recvOffset' ]

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
                              configList_loc[ 0 ], particle,
                              srcNum, momSq, dataFormat, mpi_confs_info )

momBoostNum = len( momList )

############################
# Read Two-point Functions #
############################

# Zero momentum two-point functions
# twop[ c, t ]

twop = rw.readTwopFile_zeroQ( twopDir, configList_loc, configNum,
                              twop_template, srcNum, 0, dataFormat,
                              mpi_confs_info )

# Time dimension length

T = twop.shape[ -1 ]

# Time dimension length after fold

T_fold = T // 2 + 1

##########################################
# Jackknife and fold two-point functions #
##########################################

if binNum_loc:

    twop_jk_loc = fncs.jackknifeBinSubset( twop, binSize, binList_loc )

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

comm.Allgatherv( twop_fold_loc, [ twop_fold, recvCount * T_fold,
                                  recvOffset * T_fold, MPI.DOUBLE ] )
comm.Allgatherv( mEff_loc, [ mEff, recvCount * T_fold, 
                             recvOffset * T_fold, MPI.DOUBLE ] )

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
        
    mpi_fncs.mpiPrintErr( "ERROR (lqcdjk_fitting.mEffTwopFit):" \
                          + str( error ), mpi_confs_info )
    
if rank == 0:

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

comm.Barrier()


###############################
# Boosted two-point functions #
###############################


if momSq > 0:

    twop_boost = rw.readTwopFile_zeroQ( twopDir, configList_loc, configNum,
                                        twop_template, srcNum, momSq,
                                        dataFormat, mpi_confs_info )

    twop_boost_fold_p = np.zeros( ( momBoostNum, binNum, T_fold ) )

    # Loop over momenta
    for imom in range( momBoostNum ):

        if binNum_loc:

            twop_boost_jk_loc = fncs.jackknifeBinSubset( twop_boost[ imom ],
                                                         binSize, 
                                                         binList_loc )

            twop_boost_fold_loc = fncs.fold( twop_boost_jk_loc )

        else:

            twop_boost_fold_loc = np.array( [] )

        comm.Allgatherv( twop_boost_fold_loc, [ twop_boost_fold_p[ imom ],
                                                recvCount * T_fold,
                                                recvOffset * T_fold, 
                                                MPI.DOUBLE ] )

    # End loop over momenta
# End if non-zero momentum boost
    

###########################
# Fit two-point functions #
###########################


if momSq > 0: # Boosted two-point functions

    # Average over momenta

    twop_to_fit = np.average( twop_boost_fold_p, axis=0 )
    mEff_to_fit = pq.mEffFromSymTwop( twop_to_fit )

    if np.any( np.isnan( mEff_to_fit ) ):

        rangeEnd = min( np.where( np.isnan( mEff_to_fit ) )[-1] ) - 1

        try:
            
            fitResults_tmp \
                = fit.mEffTwopFit( mEff_to_fit, twop_to_fit,
                                   rangeEnd, momSq, L, tsf,
                                   mpi_confs_info,
                                   tsf_t_low_range=[tsf_fitStart],
                                   plat_t_low_range=[plat_fitStart],
                                   checkFit=checkFit,
                                   fitType="twop" )
    
        except fit.lqcdjk_BadFitError as error:
        
            try:
            
                fitResults_tmp \
                    = fit.mEffTwopFit( mEff_to_fit, twop_to_fit,
                                       rangeEnd, momSq, L, tsf,
                                       mpi_confs_info,
                                       tsf_t_low_range=[tsf_fitStart],
                                       plat_t_low_range=[plat_fitStart],
                                       checkFit=checkFit,
                                       fitType="mEff" )

            except fit.lqcdjk_BadFitError as error:
        
                mpi_fncs.mpiPrintErr( "ERROR (lqcdjk_fitting.mEffTwopFit):"
                                      + str( error ), mpi_confs_info )

    twop_rangeStart = fitResults_tmp[ 3 ]
        
else: # Zero momentum two-point functions

    twop_to_fit = twop_fold
    
    if rank == 0:

        twop_rangeStart = rangeStart

    else:
        
        twop_rangeStart = None
    
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
    
        # Get the real part of gxDx, gyDy, gzDz, and gtDt
        # three-point functions at zero-momentum
        # threep[ iflav, c, t ]

        threeps = rw.readEMFile( threepDir, mpi_confs_info[ 'configList_loc' ],
                                 threep_tokens, srcNum, ts, momList[ imom ], \
                                 particle, dataFormat, insType, T, mpi_confs_info )

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

            comm.Gatherv( threep_p_jk_loc,
                          [ threep_p_jk[ imom, iflav, its ],
                            recvCount * T,
                            recvOffset * T,
                            MPI.DOUBLE ], root=0 )

        # End loop over flavor
    # End loop over tsink
# End loop over momenta


###################
# Calculate GE(0) #
###################


if rank == 0:
        
    ratio = np.zeros( ( flavNum, tsinkNum, binNum, T ) )

    # Average threep over momenta

    threep_jk = np.average( threep_p_jk, axis=0 )

    # Loop over flavor
    for iflav in range( flavNum ):
        # Loop over tsink
        for ts, its in zip( tsink, range( tsinkNum ) ) :
            
            ratio[ iflav, its] \
                = Zv * pq.calcMatrixElemEM_twopFit( threep_jk[iflav, its],
                                                    ts, c0, E0 ) 

        # End loop over tsink
    # End loop over flavor

    # Average over bins
    # ratio_avg[ flav, ts, t ]

    ratio_avg = np.average( ratio, axis=-2 )
    ratio_err = fncs.calcError( ratio, binNum, axis=-2 )

    threep_avg = np.average( threep_jk, axis=-2 )
    threep_err = fncs.calcError( threep_jk, binNum, axis=-2 )

    # Loop over flavor
    for iflav in range( flavNum ):
        # Loop over tsink
        for ts, its in zip( tsink, range( tsinkNum ) ) :
            
            # Write threep output files
    
            threep_outFilename = rw.makeFilename( output_template,
                                                  "threep_{}_tsink{}",
                                                  flav_str[ iflav ], ts )
            rw.writeAvgDataFile( threep_outFilename,
                                 threep_avg[ iflav, its ],
                                 threep_err[ iflav, its ] )

            # Write GE(0) output files
    
            ratio_outFilename = rw.makeFilename( output_template,
                                                 "matrixElemEM_{}_tsink{}",
                                                 flav_str[ iflav ], ts )

            rw.writeAvgDataFile( ratio_outFilename, ratio_avg[ iflav, its ],
                                 ratio_err[ iflav, its ] )

            # Fit plateau

            fitStart = [ ts // 2 - 1, ts // 2 - 2, \
                         ts // 2 - 3, ts // 2 - 4 ]
            
            fitEnd = [ ts // 2 + 1, ts // 2 + 2, \
                       ts // 2 + 3, ts // 2 + 4 ]

            # Loop over fit ranges
            for irange in range( len( fitStart ) ):

                #for x in ratio[ iflav, its ]:

                ratio_fit, chiSq = fit.fitPlateau( ratio[ iflav, its ],
                                                   ratio_err[ iflav, its ],
                                                   fitStart[ irange ],
                                                   fitEnd[ irange ] )

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
