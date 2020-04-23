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

assert dataFormat in format_list, \
    "Error: Data format not supported. " \
    + "Supported particles: " + str( format_list ) \

assert moment in moment_list, \
    "Error: Moment order not supported. " \
    + "Supported moments: " + str( moment_list )


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

if moment == 1:

    Z = 1.123
    moment_str = "avgX"

elif moment == 2:

    Z = 1.34
    moment_str = "avgX2"

elif moment == 3:

    Z = 1.0
    moment_str = "avgX3"

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

    avgOutputFilename = rw.makeFilename( output_template, 
                                         "mEff_avg_{}_psq{}" \
                                         + "_{}configs_binSize{}",
                                         particle, 0, 
                                         configNum, binSize )
    rw.writeAvgDataFile( avgOutputFilename, mEff_avg, mEff_err )

# Fit the effective mass and two-point functions 

try:
    
    fitResults = fit.effEnergyTwopFit( mEff, twop_fold,
                                       rangeEnd, 0, L, tsf,
                                       mpi_confs_info,
                                       plat_t_low_range=[plat_fitStart],
                                       checkFit=checkFit,
                                       fitType="twop",
                                       tsf_t_low_range=[tsf_fitStart] )
                                       #tsf_t_low_range=range(1,4) )

    fitType = "twop"

except fit.lqcdjk_BadFitError as error:

    mpi_fncs.mpiPrintErr( "ERROR (lqcdjk_fitting.mEffTwopFit):"
                          + str( error ), mpi_confs_info )
    """
    try:
        
        fitResults = fit.effEnergyTwopFit( mEff, twop_fold,
                                           rangeEnd, 0, L, tsf,
                                           mpi_confs_info,
                                           tsf_t_low_range=[tsf_fitStart],
                                           plat_t_low_range=[plat_fitStart],
                                           checkFit=checkFit,
                                           fitType="effEnergy" )
        
    except fit.lqcdjk_BadFitError as error:

        mpi_fncs.mpiPrintErr( "ERROR (lqcdjk_fitting.mEffTwopFit):"
                              + str( error ), mpi_confs_info )

    fitType = "effEnergy"
    """
fitParams = fitResults[ 0 ]
chiSq = fitResults[ 1 ]
mEff_fit = fitResults[ 2 ]
rangeStart = fitResults[ 3 ]
mEff_rangeStart = fitResults[ 4 ]
    
if rank == 0:

    mEffFit_str = "2s" + str( rangeStart ) \
                  + ".2e" + str( rangeEnd )

    if tsf:

        if fitType == "twop":

            E0_mEff = fitParams[ :, 2 ]
            E1_mEff = fitParams[ :, 3 ]

        elif fitType == "effEnergy":

            #c = fitParams[ :, 0 ]
            E0_mEff = fitParams[ :, 1 ]
            E1_mEff = fitParams[ :, 2 ]
        """
        # Calculate fitted curve

        curve, \
            t_s = fit.calcmEffTwoStateCurve( np.ones( binNum ),
                                             c, E0_mEff, E1_mEff, T,
                                             rangeStart,
                                             rangeEnd )
            
        curveOutputFilename = rw.makeFilename( output_template,
                                               "mEff_2sf_curve_{}",
                                               mEffFit_str )
        """
        E0_mEff_avg = np.average( E0_mEff, axis=0 )
        E0_mEff_err = fncs.calcError( E0_mEff, binNum )

        E1_mEff_avg = np.average( E1_mEff, axis=0 )
        E1_mEff_err = fncs.calcError( E1_mEff, binNum )

        mEff_tsf_outputFilename = rw.makeFilename( output_template,
                                                   "mEff_{}_2sf_{}_{}" \
                                                   + "_psq{}_{}configs_binSize{}",
                                                   fitType, particle, 
                                                   mEffFit_str, 0,
                                                   configNum, binSize )
        rw.writeFitDataFile( mEff_tsf_outputFilename, E0_mEff_avg,
                             E0_mEff_err, rangeStart, rangeEnd )

        chiSqOutputFilename = rw.makeFilename( output_template,
                                               "mEff_{}_2sf_chiSq_{}_{}" \
                                               + "_psq{}_{}configs_binSize{}",
                                               fitType, particle, 
                                               mEffFit_str, 0,
                                               configNum, binSize )

    else: # One-state fit
        """
        c = fitParams[ :, 0 ]
        E0_mEff = fitParams[ :, 1 ]

        # Calculate fitted curve

        curve, t_s = fit.calcTwopOneStateCurve( c, E0_mEff, T,
                                                rangeStart, rangeEnd )

        curveOutputFilename = rw.makeFilename( output_template,
                                               "twop_1sf_curve_{}",
        mEffFit_str )
        """
        chiSqOutputFilename = rw.makeFilename( output_template,
                                               "mEff_{}_1sf_chiSq_{}_{}" \
                                               + "_psq{}_{}configs_binSize{}",
                                               "twop", particle, 
                                               mEffFit_str, 0,
                                               configNum, binSize )

    # End if one-state fit

    #curve_avg = np.average( curve, axis=0 )
    #curve_err = fncs.calcError( curve, binNum )
            
    chiSq_avg = np.average( chiSq, axis=0 )
    chiSq_err = fncs.calcError( chiSq, binNum )
            
    # Write output files

    #rw.writeAvgDataFile_wX( curveOutputFilename, t_s,
    #                        curve_avg, curve_err )
        
    rw.writeFitDataFile( chiSqOutputFilename, chiSq_avg,
                         chiSq_err, rangeStart, rangeEnd )

    mEff_fit_avg = np.average( mEff_fit, axis=0 )
    mEff_fit_err = fncs.calcError( mEff_fit, binNum )

    mEff_plat_fit_str = "2s" + str( mEff_rangeStart ) \
                        + ".2e" + str( rangeEnd )

    mEff_outputFilename = rw.makeFilename( output_template,
                                           "mEff_plat_fit_{}_{}" \
                                           + "_psq{}_{}configs_binSize{}",
                                           particle, mEff_plat_fit_str, 
                                           0, configNum, binSize )
    rw.writeFitDataFile( mEff_outputFilename, mEff_fit_avg,
                         mEff_fit_err, mEff_rangeStart, rangeEnd )

else: # Not first process

    mEff_fit = None
    mEff_fit_avg = None

# End not first process

mEff_fit = comm.bcast( mEff_fit, root=0 )
mEff_fit_avg = comm.bcast( mEff_fit_avg, root=0 )


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

    # End nan in mEff_to_fit

    try:
            
        fitResults_twop \
            = fit.effEnergyTwopFit( mEff_to_fit, twop_to_fit,
                                    rangeEnd, momSq, L, tsf,
                                    mpi_confs_info,
                                    tsf_t_low_range=[tsf_fitStart],
                                    plat_t_low_range=[plat_fitStart],
                                    checkFit=checkFit,
                                    fitType="twop" )

        twop_rangeStart = fitResults_twop[ 3 ]
            
    except fit.lqcdjk_BadFitError as error:
        
        try:
            
            fitResults_tmp \
                = fit.effEnergyTwopFit( mEff_to_fit, twop_to_fit,
                                        rangeEnd, momSq, L, tsf,
                                        mpi_confs_info,
                                        tsf_t_low_range=[tsf_fitStart],
                                        plat_t_low_range=[plat_fitStart],
                                        checkFit=checkFit,
                                        fitType="effEnergy" )

        except fit.lqcdjk_BadFitError as error:
        
            mpi_fncs.mpiPrintErr( "ERROR (lqcdjk_fitting.mEffTwopFit):"
                                  + str( error ), mpi_confs_info )

        # End bad mEff fit

        twop_rangeStart = fitResults_tmp[ 3 ]
            
        twop_rangeStart = comm.bcast( twop_rangeStart, root=0 )
        
        E_guess = np.sqrt( mEff_fit_avg ** 2 
                           + ( 2.0 * np.pi / L ) ** 2 * momSq )

        fitParams_twop,chiSq=fit.twoStateFit_twop( twop_to_fit,
                                                   twop_rangeStart,
                                                   rangeEnd, 
                                                   E_guess, T,
                                                   mpi_confs_info )

    # End bad twop fit

else: # Zero momentum two-point functions

    twop_to_fit = twop_fold
    twop_rangeStart = rangeStart

    if fitType == "effEnergy":

        if tsf:

            E_guess = mEff_fit

            fitParams_twop, chiSq = fit.twoStateFit_twop( twop_to_fit,
                                                          twop_rangeStart,
                                                          rangeEnd, 
                                                          E_guess, T,
                                                          mpi_confs_info )

        else: # One-state fit

            fitParams_twop,chiSq=fit.oneStateFit_twop( twop_to_fit,
                                                       twop_rangeStart,
                                                       rangeEnd, T )

    elif fitType == "twop":

        fitResults_twop = fitResults
        twop_rangeStart = rangeStart

fitParams_twop = fitResults_twop[ 0 ]

if tsf:

    c0 = np.asarray( fitParams_twop[ :, 0 ], order = 'c', dtype=float )
    c1 = np.asarray( fitParams_twop[ :, 1 ], order = 'c', dtype=float )
    E0 = np.asarray( fitParams_twop[ :, 2 ], order = 'c', dtype=float )
    E1 = np.asarray( fitParams_twop[ :, 3 ], order = 'c', dtype=float )

else: # One-state Fit
    
    c0 = np.asarray( fitParams_twop[ :, 0 ], order = 'c', dtype=float )
    E0 = np.asarray( fitParams_twop[ :, 1 ], order = 'c', dtype=float )

# End one-state fit


######################################################
# Write the two-point functions and their fit curves #
######################################################


if rank == 0:

    if tsf:
        
        twop_curve, ts_twop = fit.calcTwopTwoStateCurve( c0, c1, 
                                                         E0, E1, T,
                                                         twop_rangeStart, 
                                                         rangeEnd )

        twopFit_str = "2s" + str( twop_rangeStart ) \
                        + ".2e" + str( rangeEnd )

        twopParams_filename = rw.makeFilename( output_template,
                                               "twop_2sf_params_per_bin_{}_{}" \
                                               + "_psq{}_{}configs_binSize{}",
                                               particle, twopFit_str, 
                                               momSq, configNum, binSize )

        rw.writeDataFile( twopParams_filename, fitParams_twop )

        fitParams_twop_avg = np.concatenate( ( [ 0, 0, 0 ], \
                                          np.average( fitParams_twop, \
                                                      axis=0 ) ) )
        fitParams_twop_err = np.concatenate( ( [ 0, 0, 0 ], \
                                          fncs.calcError( fitParams_twop, \
                                                          binNum ) ) )

        fitParams_twopOutputFilename \
            = rw.makeFilename( output_template,
                               "twop_2sf_params_{}_{}" \
                               + "_psq{}_{}configs_binSize{}",
                               particle, twopFit_str, 
                               momSq, configNum, binSize )
        rw.writeTSFParamsFile( fitParams_twopOutputFilename, \
                               fitParams_twop_avg, fitParams_twop_err )

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
    
    twop_outFilename = rw.makeFilename( output_template,
                                        "twop_{}_psq{}" \
                                        + "_{}configs_binSize{}",
                                        particle, momSq,
                                        configNum, binSize )

    rw.writeAvgDataFile( twop_outFilename, twop_avg, twop_err )

    twop_curve_outFilename = rw.makeFilename( output_template,
                                        "twop_2sf_curve_{}_psq{}" \
                                        + "_{}configs_binSize{}",
                                        particle, momSq,
                                        configNum, binSize )

    rw.writeAvgDataFile_wX( twop_curve_outFilename, ts_twop,
                            twop_curve_avg, twop_curve_err )

# End first process


##############################
# Read three-point functions #
##############################


# threep_p_jk[ p, flav, ts, b, t ]

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

            threep_p_jk_loc \
                = fncs.jackknifeBinSubset( threep_p[ iflav ],
                                           mpi_confs_info[ 'binSize' ],
                                           mpi_confs_info[ 'binList_loc' ] )

            comm.Allgatherv( threep_p_jk_loc,
                             [ threep_p_jk[ imom, iflav, its ],
                               recvCount * T,
                               recvOffset * T,
                               MPI.DOUBLE ] )

        # End loop over flavor
    # End loop over tsink
# End loop over momenta

# Average threep over momenta

threep_jk = np.average( threep_p_jk, axis=0 )


####################
# Calculate moment #
####################


#E = np.sqrt( mEff_fit ** 2 + (2.0*np.pi/L) ** 2 * momSq )
E = E0

E = np.asarray( E, order='c', dtype=float )

if rank == 0:

    # mellin[ flav, ts, b, t ]

    mellin = np.zeros( ( flavNum, tsinkNum, binNum, T ) )

    # Loop over flavor
    for iflav in range( flavNum ):
        # Loop over tsink
        for ts, its in zip( tsink, range( tsinkNum ) ) :

            mellin[ iflav, its ] \
                = Z * pq.calcMellin_twopFit( threep_jk[ iflav, its ],
                                             ts, mEff_fit, momSq, L, 
                                             c0, E, moment )

        # End loop over tsink
    # End loop over flavor

    # Average over bins
    # mellin_avg[ flav, ts, t ]

    mellin_avg = np.average( mellin, axis=-2 )
    mellin_err = fncs.calcError( mellin, binNum, axis=-2 )

    threep_avg = np.average( threep_jk, axis=-2 )
    threep_err = fncs.calcError( threep_jk, binNum, axis=-2 )

    # Loop over flavor
    for iflav in range( flavNum ):
        # Loop over tsink
        for ts, its in zip( tsink, range( tsinkNum ) ) :
            
            # Write threep output file
    
            threep_outFilename = rw.makeFilename( output_template,
                                                  "{}_threep_{}_{}_tsink{}" \
                                                  + "_psq{}_{}configs_binSize{}",
                                                  moment_str, particle, 
                                                  flav_str[ iflav ], ts,
                                                  momSq, configNum, binSize )
            rw.writeAvgDataFile( threep_outFilename,
                                 threep_avg[ iflav, its ],
                                 threep_err[ iflav, its ] )

            # Write moment output file

            mellin_outFilename \
                = rw.makeFilename( output_template,
                                   "{}_{}_{}_tsink{}" \
                                   + "_psq{}_{}configs_binSize{}",
                                   moment_str, particle, 
                                   flav_str[ iflav ], ts,
                                   momSq, configNum, binSize )
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
                                       "{}_plat_fit_{}_{}" \
                                       + "_{}_{}_tsink{}" \
                                       + "_psq{}_{}configs_binSize{}",
                                       moment_str, rangeStart_plat[ irange ],
                                       fitEnd_plat[ irange ],
                                       particle, flav_str[ iflav ], ts,
                                       momSq, configNum, binSize )
                rw.writeFitDataFile( mellin_fit_outFilename, mellin_fit_avg,
                                     mellin_fit_err, rangeStart_plat[ irange ],
                                     fitEnd_plat[ irange ] )
            
            # End loop over fit ranges
        # End loop over tsink
    # End loop over flavor
# End first process


##################
# Two-state Fit  #
##################


if tsf:

    mpi_fncs.mpiPrint( "Will perform the two-state fit", mpi_confs_info )

    comm.Bcast( E, root=0 )
    comm.Bcast( E1, root=0 )

    # Loop over flavors
    for iflav in range( flavNum ):
        # Loop over number of neglected three-point functions
        for neglect in 2,3:

            ti_to_fit = fncs.initEmptyList( tsinkNum, 1 )
            
            # Loop over tsink
            for its, ts in zip( range( tsinkNum ), tsink ):

                ti_to_fit[ its ] = np.arange( neglect, ts - neglect + 1 )
                
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

            fitParams_threep, \
                chiSq = fit.twoStateFit_threep( threep_jk[ iflav ],
                                                ti_to_fit, tsink,
                                                E, E1, T,
                                                mpi_confs_info )
                    
            if rank == 0:

                a00 = fitParams_threep[ :, 0 ]
                a01 = fitParams_threep[ :, 1 ]
                a11 = fitParams_threep[ :, 2 ]
          
                fitParams = np.stack( ( a00, a01, a11,
                                        c0, c1, E0, E1 ),
                                      axis=1 )
                
                # Calculate curve with constant tsink
                
                # curve[ b, ts, t ]
                # ti[ ts, t ]

                threep_curve, \
                    ti_threep = fit.calcThreepTwoStateCurve( a00, a01,
                                                             a11, E,
                                                             E1, T,
                                                             tsink,
                                                             ti_to_fit,
                                                             neglect )

                mellin_curve_const_ts, \
                    ti_mellin = fit.calcAvgXTwoStateCurve_const_ts( a00, 
                                                                    a01, a11,
                                                                    c0, c1, 
                                                                    E, E1,
                                                                    mEff_fit,
                                                                    momSq, 
                                                                    L, T, Z, 
                                                                    tsink,
                                                                    ti_to_fit,
                                                                    neglect )
                
                mellin_curve_const_ti, \
                    ts_mellin = fit.calcAvgXTwoStateCurve_const_ti( a00, 
                                                                    a01, a11,
                                                                    c0, c1,
                                                                    E, E1,
                                                                    mEff_fit,
                                                                    momSq,
                                                                    L, T, Z,
                                                                    tsink[0]
                                                                    -2,
                                                                    tsink[-1]
                                                                    +5 )
                    
                # mellin[ b ]

                mellin = pq.calcAvgX_twoStateFit( a00, c0, mEff_fit, 
                                                  momSq, L, Z )
                
                # Average over bins
                    
                threep_curve_avg = np.average( threep_curve, axis=0 )
                threep_curve_err = fncs.calcError( threep_curve, 
                                               binNum )
            
                mellin_curve_const_ts_avg \
                    = np.average( mellin_curve_const_ts, axis=0 )
                mellin_curve_const_ts_err \
                    = fncs.calcError( mellin_curve_const_ts, binNum )
                
                mellin_curve_const_ti_avg \
                    = np.average( mellin_curve_const_ti, axis=0 )
                mellin_curve_const_ti_err \
                    = fncs.calcError( mellin_curve_const_ti, binNum )
                
                fitParams_avg = np.average( fitParams, axis=0 )
                fitParams_err = fncs.calcError( fitParams, binNum )
                
                chiSq_avg = np.average( chiSq, axis=0 )
                chiSq_err = fncs.calcError( chiSq, binNum )
                
                mellin_avg = np.average( mellin )
                mellin_err = fncs.calcError( mellin, binNum )
    
                # Write output file
            
                tsf_threep_range_str = twopFit_str + ".3n" \
                                       + str( neglect )

                mellinOutputFilename \
                    = rw.makeFilename( output_template,
                                       "{}_2sf_{}_{}_{}_{}" \
                                       + "_psq{}_{}configs_binSize{}",
                                       moment_str, particle, flav_str[ iflav ],
                                       tsf_threep_range_str,
                                       ts_range_str,
                                       momSq, configNum, binSize )
                rw.writeFitDataFile( mellinOutputFilename,
                                     mellin_avg, mellin_err, 0, 0 )
                
                chiSqOutputFilename \
                    = rw.makeFilename( output_template,
                                       "{}_2sf_threep_chiSq_{}_{}_{}_{}" \
                                       + "_psq{}_{}configs_binSize{}",
                                       moment_str, particle, flav_str[ iflav ],
                                       tsf_threep_range_str,
                                       ts_range_str,
                                       momSq, configNum, binSize )
        
                rw.writeFitDataFile( chiSqOutputFilename,
                                     chiSq_avg, chiSq_err, 0, 0 )
                
                mellinParamsOutputFilename \
                    = rw.makeFilename( output_template,
                                       "{}_2sf_params_{}_{}_{}_{}" \
                                       + "_psq{}_{}configs_binSize{}",
                                       moment_str, particle, flav_str[ iflav ],
                                       tsf_threep_range_str,
                                       ts_range_str,
                                       momSq, configNum, binSize )

                rw.writeTSFParamsFile( mellinParamsOutputFilename,
                                       fitParams_avg, fitParams_err )
            
                curveOutputFilename \
                    = rw.makeFilename( output_template,
                                       "{}_2sf_curve_{}_{}_{}_{}" \
                                       + "_psq{}_{}configs_binSize{}",
                                       moment_str, particle, flav_str[ iflav ],
                                       tsf_threep_range_str,
                                       ts_range_str,
                                       momSq, configNum, binSize )
                rw.writeAvgDataFile_wX( curveOutputFilename,
                                        ts_mellin,
                                        mellin_curve_const_ti_avg,
                                        mellin_curve_const_ti_err )
            
                for ts in range( tsinkNum ):
            
                    threep_curveOutputFilename \
                        = rw.makeFilename( output_template,
                                           "{}_2sf_threep_curve_{}_{}_tsink{}_{}_{}" \
                                           + "_psq{}_{}configs_binSize{}",
                                           moment_str, particle, flav_str[ iflav ],
                                           tsink[ ts ],
                                           tsf_threep_range_str,
                                           ts_range_str,
                                           momSq, configNum, binSize )
                    rw.writeAvgDataFile_wX( threep_curveOutputFilename,
                                            ti_threep[ ts ],
                                            threep_curve_avg[ ts ],
                                            threep_curve_err[ ts ] )
                    
                    curveOutputFilename \
                        = rw.makeFilename( output_template,
                                           "{}_2sf_curve_{}_{}_tsink{}_{}_{}" \
                                           + "_psq{}_{}configs_binSize{}",
                                           moment_str, particle, flav_str[ iflav ],
                                           tsink[ ts ],
                                           tsf_threep_range_str,
                                           ts_range_str,
                                           momSq, configNum, binSize )
                    rw.writeAvgDataFile_wX( curveOutputFilename,
                                            ti_mellin[ ts ],
                                            mellin_curve_const_ts_avg[ ts ],
                                            mellin_curve_const_ts_err[ ts ] )
            
                # End loop over tsink
            # End first process
        # End loop over number of neglected three-point functions
    # End loop over flavors
# End two-state fit
