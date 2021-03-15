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

srcNum = 56

particle_list = [ "pion", "kaon", "nucleon" ]

format_list = [ "gpu", "cpu" ]

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Test fit ranges of two-point functions" )

parser.add_argument( "twop_dir", action='store', type=str )

parser.add_argument( "twop_template", action='store', type=str )

parser.add_argument( "fit_range_end", action='store', type=int )

parser.add_argument( "particle", action='store', \
                     help="Particle to calculate gA for. " \
                     + "Should be 'pion' or 'kaon'.", type=str )

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

mpi_confs_info = mpi_fncs.lqcdjk_mpi_init()

comm = mpi_confs_info[ 'comm' ]
rank = mpi_confs_info[ 'rank' ]

# Input directories and filename templates

twopDir = args.twop_dir

twop_template = args.twop_template

# Last point to fit

rangeEnd = args.fit_range_end

# Info on what to analyze

particle = args.particle

# Other info

binSize = args.binSize

momSq = args.mom_squared

output_template = args.output_template

tsf = args.two_state_fit

dataFormat = args.data_format

# Get configurations from given list or from given 
# threep directory if list not given

mpi_confs_info[ 'configList' ] = fncs.getConfigList( args.config_list, 
                                                     twopDir )
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

assert particle in particle_list, \
    "Error: Particle not supported. " \
    + "Supported particles: " + str( particle_list )

assert dataFormat in format_list, \
    "Error: Data format not supported. " \
    + "Supported particles: " + str( format_list )

# Read momentum list

momList = rw.readMomentaList( twopDir, twop_template, \
                              configList_loc[ 0 ], particle, \
                              srcNum, momSq, dataFormat, mpi_confs_info )

momBoostNum = len( momList )

############################
# Read Two-point Functions #
############################

# Zero momentum two-point functions
# twop[ p, c, t ]

if momSq > 0:

    twop_p = rw.readTwopFile_zeroQ( twopDir, configList_loc, configNum, \
                                  twop_template, srcNum, momSq, \
                                  dataFormat, mpi_confs_info )

else:

    twop_p = np.array( [ rw.readTwopFile_zeroQ( twopDir, configList_loc, 
                                                configNum, twop_template, 
                                                srcNum, 0, dataFormat,
                                                mpi_confs_info ) ] )

T = twop_p.shape[ -1 ]

# Time dimension length after fold

T_fold = T // 2 + 1

##########################################
# Jackknife and fold two-point functions #
##########################################

twop_jk_p = np.zeros( ( momBoostNum, binNum, T_fold ) )
mEff_p = np.zeros( ( momBoostNum, binNum, T_fold ) )

for ip in range( momBoostNum ):

    if binNum_loc:

        twop_jk_p_loc = fncs.jackknifeBinSubset( twop_p[ ip ], binSize, \
                                                 binList_loc )

        # twop_fold[ b, t ]

        twop_fold_loc = fncs.fold( twop_jk_p_loc )

        mEff_p_loc = pq.mEffFromSymTwop( twop_fold_loc )

    else:

        twop_fold_loc = np.array( [] )

        mEff_p_loc = np.array( [] )


    comm.Allgatherv( twop_fold_loc, [ twop_jk_p[ ip ], recvCount * T_fold, \
                                      recvOffset * T_fold, MPI.DOUBLE ] )
    comm.Allgatherv( mEff_p_loc, [ mEff_p[ ip ], recvCount * T_fold, \
                                 recvOffset * T_fold, MPI.DOUBLE ] )

# End loop over p

# Average over momenta

twop_jk = np.average( twop_jk_p, axis=0 )
mEff = np.average( mEff_p, axis=0 )


###############################
# Fit the two-point functions #
###############################


# Fit the effective mass and two-point functions 

if np.any( np.isnan( mEff ) ):

    rangeEnd = min( np.where( np.isnan( mEff ) )[-1] ) - 1

# fitResults( fitParams, chiSq, t_low )

mEff_fitResults, twop_fitResults, mEff_tsf_fitResults \
    = fit.testEffEnergyTwopFit( mEff, twop_jk, rangeEnd,
                                momSq, L, particle,
                                tsf, mpi_confs_info )

if rank == 0:

    # Average twop over bins

    twop_avg = np.average( twop_jk, axis=-2 )
    twop_err = fncs.calcError( twop_jk, binNum, axis=-2 )

    twop_filename = rw.makeFilename( output_template, 
                                     "twop" )
    rw.writeAvgDataFile( twop_filename, twop_avg, twop_err )

    mEff_avg = np.average( mEff, axis=-2 )
    mEff_err = fncs.calcError( mEff, binNum, axis=-2 )

    mEff_filename = rw.makeFilename( output_template, 
                                     "mEff" )
    rw.writeAvgDataFile( mEff_filename, mEff_avg, mEff_err )

    # mEff_avg[ t ]

    for fitR in mEff_fitResults:

        fitParams = fitR[ 0 ]
        chiSq = fitR[ 1 ]
        rangeStart = fitR[ 2 ]

        chiSq_avg = np.average( chiSq, axis=0 )
        chiSq_err = fncs.calcError( chiSq, binNum )
        
        mEff_fit_avg = np.average( fitParams, axis=0 )
        mEff_fit_err = fncs.calcError( fitParams, binNum )
            
        mEff_outputFilename = rw.makeFilename( output_template, 
                                               "mEff_fit_t_low_{:0>2}", 
                                               rangeStart )
        rw.writeFitDataFile( mEff_outputFilename, mEff_fit_avg, \
                             mEff_fit_err, rangeStart, rangeEnd )
    
        mEff_chiSqOutputFilename \
            = rw.makeFilename( output_template, 
                               "mEff_chiSq_t_low_{:0>2}", 
                               rangeStart )
        rw.writeFitDataFile( mEff_chiSqOutputFilename, chiSq_avg, \
                             chiSq_err, rangeStart, rangeEnd )

    # End loop over mEff fit results

    for fitR in twop_fitResults:

        fitParams = fitR[ 0 ]
        chiSq = fitR[ 1 ]
        rangeStart = fitR[ 2 ]

        if tsf:

            chiSqOutputFilename \
                = rw.makeFilename( output_template,
                                   "twop_2sf_chiSq_t_low_{:0>2}", 
                                   rangeStart )

            fitParams_avg = np.concatenate( ( [ 0, 0, 0 ], \
                                              np.average( fitParams, \
                                                          axis=0 ) ) )
            fitParams_err = np.concatenate( ( [ 0, 0, 0 ], \
                                              fncs.calcError( fitParams, \
                                                              binNum ) ) )
            
            fitParamsOutputFilename \
                = rw.makeFilename( output_template,
                                   "twop_2sf_params_t_low_{:0>2}",
                                   rangeStart )
            rw.writeTSFParamsFile( fitParamsOutputFilename, \
                               fitParams_avg, fitParams_err )
                
            # Calculate fitted curve

            c0 = fitParams[ :, 0 ]
            c1 = fitParams[ :, 1 ]
            E0 = fitParams[ :, 2 ]
            E1 = fitParams[ :, 3 ]
            
            curve, ts = fit.calcmEffTwoStateCurve( np.ones( binNum ), \
                                                   c1/c0, E0, E1, T, \
                                                   rangeStart, \
                                                   rangeEnd )
            
            #curve, ts = fit.calcTwopTwoStateCurve( c0, c1, \
            #                                       E0, E1, T, \
            #                                       rangeStart, \
            #                                       rangeEnd )
            
            curve_avg = np.average( curve, axis=0 )
            curve_err = fncs.calcError( curve, binNum )
        
            curveOutputFilename \
                = rw.makeFilename( output_template,
                                   "twop_2sf_curve_t_low_{:0>2}",
                                   rangeStart )
            rw.writeAvgDataFile_wX( curveOutputFilename, ts, \
                                    curve_avg, curve_err )
        
        else: # One-state fit
            
            chiSqOutputFilename \
                = rw.makeFilename( output_template, 
                                   "twop_oneStateFit_chiSq_t_low_{:0>2}",
                                   rangeStart )
            
        # End one-state fit

        chiSq_avg = np.average( chiSq, axis=0 )
        chiSq_err = fncs.calcError( chiSq, binNum )
            
        # Write output files

        rw.writeFitDataFile( chiSqOutputFilename, chiSq_avg, \
                             chiSq_err, rangeStart, rangeEnd )

    # End loop over fit results

    for fitR in mEff_tsf_fitResults:

        fitParams = fitR[ 0 ]
        chiSq = fitR[ 1 ]
        rangeStart = fitR[ 2 ]
    
        c = fitParams[ :, 0 ]
        E0 = fitParams[ :, 1 ]
        E1 = fitParams[ :, 2 ]
        
        fitParams_avg = np.concatenate( ( [ 0, 0, 0, 1 ], \
                                          np.average( fitParams, \
                                                      axis=0 ) ) )
        fitParams_err = np.concatenate( ( [ 0, 0, 0, 1 ], \
                                          fncs.calcError( fitParams, \
                                                          binNum ) ) )
    
        chiSq_avg = np.average( chiSq, axis=0 )
        chiSq_err = fncs.calcError( chiSq, binNum )
        
        # Calculate fitted curve

        curve, ts = fit.calcmEffTwoStateCurve( np.ones( binNum ), \
                                               c, E0, E1, T, \
                                               rangeStart, \
                                               rangeEnd )
        
        curve_avg = np.average( curve, axis=0 )
        curve_err = fncs.calcError( curve, binNum )
    
        # Write output files

        fitParamsOutputFilename \
            = rw.makeFilename( output_template,
                               "mEff_2sf_params_t_low_{:0>2}", 
                               rangeStart )
        rw.writeTSFParamsFile( fitParamsOutputFilename, \
                               fitParams_avg, fitParams_err )
        
        chiSqOutputFilename \
            = rw.makeFilename( output_template,
                               "mEff_2sf_chiSq_t_low_{:0>2}", 
                               rangeStart )
        rw.writeFitDataFile( chiSqOutputFilename, chiSq_avg, \
                             chiSq_err, rangeStart, rangeEnd )

        curveOutputFilename \
            = rw.makeFilename( output_template,
                               "mEff_2sf_curve_t_low_{:0>2}", 
                               rangeStart )
        rw.writeAvgDataFile_wX( curveOutputFilename, ts, \
                                curve_avg, curve_err )
        
    # End loop over fit results
# End first process
