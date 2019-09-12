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

parser.add_argument( "twop_dir", action='store', type=str )

parser.add_argument( "twop_template", action='store', type=str )

parser.add_argument( "fit_range_end", action='store', type=int )

parser.add_argument( "particle", action='store', \
                     help="Particle to calculate gA for. " \
                     + "Should be 'pion' or 'kaon'.", type=str )

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

twopDir = args.twop_dir

twop_template = args.twop_template

# Last point to fit

rangeEnd = args.fit_range_end

# Info on what to analyze

particle = args.particle

# Other info

binSize = args.binSize

output_template = args.output_template

tsf = args.two_state_fit

dataFormat = args.data_format

# Get configurations from given list or from given 
# threep directory if list not given

configList = np.array( fncs.getConfigList( args.config_list, twopDir ) )
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
                               "twop" )[ :, 0, 0, :, 0, 0 ]

twop_loc = np.asarray( twop_loc, order='c', dtype=float )

mpi_fncs.mpiPrint( "Read two-point functions from HDF5 files " \
                   + "in {:.3} seconds".format( time.time() - t0 ), rank )

T = twop_loc.shape[ -1 ]

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

    twop_fold_loc = fncs.fold( twop_jk_loc )

    # mEff[ b, t ]

    mEff_loc = pq.mEffFromSymTwop( twop_fold_loc )

else:

    twop_jk_loc = np.array( [] )

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

    fitResults = fit.testmEffTwopFit( mEff, twop_fold, \
                                      rangeEnd, 0, L, tsf )

    t_low_str_template = "t_low_{0:0>2}"
    
    for fitR in fitResults:

        fitParams = fitR[ 0 ]
        chiSq = fitR[ 1 ]
        mEff_fit = fitR[ 2 ]
        mEff_chiSq = fitR[ 3 ]
        rangeStart = fitR[ 4 ]
        mEff_rangeStart = fitR[ 5 ]

        twopFit_str = t_low_str_template.format( rangeStart )
        mEffFit_str = t_low_str_template.format( mEff_rangeStart )

        if tsf:

            c0 = fitParams[ :, 0 ]
            c1 = fitParams[ :, 1 ]
            E0 = fitParams[ :, 2 ]
            E1 = fitParams[ :, 3 ]

            # Calculate fitted curve

            curve, t_s = fit.calcTwopTwoStateCurve( c0, c1, E0, E1, T, \
                                                rangeStart, rangeEnd )
            
            curveOutputFilename \
                = output_template.replace( "*", \
                                           "twop_twoStateFit_curve_" \
                                           + twopFit_str )
            chiSqOutputFilename \
                = output_template.replace( "*", \
                                           "twop_twoStateFit_chiSq_" \
                                           + twopFit_str )

            fitParams_avg = np.concatenate( ( [ 0, 0, 0 ], \
                                              np.average( fitParams, \
                                                          axis=0 ) ) )
            fitParams_err = np.concatenate( ( [ 0, 0, 0 ], \
                                              fncs.calcError( fitParams, \
                                                              binNum_glob ) ) )

            fitParamsOutputFilename \
                = output_template.replace( "*", \
                                           "twop_twoStateFitParams_" \
                                           + twopFit_str )
            rw.writeTSFParamsFile( fitParamsOutputFilename, \
                                   fitParams_avg, fitParams_err )

            mEff_curve, mEff_ts = fit.calcmEffTwoStateCurve( c0, c1, \
                                                             E0, E1, T, \
                                                             rangeStart, \
                                                             rangeEnd )
                        
            mEff_curve_avg = np.average( mEff_curve, axis=0 )
            mEff_curve_err = fncs.calcError( mEff_curve, binNum_glob )
            
            mEff_curveOutputFilename \
                = output_template.replace( "*", \
                                           "mEff_twoStateFit_curve_" \
                                           + mEffFit_str )
            rw.writeAvgDataFile_wX( mEff_curveOutputFilename, t_s, \
                                    mEff_curve_avg, mEff_curve_err )
        
        else: # One-state fit

            G = fitParams[ :, 0 ]
            E = fitParams[ :, 1 ]

            # Calculate fitted curve

            curve, t_s = fit.calcTwopOneStateCurve( G, E, T, \
                                                    rangeStart, \
                                                    rangeEnd )

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
            
        mEff_chiSq_avg = np.average( mEff_chiSq, axis=0 )
        mEff_chiSq_err = fncs.calcError( mEff_chiSq, binNum_glob )
            
        # Write output files

        rw.writeAvgDataFile_wX( curveOutputFilename, t_s, \
                                curve_avg, curve_err )
        
        rw.writeFitDataFile( chiSqOutputFilename, chiSq_avg, \
                             chiSq_err, rangeStart, rangeEnd )

        mEff_fit_avg = np.average( mEff_fit, axis=0 )
        mEff_fit_err = fncs.calcError( mEff_fit, binNum_glob )

        mEff_outputFilename = output_template.replace( "*", "mEff_fit_" + mEffFit_str )
        rw.writeFitDataFile( mEff_outputFilename, mEff_fit_avg, \
                             mEff_fit_err, mEff_rangeStart, rangeEnd )

        mEff_chiSqOutputFilename \
            = output_template.replace( "*", \
                                       "mEff_chiSq_" \
                                       + mEffFit_str )
        rw.writeFitDataFile( mEff_chiSqOutputFilename, mEff_chiSq_avg, \
                             mEff_chiSq_err, rangeStart, rangeEnd )

    # End loop over fit results
# End if first process