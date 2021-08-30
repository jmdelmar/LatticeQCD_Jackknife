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

a = 0.093
L = 32.0

particle_list = fncs.particleList()

form_factor_list = fncs.formFactorList()

data_format_list = fncs.dataFormatList()

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Calculate effective energy " \
                              + "from two-state fit and dispersion " \
                              + "relation." )

parser.add_argument( "twop_dir", action='store',
                     help="Two-point function filename directory",
                     type=str )

parser.add_argument( "twop_template", action='store',
                     help="Two-point function filename template",
                     type=str )

parser.add_argument( "particle", action='store', \
                     help="Particle to calculate for. " \
                     + "Should be 'pion', 'kaon', or 'nucleon'.", \
                     type=str )

parser.add_argument( "binSize", action='store', type=int )

parser.add_argument( "qSq_last_index", action='store', type=int )

parser.add_argument( "-o", "--output_template", action='store', \
                     type=str, default="./*.dat" )

parser.add_argument( "-sn", "--source_number", action='store', type=int, \
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

mpi_confs_info = mpi_fncs.lqcdjk_mpi_init()

comm = mpi_confs_info[ 'comm' ]
procNum = mpi_confs_info[ 'procNum' ]
rank = mpi_confs_info[ 'rank' ]

# Input directories and filename templates

twopDir = args.twop_dir

twop_template = args.twop_template

# Info on what to analyze

particle = args.particle

# Other info

binSize = args.binSize

iqSq_last = args.qSq_last_index

output_template = args.output_template

srcNum = args.source_number

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
configList = mpi_confs_info[ 'configList' ]
configList_loc = mpi_confs_info[ 'configList_loc' ]
recvCount = mpi_confs_info[ 'recvCount' ]
recvOffset = mpi_confs_info[ 'recvOffset' ]

# Check inputs

assert particle in particle_list, \
    "Error: Particle not supported. " \
    + "Supported particles: " + str( particle_list )

assert dataFormat in data_format_list, \
    "Error: Data format not supported. " \
    + "Supported formats: " + str( data_format_list )

assert configNum % binSize == 0, "Number of configurations " \
    + str( configNum ) + " not evenly divided by bin size " \
    + str( binSize ) + "."

q, qNum, qSq, qSqNum, \
    qSq_start, qSq_end, qSq_where \
    = rw.readMomentumTransferList( twopDir,
                                   twop_template,
                                   [configList[ 0 ]], particle, 
                                   srcNum, dataFormat,
                                   args.momentum_transfer_list, 
                                   mpi_confs_info )
   
# Cut q off at iqSq_last

mpi_fncs.mpiPrint(qSq,mpi_confs_info)

qSq = qSq[ :iqSq_last ]
qSqNum = len( qSq )
qSq_start = qSq_start[ :iqSq_last ]
qSq_end = qSq_end[ :iqSq_last ]
qSq_where = qSq_where[ :qSq_end[ -1 ] + 1 ]

mpi_fncs.mpiPrint(qSq,mpi_confs_info)

q = q[ :qSq_end[ -1 ] + 1 ]
qNum = len( q )


############################
# Read Two-point Functions #
############################

# twop_q[ c, Q, t ]

twop_q = rw.readTwopFile( twopDir,
                          twop_template,
                          configList_loc, 
                          configNum, q, qSq,
                          qSq_start, qSq_end, 
                          particle, srcNum,
                          dataFormat,
                          mpi_confs_info )

# Time dimension length

T = twop_q.shape[ -1 ]

# Time dimension length after fold

T_fold = T // 2 + 1

rangeEnd = T // 2 - 1


##########################################
# Jackknife and fold two-point functions #
##########################################


if binNum_loc:

    # twop_jk_loc[ b_loc, q, t ]

    twop_jk_loc = fncs.jackknifeBinSubset( twop_q,
                                           binSize,
                                           binList_loc )

    # twop_jk_loc[ b_loc, q, t ]

    twop_fold_loc = fncs.fold( twop_jk_loc )


    #############################################
    # Average two-point functions over momentum #
    #############################################

    
    # twop_loc[ b_loc, qSq, t ]

    twop_loc = fncs.averageOverQsq( twop_fold_loc[ :, :, : ],
                                    qSq_start, qSq_end )
        

    ####################
    # Effective Energy #
    ####################


    # effEnergy_loc[ b_loc, smr, qSq, t ]

    effEnergy_loc = pq.mEffFromSymTwop( twop_loc )

else:

    twop_fold_loc = np.array( [] )
    twop_loc = np.array( [] )
    effEnergy_loc = np.array( [] )


####################################################
# Gather two-point functions  and effective energy #
####################################################


# twop[ b, qSq, t ]
# effEnergy[ b, qSq, t ]

twop = np.zeros( ( binNum, qSqNum, T_fold ) )
effEnergy = np.zeros( twop.shape )

comm.Allgatherv( twop_loc,
                 [ twop, 
                   recvCount \
                   * np.prod( twop.shape[ 1: ] ),
                   recvOffset \
                   * np.prod( twop.shape[ 1: ] ), 
                   MPI.DOUBLE ] )

comm.Allgatherv( effEnergy_loc,
                 [ effEnergy, 
                   recvCount \
                   * np.prod( effEnergy.shape[ 1: ] ),
                   recvOffset \
                   * np.prod( effEnergy.shape[ 1: ] ), 
                   MPI.DOUBLE ] )


###########################
# Fit two-point functions #
###########################


#c0_tsf = np.zeros( ( binNum, qSqNum ) )
#c1_tsf = np.zeros( ( binNum, qSqNum ) )
#E1_tsf = np.zeros( ( binNum, qSqNum ) )
E_fit = np.zeros( ( binNum, qSqNum ) )
E_disp = np.zeros( ( binNum, qSqNum ) )

# Fit q^2 two-point functions to get mEff and fit starts

fitResults = fit.effEnergyTwopFit( effEnergy[ :, 0, : ],
                                   twop[ :, 0, : ], rangeEnd,
                                   0, L, True, mpi_confs_info,
                                   fitType="twop" )

E_fit[ :, 0 ] = fitResults[ 2 ]
E_disp[ :, 0 ] = fitResults[ 2 ]
tsf_fitStart = fitResults[ 3 ]
plat_fitStart = fitResults[ 4 ]

for iq in range( 1, qSqNum ):
    
    mpi_fncs.mpiPrint(iq,mpi_confs_info)

    #E_fit[ :, iq ], dummy \
    #    = fit.fitPlateau_parallel( effEnergy[ :, iq, : ],
    #                               plat_fitStart, rangeEnd,
    #                               mpi_confs_info )

    E_disp[ :, iq ] = pq.energy( E_fit[ :, 0 ], qSq[ iq ], L )

    E_guess = np.average( E_disp[ :, iq ] )

    fitParams, dummy = fit.twoStateFit_twop( twop[ :, iq, : ],
                                             tsf_fitStart, rangeEnd,
                                             E_guess, T,
                                             mpi_confs_info )
    
    E_fit[ :, iq ] = fitParams[ :, 2 ]

    #fitParams, chiSq \
    #    = fit.twoStateFit_twop_dispersionRelation( twop[ :, ismr, iq, : ],
    #                                               tsf_fitStart,
    #                                               rangeEnd, 
    #                                               mEff_plat[ ismr ],
    #                                               qSq[ ismr, iq ], L,
    #                                               mpi_confs_info )

    #c0[ :, iq ] = fitParams[ :, 0 ]
    #c1[ :, iq ] = fitParams[ :, 1 ]
    #E1[ :, iq ] = fitParams[ :, 2 ]
    
    #mpi_fncs.mpiPrint( "Fit two-point functions at " \
    #                   + "Q^2={}".format( qSq[ ismr, iq ] ),
    #                   mpi_confs_info )
    
# End loop over q^2

##########
# Output #
##########

if rank == 0:

    # Average over bins

    E_fit_avg = np.average( E_fit, axis=0 )
    E_fit_err = fncs.calcError( E_fit, binNum, axis=0 )

    E_disp_avg = np.average( E_disp, axis=0 )
    E_disp_err = fncs.calcError( E_disp, binNum, axis=0 )

    # Write files

    Qsq_GeV = pq.convertQsqToGeV( qSq, E_fit_avg[ 0 ], a, L )

    platFilename = rw.makeFilename( output_template, 
                                    "E_fit_{}" \
                                    + "_{}configs_binSize{}",
                                    particle,
                                    configNum, binSize )

    #print(platFilename)

    rw.writeAvgDataFile_wX( platFilename, Qsq_GeV, E_fit_avg, E_fit_err )

    dispFilename = rw.makeFilename( output_template, 
                                    "E_disp_{}" \
                                    + "_{}configs_binSize{}",
                                    particle,
                                    configNum, binSize )

    #print(dispFilename)

    rw.writeAvgDataFile_wX( dispFilename, Qsq_GeV, E_disp_avg, E_disp_err )

# End first process

exit()
