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

#parser.add_argument( "pSq_last_index", action='store', type=int )

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

#ipSq_last = args.pSq_last_index

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

p, pNum, pSq, pSqNum, \
    pSq_start, pSq_end, pSq_where \
    = rw.readMomentumTransferList( twopDir,
                                   twop_template,
                                   [configList[ 0 ]], particle, 
                                   srcNum, dataFormat,
                                   args.momentum_transfer_list, 
                                   mpi_confs_info )
"""
# Cut p off at ipSq_last

mpi_fncs.mpiPrint(pSq,mpi_confs_info)

pSq = pSq[ :ipSq_last ]
pSqNum = len( pSq )
pSq_start = pSq_start[ :ipSq_last ]
pSq_end = pSq_end[ :ipSq_last ]
pSq_where = pSq_where[ :pSq_end[ -1 ] + 1 ]

mpi_fncs.mpiPrint(pSq,mpi_confs_info)

p = p[ :pSq_end[ -1 ] + 1 ]
pNum = len( p )
"""

############################
# Read Two-point Functions #
############################

# twop_p[ c, P, t ]

twop_p = rw.readTwopFile( twopDir,
                          twop_template,
                          configList_loc, 
                          configNum, p, pSq,
                          pSq_start, pSq_end, 
                          particle, srcNum,
                          dataFormat,
                          mpi_confs_info )

# Time dimension length

T = twop_p.shape[ -1 ]

# Time dimension length after fold

T_fold = T // 2 + 1

rangeEnd = T // 2 - 1


##########################################
# Jackknife and fold two-point functions #
##########################################


if binNum_loc:

    # twop_jk_loc[ b_loc, p, t ]

    twop_jk_loc = fncs.jackknifeBinSubset( twop_p,
                                           binSize,
                                           binList_loc )

    # twop_jk_loc[ b_loc, p, t ]

    twop_fold_loc = fncs.fold( twop_jk_loc )


    #############################################
    # Average two-point functions over momentum #
    #############################################

    
    # twop_loc[ b_loc, pSq, t ]

    twop_loc = fncs.averageOverQsq( twop_fold_loc[ :, :, : ],
                                    pSq_start, pSq_end )
        

    ####################
    # Effective Energy #
    ####################


    # effEnergy_loc[ b_loc, smr, pSq, t ]

    effEnergy_loc = pq.mEffFromSymTwop( twop_loc )

else:

    twop_fold_loc = np.array( [] )
    twop_loc = np.array( [] )
    effEnergy_loc = np.array( [] )


####################################################
# Gather two-point functions  and effective energy #
####################################################


# twop[ b, pSq, t ]
# effEnergy[ b, pSq, t ]

twop = np.zeros( ( binNum, pSqNum, T_fold ) )
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


#c0_tsf = np.zeros( ( binNum, pSqNum ) )
#c1_tsf = np.zeros( ( binNum, pSqNum ) )
#E1_tsf = np.zeros( ( binNum, pSqNum ) )
E_plat = np.zeros( ( binNum, pSqNum ) )
E_tsf = np.zeros( ( binNum, pSqNum ) )
E_disp = np.zeros( ( binNum, pSqNum ) )

# Fit p^2 two-point functions to get mEff and fit starts

fitResults = fit.effEnergyTwopFit( effEnergy[ :, 0, : ],
                                   twop[ :, 0, : ], rangeEnd,
                                   0, L, True, mpi_confs_info,
                                   fitType="twop" )

E_plat[ :, 0 ] = fitResults[ 2 ]
E_tsf[ :, 0 ] = fitResults[ 0 ][ :, 2 ]
E_disp[ :, 0 ] = fitResults[ 2 ]
tsf_fitStart = fitResults[ 3 ]
plat_fitStart = fitResults[ 4 ]

# Calculate dispertion relation

# Loop over p^2
for ip in range( 1, pSqNum ):
    
    E_disp[ :, ip ] = pq.energy( E_plat[ :, 0 ], pSq[ ip ], L )

# End loop over p^2

# Calculate plateau fit

# Loop over p^2
for ip in range( 1, pSqNum ):

    try:

        E_plat[ :, ip ], dummy \
            = fit.fitPlateau_parallel( effEnergy[ :, ip, : ],
                                       plat_fitStart, rangeEnd,
                                       mpi_confs_info )

    except ( ValueError, TypeError ) as error:

        msg_template = "Plateau fit at p^2={} threw error:"

        mpi_fncs.mpiPrint( msg_template.format( pSq[ ip ]), mpi_confs_info )

        mpi_fncs.mpiPrint( error, mpi_confs_info )

        break

# End loop over p^2

# Calculate 2-state fit

# Loop over p^2
for ip in range( 1, pSqNum ):

    E_guess = np.average( E_disp[ :, ip ] )

    try:

        fitParams, dummy = fit.twoStateFit_twop( twop[ :, ip, : ],
                                                 tsf_fitStart, rangeEnd,
                                                 E_guess, T,
                                                 mpi_confs_info )
    
    except ValueError as error:

        msg_template = "2-state fit at p^2={} threw error:"

        mpi_fncs.mpiPrint( msg_template.format( pSq[ ip ]), mpi_confs_info )

        mpi_fncs.mpiPrint( error, mpi_confs_info )

        break

    E_tsf[ :, ip ] = fitParams[ :, 2 ]
    
# End loop over p^2

##########
# Output #
##########

if rank == 0:

    # Average over bins

    E_plat_avg = np.average( E_plat, axis=0 )
    E_plat_err = fncs.calcError( E_plat, binNum, axis=0 )

    E_tsf_avg = np.average( E_tsf, axis=0 )
    E_tsf_err = fncs.calcError( E_tsf, binNum, axis=0 )

    E_disp_avg = np.average( E_disp, axis=0 )
    E_disp_err = fncs.calcError( E_disp, binNum, axis=0 )

    pSq_GeV = pq.convertQsqToGeV( pSq, E_plat_avg[ 0 ], a, L )

    # Write plateau file

    output_filename = rw.makeFilename( output_template, 
                                       "E_plat_{}" \
                                       + "_{}configs_binSize{}",
                                       particle,
                                       configNum, binSize )

    rw.writeAvgDataFile_wX( output_filename, pSq_GeV, E_plat_avg, E_plat_err )

    # Write 2-state fit file

    output_filename = rw.makeFilename( output_template, 
                                       "E_tsf_{}" \
                                       + "_{}configs_binSize{}",
                                       particle,
                                       configNum, binSize )

    rw.writeAvgDataFile_wX( output_filename, pSq_GeV, E_tsf_avg, E_tsf_err )

    # Write dispersion relation file

    output_filename = rw.makeFilename( output_template, 
                                       "E_disp_{}" \
                                       + "_{}configs_binSize{}",
                                       particle,
                                       configNum, binSize )

    rw.writeAvgDataFile_wX( output_filename, pSq_GeV, E_disp_avg, E_disp_err )

# End first process

exit()
