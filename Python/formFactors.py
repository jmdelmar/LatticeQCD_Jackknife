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

# Set whether or not to use the dispersion
# relation E = sqrt( m^2 + p^2 ) when fitting
# two-point functions

dispRel = True

# Set which q^2 index to stop at for 
# three-point functions. This index will
# not be included

iqSq_last = 16

# Set option so that entire numpy arrays are printed

np.set_printoptions(threshold=sys.maxsize)

# Lattice spacing and length

a = 0.093
L = 32.0

# Lists of possible form factors and particles.
# Will check that input values are in the list.

form_factor_list = fncs.formFactorList()

particle_list = fncs.particleList()

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Calculate form factors for "
                              + "non-zero momentum transfer" )

# Set input arguments

parser.add_argument( "threep_dir", action='store', type=str,
                     help="Directory containing subdirectories of "
                     + "three-point functions. Subdirectories should "
                     + "contain three-point functions from a single "
                     + "configurations and be named after that "
                     + "configuration." )

parser.add_argument( "threep_tokens", action='store',
                     type=lambda s: [str(token) for token in s.split(',')],
                     help="Comma seperated list of filename tokens. "
                     + "CPU: part before tsink, part before momentum "
                     + "boost components. GPU: part before momentum "
                     + "boost components, part after momentum boost; "
                     + "* for configuration number." )

parser.add_argument( "twop_dir", action='store', 
                     help="Comma seperated list of directories "
                     + "containing subdirectories of two-point "
                     + "functions for each particle/flavor "
                     + "combination. Subdirectories should "
                     + "contain three-point functions from a single "
                     + "configurations and be named after that "
                     + "configuration. ",
                     type=lambda s: [str(item) for item in s.split(',')] )

parser.add_argument( "twop_template", action='store',
                     help="Comma seperated list of twop filename templates"
                     + " with same order as 'twop_dir'. CPU: two-point function "
                     + "filenames, GPU: two-point function filenames with '*' "
                     + "for configuration name.",
                     type=lambda s: [str(item) for item in s.split(',')] )

parser.add_argument( "particle", action='store',
                     help="Particle to calculate form facors for. "
                     + "Must be one of"
                     + ", ".join( particle_list ),
                     type=str )

parser.add_argument( 't_sink', action='store',
                     help="Comma seperated list of t sink's",
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "threep_final_momentum_squared",
                     action='store', type=int,
                     help="Final momentum p^2 of three-point functions." )

parser.add_argument( "form_factor", action='store', type=str,
                     help="Form factor to calculate. Must be one of "
                     + ", ".join( form_factor_list ) )

parser.add_argument( "binSize", action='store', type=int,
                     help="Size of bins for jackknife resampling." )

parser.add_argument( "-o", "--output_template", action='store',
                     type=str, default="./*.dat",
                     help="Template for output files. '*' will be "
                     + "replaced with text depending on output data." )

parser.add_argument( "-sn", "--source_number", action='store',
                     help="Comma seperated list of number of sources "
                     + "correlators were averaged over for each tsink",
                     type=lambda s: [int(item) for item in s.split(',')],
                     default=None )

parser.add_argument( "--tsf_fit_start", action='store', type=int,
                     help="If given, will perform 2-state fit on " 
                     + "two-point functions or effective energy "
                     + "starting at given t value, otherwise, will "
                     + "use lowest t value which satisfies condition.",
                     default=None )

parser.add_argument( "--plat_fit_start", action='store', type=int,
                     help="If given, will perform plateau fit on effective "
                     + "mass starting at given t value, otherwise, will "
                     + "use lowest t value which satisfies condition.",
                     default=None )

parser.add_argument( "-c", "--config_list", action='store',
                     type=str, default="",
                     help="Filename of configuration list file. "
                     + "If not given, will use list of subdirectories "
                     + "in 'threepDir'." )

parser.add_argument( "-m", "--momentum_transfer_list", action='store',
                     type=str, default="",
                     help="Filename of momentum transfer list file. "
                     + "If not given, will use list from two-point "
                     + "function data file." )

parser.add_argument( "-e", "--error_threshold", action='store',
                     type=float, default=None,
                     help="Threshold used to cut Q^2 data. "
                     + "Any data points with relative error greater "
                     + "than threshold will be neglected." )

parser.add_argument( "--pSq_last", action='store', type=int, default=None,
                     help="Last p^2 to be used in form factor ratio "
                     + "in lattice momentum units (2pi/L)^2" )

parser.add_argument( "--Qsq_last_monopole", action='store',
                     type=float, default=None,
                     help="Last Q^2 to be used in monopole fit in GeV" )

# Parse

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

threepDir = args.threep_dir

twopDir = args.twop_dir

threep_tokens = args.threep_tokens

twop_template = args.twop_template

# Info on what to analyze

particle = args.particle

tsink = args.t_sink
tsinkNum = len( tsink )
ts_range_str = "tsink" + str(tsink[0]) + "_" + str(tsink[-1])

# Other info

binSize = args.binSize

output_template = args.output_template

# Set whether or not to check that the 2-state fit on 
# the two-point functions and plateau fit on the 
# effective mass meet the condition
# | E_2sf - E_plat | = dE_plat / 2

if args.tsf_fit_start and args.plat_fit_start:

    checkFit = False

else:

    checkFit = True

srcNum = args.source_number

pSq_fin = args.threep_final_momentum_squared

formFactor = args.form_factor

errorThreshold = args.error_threshold

pSq_last = args.pSq_last

Qsq_last_monopole = args.Qsq_last_monopole

# Set the number of form factors

formFactorNum = 3 if formFactor == "A40_B40_C40" else 2

# Set string used in output file based on form factor

if formFactor == "GE_GM":
                            
    if particle == "nucleon":
        
        F_str = [ "GE", "GM" ]
                        
    else:
                
        F_str = [ "GE" ]
                
elif formFactor == "BT10":

    F_str = [ "tensorFF" ]

elif formFactor == "FS":

    F_str = [ "scalarFF" ]

elif formFactor == "A20_B20":
        
    F_str = [ "A20", "B20" ]

elif formFactor == "A30_B30":

    F_str = [ "A30", "B30" ]

elif formFactor == "A40_B40_C40":

    F_str = [ "A40", "B40", "C40" ]

# Get configurations from given list or from given 
# threep directory if list not given

mpi_confs_info[ 'configList' ] = fncs.getConfigList( args.config_list, 
                                                     threepDir )

configNum = len( mpi_confs_info[ 'configList' ] )
mpi_confs_info[ 'configNum' ] = configNum

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

# Check that particle is valid

assert particle in particle_list, \
    "Error: Particle not supported. " \
    + "Supported particles: " + str( particle_list )

# Check that configNum is evenely divided by binSize

assert configNum % binSize == 0, "Number of configurations " \
    + str( configNum ) + " not evenly divided by bin size " \
    + str( binSize ) + "."

# Check that three-point functions are in boosted frame
# if we are calculating the 2- or 3-derivative form factors

if formFactor in [ "A30_B30", "A40_B40_C40" ] and pSq_fin == 0:

    errorMessage = "Error (formFactors.py): {} form factors cannot be " \
                   + "calculated in the rest frame."

    mpi_fncs.mpiPrintError( errorMessage.format( formFactor ),
                            mpi_confs_info )

# Set projectors and flavors based on particle

if particle == "pion":

    projector = [ "" ]

    flav_str = [ "u" ]

elif particle == "kaon":

    projector = [ "" ]

    flav_str = [ "u", "s" ]
    
elif particle == "nucleon":

    projector = [ "0", "4", "5", "6" ]

    flav_str = [ "IV", "IS" ]

flavNum = len( flav_str )

# Set the number of ratios to be included in each SVD process

ratioNum = fncs.setRatioNumber( formFactor, particle, mpi_confs_info )

# Set data format for twop and threep files 
# and smear strings based on 
# particle and p^2

dataFormat_twop, dataFormat_threep \
    = fncs.setDataFormat( particle, pSq_fin )

smear_str_list, smear_str_list_boost, smearNum, smearNum_boost \
    = fncs.setSmearString( particle, pSq_fin )

if particle == "kaon" and pSq_fin > 0:

    # [ gN40a0p2_gN50a0p2, gN40a0p2 ]
    # -> [ gN40a0p2, gN40a0p2_gN50a0p2 ]

    smear_str_list = [ smear_str_list_boost[ 1 ],
                       smear_str_list_boost[ 0 ] ]
    smearNum = smearNum_boost

# Renormalization factor

if formFactor == "GE_GM":

    Z = 1.0

elif formFactor == "BT10":

    Z = 0.828

elif formFactor == "FS":

    Z = 0.475
    
elif formFactor == "A20_B20":

    Z = [ 1.123, 1.134 ]

elif formFactor == "A30_B30":

    Z = 1.34

elif formFactor == "A40_B40_C40":

    Z = 1.668

# Momentum transfer list

q, qNum, qSq, qSqNum, qSq_start, qSq_end, qSq_where \
    = [ [ [] for smr in smear_str_list ] for qq in range( 7 ) ]

for smr, ismr in fncs.zipXandIndex( smear_str_list ):

    twop_template_smr = twop_template[ ismr ].format( smr )

    q[ ismr ], qNum[ ismr ], qSq[ ismr ], qSqNum[ ismr ], \
    qSq_start[ ismr ], qSq_end[ ismr ], qSq_where[ ismr ] \
    = rw.readMomentumTransferList( twopDir[ ismr ],
                                   twop_template_smr,
                                   [configList[ 0 ]], particle, 
                                   srcNum[ -1 ], dataFormat_twop[ ismr ],
                                   args.momentum_transfer_list, 
                                   mpi_confs_info )
    
# Make sure all files have same number of Q and Q^2

for qsqN in qNum:

    if qsqN != qNum[ 0 ]:

        mpi_fncs.mpiPrintError( "Error (formFactors.py): " \
                           + "two-point function files " \
                           + "do not have the same number " \
                           + "of q. Number of q's are: " \
                           + ",".join( qNum ) + ".",
                           mpi_confs_info )

for qsqN in qSqNum:

    if qsqN != qSqNum[ 0 ]:

        mpi_fncs.mpiPrintError( "Error (formFactors.py): " \
                           + "two-point function files " \
                           + "do not have the same number " \
                           + "of q^2. Number of q^2's are: " \
                           + ",".join( qSqNum ) + ".",
                           mpi_confs_info )

qNum = qNum[ 0 ]
qSqNum = qSqNum[ 0 ]

q = np.array( q, dtype=int )
#q = -1 * np.array( q, dtype=int )
qSq = np.array( qSq )
qSq_start = np.array( qSq_start )
qSq_end = np.array( qSq_end )
qSq_where = np.array( qSq_where )

# Write momentum transfer list file

if rank == 0:
    
    q_filename = rw.makeFilename( output_template,
                                  "momentum_transfer_list_pSq{}",
                                  pSq_fin )

    rw.writeMomentumList( q_filename, q[ 0 ] )

# Read final momentum list

p_fin = rw.readMomentaList( twopDir[ 0 ],
                            twop_template[ 0 ].format(smear_str_list[ 0 ]),
                            configList_loc[ 0 ],
                            particle,
                            srcNum[ -1 ], pSq_fin,
                            dataFormat_threep,
                            mpi_confs_info )

#p_fin = np.array( [ [ 1, 1, 1 ] ] )

finalMomentaNum = len( p_fin )


############################
# Read Two-point Functions #
############################

# Zero momentum two-point functions
# twop_q[ smr, c, q, t ]

twop_q = [ [] for smr in smear_str_list ]

# Loop over smears
for smr, ismr in zip( smear_str_list, range( smearNum ) ):

    twop_template_smr = twop_template[ ismr ].format( smr )

    twop_q[ ismr ] = rw.readTwopFile( twopDir[ ismr ],
                                      twop_template_smr,
                                      configList_loc, 
                                      configNum, q[ ismr ], qSq[ ismr ],
                                      qSq_start[ ismr ], qSq_end[ ismr ], 
                                      particle, srcNum[ -1 ],
                                      dataFormat_twop[ ismr ],
                                      mpi_confs_info )

# End loop over smears

twop_q = np.array( twop_q )

# Time dimension length

T = twop_q.shape[ -1 ]

# Time dimension length after fold

T_fold = T // 2 + 1

rangeEnd = T // 2 - 1

# Move configurations to first dimension
# twop_q[ smr, c, q, t ] -> twop_q[ c, smr, q, t ]

twop_q = np.moveaxis( twop_q, 1, 0 )


##########################################
# Jackknife and fold two-point functions #
##########################################


if binNum_loc:

    # twop_jk_loc[ b_loc, smr, q, t ]

    twop_jk_loc = fncs.jackknifeBinSubset( twop_q,
                                           binSize,
                                           binList_loc )

    # twop_jk_loc[ b_loc, smr, q, t ]

    twop_fold_loc = fncs.fold( twop_jk_loc )


    #############################################
    # Average two-point functions over momentum #
    #############################################

    
    # twop_loc[ b_loc, smr, qSq, t ]

    twop_loc = np.zeros( ( binNum_loc, smearNum, qSqNum, T_fold ) )

    for ismr in range( smearNum ):

        twop_loc[ :, ismr, :, : ] \
            = fncs.averageOverQsq( twop_fold_loc[ :, ismr, :, : ],
                                   qSq_start[ ismr ],
                                   qSq_end[ ismr ] )
        

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


# twop[ b, smr, qSq, t ]
# effEnergy[ b, smr, qSq, t ]

twop = np.zeros( ( binNum, smearNum, qSqNum, T_fold ) )
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
"""
if rank == 0:

    #iqSq_test = 1

    #twop_test_avg = np.average( twop[:,:,iqSq_test,:], axis=0 )
    #twop_test_err = fncs.calcError( twop[:,:,iqSq_test,:],
    #                                binNum, axis=0 )

    twop_test_avg = np.average( twop, axis=0 )
    twop_test_err = fncs.calcError( twop, binNum )

    for smr, ismr in zip( smear_str_list, range( smearNum ) ):

        for qsq, iqsq in fncs.zipXandIndex( qSq[ ismr ] ):

            filename_test_avg = rw.makeFilename( output_template,
                                                 "twop_test_avg_{}_qSq{}_" \
                                                 + "{}configs_binSize{}",
                                                 particle, qsq,
                                                 configNum, binSize )

            rw.writeAvgDataFile( filename_test_avg,
                                 twop_test_avg[ ismr, iqsq, : ],
                                 twop_test_err[ ismr, iqsq, : ] )

        filename_test_avg = rw.makeFilename( output_template,
                                             "twop_test_avg_{}_qSq{}_" \
                                             + "{}configs_binSize{}",
                                             particle, qSq[ ismr, iqSq_test ],
                                             configNum, binSize )

        rw.writeAvgDataFile( filename_test_avg,
                             twop_test_avg[ ismr ],
                             twop_test_err[ ismr ] )
"""

###########################
# Fit two-point functions #
###########################


# Initialize fit parameter arrays

c0 = np.zeros( ( smearNum, binNum, qSqNum ) )
c1 = np.zeros( ( smearNum, binNum, qSqNum ) )
E0 = np.zeros( ( smearNum, binNum, qSqNum ) )
E1 = np.zeros( ( smearNum, binNum, qSqNum ) )
mEff_plat = np.zeros( ( smearNum, binNum ) )

for ismr in range( smearNum ):

    # Calculate the plateau fit of the ground state effective mass
    # to use in the dispersion relation
    
    #mEff_plat[ ismr ], chiSq \
    #    = fit.fitPlateau_parallel( effEnergy[ :, ismr, 0, : ],
    #                               plat_fitStart, rangeEnd, 
    #                               mpi_confs_info )

    # fitResults = ( fitParams, chiSq, plat_fit,
    #                twop_t_low, plat_t_low, fitType )

    # Fit the two-point functions and effective mass

    fitResults = fit.effEnergyTwopFit( effEnergy[ :, ismr, 0, : ],
                                       twop[ :, ismr, 0, : ], rangeEnd,
                                       0, L, True, mpi_confs_info,
                                       plat_t_low_range=[args.plat_fit_start],
                                       tsf_t_low_range=[args.tsf_fit_start],
                                       checkFit=checkFit,
                                       fitType="twop" )

                                           

    # Set fitting parameters

    c0[ ismr, :, 0 ] = fitResults[ 0 ][ :, 0 ]
    c1[ ismr, :, 0 ] = fitResults[ 0 ][ :, 1 ]
    E0[ ismr, :, 0 ] = fitResults[ 0 ][ :, 2 ]
    E1[ ismr, :, 0 ] = fitResults[ 0 ][ :, 3 ]

    mEff_plat[ ismr ] = fitResults[ 2 ]

    if ismr == 0:

        # Set start of fit ranges

        tsf_fitStart = fitResults[ 3 ]
        plat_fitStart = fitResults[ 4 ]

    # Fit the two-point functions for each q > 0
    
    # Loop over momenta
    for iq in range( 1, qSqNum ):

        if dispRel:

            # Fit using E = sqrt( m^2 + p^2 )

            fitParams, chiSq \
                = fit.twoStateFit_twop_dispersionRelation( twop[ :, ismr, iq, : ],
                                                           tsf_fitStart,
                                                           rangeEnd, 
                                                           mEff_plat[ ismr ],
                                                           qSq[ ismr, iq ], L,
                                                           mpi_confs_info )

            c0[ ismr, :, iq ] = fitParams[ :, 0 ]
            c1[ ismr, :, iq ] = fitParams[ :, 1 ]
            E1[ ismr, :, iq ] = fitParams[ :, 2 ]

        else:

            # Fit with E0 a parameter

            E_guess = pq.energy( np.average( mEff_plat[ ismr ], axis=0 ),
                                 qSq[ ismr, iq ], L )

            fitParams, chiSq \
                = fit.twoStateFit_twop( twop[ :, ismr, iq, : ],
                                        tsf_fitStart,
                                        rangeEnd,
                                        E_guess, T,
                                        mpi_confs_info,
                                        method="BFGS")
            
            c0[ ismr, :, iq ] = fitParams[ :, 0 ]
            c1[ ismr, :, iq ] = fitParams[ :, 1 ]
            E0[ ismr, :, iq ] = fitParams[ :, 2 ]
            E1[ ismr, :, iq ] = fitParams[ :, 3 ]
            
        mpi_fncs.mpiPrint( "Fit two-point functions at " \
                           + "Q^2={}".format( qSq[ ismr, iq ] ),
                           mpi_confs_info )

    # End loop over q^2
# End loop over smear

# Average over bins

mEff_plat_avg = np.average( mEff_plat[ 0 ], axis=-1 )
mEff_plat_err = fncs.calcError( mEff_plat[ 0 ], binNum, axis=-1 )


##################################
# Write the effective mass files #
##################################


if rank == 0:

    # Average over bins
    # mEff_avg[ t ]

    mEff_avg = np.average( effEnergy[ :, 0, 0, : ], axis=-2 )
    mEff_err = fncs.calcError( effEnergy[ :, 0, 0, : ], 
                               binNum, axis=-2 )

    # Write average effective masses

    avgOutputFilename = rw.makeFilename( output_template, 
                                         "mEff_avg_{}" \
                                         + "_{}configs_binSize{}",
                                         particle,
                                         configNum, binSize )
    rw.writeAvgDataFile( avgOutputFilename, mEff_avg, mEff_err )

    # Write plateau fit

    mEff_plat_str = "2s" + str( plat_fitStart ) \
                    + ".2e" + str( rangeEnd )

    mEff_outputFilename = rw.makeFilename( output_template,
                                           "mEff_plat_{}_{}" \
                                           + "_{}configs_binSize{}",
                                           particle, mEff_plat_str, 
                                           configNum, binSize )

    rw.writeFitDataFile( mEff_outputFilename, mEff_plat_avg,
                         mEff_plat_err, plat_fitStart, rangeEnd )

    # Write c0 and plateau fit for each bin

    if particle == "kaon":

        for smr, ismr in zip( smear_str_list, range( smearNum ) ):

            c0_outputFilename = rw.makeFilename( output_template,
                                                 "c0_per_bin_{}{}_psq{}" \
                                                 + "_{}configs_binSize{}",
                                                 particle, smr, pSq_fin,
                                                 configNum, binSize )
            
            rw.writeDataFile_wX( c0_outputFilename, qSq[ ismr ], 
                                 c0[ ismr ] )

            mEff_plat_outputFilename = rw.makeFilename( output_template,
                                                 "mEff_plat_per_bin_{}{}_psq{}" \
                                                 + "_{}configs_binSize{}",
                                                 particle, smr, pSq_fin,
                                                 configNum, binSize )
            
            rw.write2ValueDataFile( mEff_plat_outputFilename, np.arange( binNum ),
                                    mEff_plat[ ismr ] )

    else:

        c0_outputFilename = rw.makeFilename( output_template,
                                             "c0_per_bin_{}_psq{}" \
                                             + "_{}configs_binSize{}",
                                             particle, pSq_fin,
                                             configNum, binSize )
        
        rw.writeDataFile_wX( c0_outputFilename, qSq[ 0 ], 
                             c0[ 0 ] )

        mEff_plat_outputFilename = rw.makeFilename( output_template,
                                             "mEff_plat_per_bin_{}_psq{}" \
                                             + "_{}configs_binSize{}",
                                             particle, pSq_fin,
                                             configNum, binSize )
        
        rw.write2ValueDataFile( mEff_plat_outputFilename, np.arange( binNum ), 
                                mEff_plat[ 0 ] )

# End first process

# Set the smear index for each flavor based on the particle

if particle == "pion" or particle == "nucleon":

    ismr_flav = [ 0 ]

else: # particle == "kaon"

    if pSq_fin == 0:

        # [ gN40a0p2_gN50a0p2, gN50a0p2_gN40a0p2 ]

        ismr_flav = [ 1, 2 ]

    else:

        # [ gN40a0p2_gN50a0p2, gN40a0p2 ]

        ismr_flav = [ 1, 0 ]


########################################
# Get q list for three-point functions #
########################################


q_threep= [ [ [] for p in p_fin ]
            for ts in tsink ]

if dataFormat_threep == "gpu":            
    # Loop over tsink
    for ts, its in zip( tsink, range( tsinkNum ) ) :
        # Loop over final momenta
        for p, ip in zip( p_fin, range( finalMomentaNum ) ):

            threep_template = "{}{}{}".format( threep_tokens[0],
                                               particle,
                                               threep_tokens[1] )

            q_threep[ its ][ ip ], \
                qNum_threep, \
                qSq_threep, \
                qSqNum_threep, \
                qSq_start_threep, \
                qSq_end_threep, \
                qSq_where_threep \
                = rw.readMomentumTransferList( threepDir, threep_template, 
                                               [configList[ 0 ]], particle, 
                                               srcNum[ its ],
                                               dataFormat_threep,
                                               args.momentum_transfer_list, 
                                               mpi_confs_info )
            
            if not np.array_equal( q_threep[ its ][ ip ],
                                   q_threep[ 0 ][ 0 ] ):
            
                mpi_fncs.mpiPrintError( "Error (formFactors.py): " \
                                        + "list of q in three-point " \
                                    + "funtion files are not consistant " \
                                        + "across tsink or final momentum.",
                                        mpi_confs_info )

        # End loop over final momenta
    # End loop over tsink

    q_threep = q_threep[ 0 ][ 0 ]

elif dataFormat_threep == "cpu":

    if np.any( df_twop == "cpu" for df_twop in dataFormat_twop ):

        # Loop over twop data formats
        for df_twop, ismr in fncs.zipXandIndex( dataFormat_twop ):

            if df_twop == "cpu":

                q_threep = q[ ismr ]
                qNum_threep = qNum
                qSq_threep = qSq[ ismr ]
                qSqNum_threep = qSqNum
                qSq_start_threep = qSq_start[ ismr ]
                qSq_end_threep = qSq_end[ ismr ]
                qSq_where_threep = qSq_where[ ismr ]
                
                break

            # End if cpu twop format
        # End loop over data formats
    
    else:

        error_str = "CPU data format not supported " \
                    + "for reading three-point " \
                    + "function momentum transfer list"

        mpi_fncs.mpiPrintError( error_str, mpi_confs_info )

# End cpu threep format

# Cut qSq_threep at iqSq_last

qSq_threep = qSq_threep[ :iqSq_last ]
qSqNum_threep = len( qSq_threep )
qSq_start_threep = qSq_start_threep[ :iqSq_last ]
qSq_end_threep = qSq_end_threep[ :iqSq_last ]
qSq_where_threep = qSq_where_threep[ :qSq_end_threep[ -1 ] + 1 ]

q_threep = q_threep[ :qSq_end_threep[ -1 ] + 1 ]
qNum_threep = len( q_threep )

# q needs sign change for rest frame strange part because
# complex conjugate is taken after Fourier transform

q_threep = np.array( [ q_threep, q_threep ] )

if pSq_fin == 0:

    q_threep[ 1 ] = -1 * q_threep[ 1 ]


##############################
# Read three-point functions #
##############################


# threep_jk[ ts, p, b, flav, q, proj*curr, t ]

threep_jk_loc = np.zeros( ( tsinkNum, finalMomentaNum,
                            binNum_loc, flavNum,
                            qNum_threep,
                            ratioNum,
                            T ) )
#threep_jk = np.zeros( ( tsinkNum, finalMomentaNum,
#                        binNum, flavNum,
#                        qNum_threep,
#                        ratioNum,
#                        T ) )

# Loop over tsink
for ts, its in fncs.zipXandIndex( tsink ):
    # Loop over final momenta
    for p, ip in fncs.zipXandIndex( p_fin ):

        # threep[ conf, flav, q, proj*curr, t ]

        threep = rw.readFormFactorFile( threepDir, threep_tokens,
                                        formFactor, srcNum[ its ],
                                        qSq_threep,
                                        qSq_start_threep,
                                        qSq_end_threep,
                                        qNum_threep,
                                        ts, projector,
                                        p, T, particle,
                                        dataFormat_threep,
                                        mpi_confs_info )


        ###################################
        # Jackknife three-point functions #
        ###################################


        # If bin on this process
        if binNum_loc:

            # Jackknife
            # threep_jk_loc[ ts, p, b, flav, q, proj*curr, t ]
            
            threep_jk_loc[ its, ip ] \
                = fncs.jackknifeBinSubset( threep,
                                           binSize,
                                           binList_loc )
        # End if bin on local process

        for flav, iflav in fncs.zipXandIndex( flav_str ):

            # Gather threep_jk
            
            threep_jk = np.zeros( ( binNum, T ), order='c' )
            
            threep_jk_loc_buffer = np.array( threep_jk_loc[ its, ip, :, iflav, 0, 0 ],
                                             order='c' )
            
            comm.Gatherv( threep_jk_loc_buffer,
                          [ threep_jk,
                            recvCount \
                            * np.prod( threep_jk_loc_buffer.shape[ 1: ] ),
                            recvOffset \
                            * np.prod( threep_jk_loc_buffer.shape[ 1: ] ),
                            MPI.DOUBLE ], root=0 )

            #if rank == 0:
            if False:

                # Average over bins

                threep_jk_avg = np.average( threep_jk, axis=0 )
                threep_jk_err = fncs.calcError( threep_jk, binNum )

                # Write threep files

                output_filename \
                    = rw.makeFilename( output_template,
                                       "{:}_threep_ti_{:}_{:}_tsink{:}" \
                                       + "_{:+}_{:+}_{:+}_Qsq0" \
                                       + "_{:}configs_binSize{:}",
                                       F_str[ 0 ],
                                       particle,
                                       flav_str[ iflav ],
                                       ts, p[ 0 ], p[ 1 ], p[ 2 ],
                                       configNum, binSize )
                    
                rw.writeAvgDataFile( output_filename,
                                     threep_jk_avg,
                                     threep_jk_err )

                output_filename \
                    = rw.makeFilename( output_template,
                                       "{:}_threep_per_bin_{:}_{:}_tsink{:}" \
                                       + "_{:+}_{:+}_{:+}_Qsq0" \
                                       + "_{:}configs_binSize{:}",
                                       F_str[ 0 ],
                                       particle,
                                       flav_str[ iflav ],
                                       ts, p[ 0 ], p[ 1 ], p[ 2 ],
                                       configNum, binSize )
                    
                rw.writeDataFile( output_filename,
                                  threep[ :, iflav, 0, 0 ] )
                
                output_filename \
                    = rw.makeFilename( output_template,
                                       "{:}_threep_jk_per_bin_{:}_{:}_tsink{:}" \
                                       + "_{:+}_{:+}_{:+}_Qsq0" \
                                       + "_{:}configs_binSize{:}",
                                       F_str[ 0 ],
                                       particle,
                                       flav_str[ iflav ],
                                       ts, p[ 0 ], p[ 1 ], p[ 2 ],
                                       configNum, binSize )
                
                rw.writeDataFile( output_filename,
                                  threep_jk )

            # End 1st process
        # End loop over flavor
    # End loop over final momenta
# End loop over tsink

# threep_jk[ ts, p, b_loc, flav, q, ratio, t ]
# -> threep_jk[ ts, flav, b_loc, p, q, ratio, t ]

threep_jk_loc = np.moveaxis( threep_jk_loc, [ 1, 3 ], [ 3, 1 ] )

# Change sign of strange part of final momentum
# because conjugate is taken after phase

p_fin = np.array( [ p_fin, -1 * p_fin ] )
#p_fin = [ -1 * p_fin, p_fin ]
        
####################
# Calculate ratios #
####################


# Calculate Q^2 = (p_f - p_i)^2 - (E_f - E_i)^2

Qsq_where = [ [] for flav in flav_str ]

# Loop over flavor
for iflav in range( flavNum ):

    # Qsq_loc[ b_loc, qs ], QsqNum, Qsq_where[ flav, qsq, p, q ]

    Qsq_loc, QsqNum, Qsq_where[ iflav ] \
        = pq.calcQsq( p_fin[ iflav ], q_threep[ iflav ],
                      mEff_plat[ 0, binList_loc ],
                      L, mpi_confs_info )

    if pSq_last:

        # Don't use Qsq where pSq_ini > pSq_last

        pSq_ini = np.zeros( ( finalMomentaNum, qNum_threep ) )

        # Loop over p_fin
        for p, ip in fncs.zipXandIndex( p_fin[ iflav ] ):
            # Loop over q
            for qq, iq in fncs.zipXandIndex( q_threep[ iflav ] ):
                
                pSq_ini[ ip, iq ] = np.dot( p - qq, p - qq )

            # End loop over q
        # End loop over p_fin
        
        for iQ in range( QsqNum ):

            Qsq_where[ iflav ][ iQ ] \
                = Qsq_where[ iflav ][ iQ ] & ( pSq_ini <= pSq_last )

    # End if pSq_last
# End loop over flavor

# Gather Q^2

Qsq = np.zeros( ( binNum, QsqNum ), dtype=float, order='c' )

comm.Gatherv( Qsq_loc,
              [ Qsq,
                recvCount \
                * np.prod( Qsq_loc.shape[ 1: ] ),
                recvOffset \
                * np.prod( Qsq_loc.shape[ 1: ] ),
                MPI.DOUBLE ],
              root=0 )

F_loc = np.zeros( ( binNum_loc, tsinkNum, flavNum, QsqNum, formFactorNum ),
                  dtype=float, order='c' )

# Loop over tsink
for ts, its in fncs.zipXandIndex( tsink ):

    # Loop over flavor
    for iflav in range( flavNum ):

        # Smear index for this flavor

        ismr = ismr_flav[ iflav ]

        # ratio_loc[ b_loc, p, q, proj*curr, t ]
        
        if particle == "nucleon":
            
            # Calculate ratio with two-point functions

            ratio_loc \
                = pq.calcFormFactorRatio( threep_jk_loc[ its, iflav ],
                                          twop_fold_loc[ binList_loc,
                                                         ismr,
                                                         :, : ], ts )

        else: # particle is meson

            if dispRel:

                # Calculate ratio with modified two-point functions
                # C = c0 exp( -sqrt( m^2 + p^2 ) t )

                ratio_loc \
                    =pq.calcFormFactorRatio_twopFit(threep_jk_loc[ its,
                                                                   iflav ],
                                                    c0[ ismr,
                                                    binList_loc ],
                                                    mEff_plat[ ismr,
                                                               binList_loc ],
                                                    ts, p_fin[ iflav ],
                                                    q_threep[ iflav ],
                                                    qSq[ ismr ],
                                                    L, True, mpi_confs_info)

            else:

                # Calculate ratio with modified two-point functions
                # C = c0 exp( -E0 t )

                ratio_loc \
                        =pq.calcFormFactorRatio_twopFit(threep_jk_loc[ its,
                                                                   iflav ],
                                                        c0[ ismr,
                                                            binList_loc ],
                                                        E0[ ismr,
                                                            binList_loc ],
                                                        ts, p_fin[ iflav ],
                                                        q_threep[ iflav ],
                                                        qSq[ ismr ],
                                                        L, False, mpi_confs_info)
        
        # End if meson

        # Gather ratio
        # ratio[ b, p, q, proj*curr, t ]

        ratio = np.zeros( ( binNum, ) + ratio_loc.shape[ 1: ] )

        comm.Allgatherv( ratio_loc,
                         [ ratio,
                           recvCount \
                           * np.prod( ratio_loc.shape[ 1: ] ),
                           recvOffset \
                           * np.prod( ratio_loc.shape[ 1: ] ),
                           MPI.DOUBLE ] )

        #if rank == 0:
        if False:

            # Average over bins

            ratio_avg = np.average( ratio, axis=0 )
            ratio_err = fncs.calcError( ratio, binNum )

            # Write ratios for each final momentum

            for p, ip in fncs.zipXandIndex( p_fin[ iflav ] ):

                output_filename \
                    = rw.makeFilename( output_template,
                                       "{:}_ratio_{:}_{:}_tsink{:}" \
                                       + "_{:+}_{:+}_{:+}_Qsq0" \
                                       + "_{:}configs_binSize{:}",
                                       F_str[ 0 ],
                                       particle,
                                       flav_str[ iflav ],
                                       ts, p[ 0 ], p[ 1 ], p[ 2 ],
                                       configNum, binSize )

                rw.writeAvgDataFile( output_filename,
                                     ratio_avg[ ip, 0, 0 ],
                                     ratio_err[ ip, 0, 0 ] )

        # ratio_err[ p, q, ratio, t ]

        ratio_err = fncs.calcError( ratio, binNum )

        # ratio_fit_loc[ b_loc, p, q, ratio ]

        if binNum_loc:

            # Fit ratios to a plateau fit

            ratio_fit_loc = fit.fitFormFactor( ratio_loc,
                                               ratio_err,
                                               ts, 1 )

            # Multiply by renormalization factor

            if formFactor == "A20_B20":

                # Off-diagonal insertions have a different factor

                ratio_fit_loc[ ..., 0 ] = Z[ 0 ] * ratio_fit_loc[ ..., 0 ]

                ratio_fit_loc[ ..., 1: ] = Z[ 1 ] * ratio_fit_loc[ ..., 1: ]

            else:

                ratio_fit_loc = Z * ratio_fit_loc

        else: # No bin on this process

            ratio_fit_loc = None

        # Gather ratio fits
        # ratio_fit[ b, p, q, proj*curr ]

        ratio_fit = np.zeros( ( binNum, ) + ratio_fit_loc.shape[ 1: ] )

        comm.Allgatherv( ratio_fit_loc,
                         [ ratio_fit,
                           recvCount \
                           * np.prod( ratio_fit_loc.shape[ 1: ] ),
                           recvOffset \
                           * np.prod( ratio_fit_loc.shape[ 1: ] ),
                           MPI.DOUBLE ] )

        #if rank == 0:
        if False:

            # Average over bins

            ratio_fit_avg = np.average( ratio_fit, axis=0 )
            ratio_fit_err = fncs.calcError( ratio_fit, binNum )

            # Write ratio_fits for each final momentum

            for p, ip in fncs.zipXandIndex( p_fin[ iflav ] ):

                output_filename \
                    = rw.makeFilename( output_template,
                                       "{:}_ratio_fit_per_bin_{:}_{:}_tsink{:}" \
                                       + "_{:+}_{:+}_{:+}_Qsq0" \
                                       + "_{:}configs_binSize{:}",
                                       F_str[ 0 ],
                                       particle,
                                       flav_str[ iflav ],
                                       ts, p[ 0 ], p[ 1 ], p[ 2 ],
                                       configNum, binSize )

                rw.writeAvgDataFile( output_filename,
                                     ratio_fit[ :, ip, 0, 0 ],
                                     np.zeros( binNum ) )

                output_filename \
                    = rw.makeFilename( output_template,
                                       "{:}_ratio_fit_{:}_{:}_tsink{:}" \
                                       + "_{:+}_{:+}_{:+}_Qsq0" \
                                       + "_{:}configs_binSize{:}",
                                       F_str[ 0 ],
                                       particle,
                                       flav_str[ iflav ],
                                       ts, p[ 0 ], p[ 1 ], p[ 2 ],
                                       configNum, binSize )

                rw.writeFitDataFile( output_filename,
                                     ratio_fit_avg[ ip, 0, 0 ],
                                     ratio_fit_err[ ip, 0, 0 ],
                                     ts // 2 - 1, ts // 2 + 1)

        # ratio_fit_err[ p, q, ratio ]

        ratio_fit_err = fncs.calcError( ratio_fit, binNum )        

        # Calculate kinematic factors
        # kineFacter_loc[ b_loc, p, q, r, [ F1, F2 ] ]
        
        kineFactor_loc = pq.kineFactor( ratio_fit_err,
                                        formFactor,
                                        particle,
                                        flav_str[ iflav ],
                                        mEff_plat[ 0, binList_loc ],
                                        p_fin[ iflav ], q_threep[ iflav ], L,
                                        mpi_confs_info )

        if pSq_fin == 0:
        #if True:
        #for p, ip in fncs.zipXandIndex( p_fin[ iflav ] ):
        #if False:

            # Calculate F as a function of ti for Q^2=1
            # F_ti_loc[ b_loc, ff, t ]

            F_ti_loc \
                = Z * pq.calcFormFactors_ti( kineFactor_loc,
                                             ratio_loc, ratio_fit_err,
                                             Qsq_where[ iflav ], 1,
                                             mpi_confs_info )

            #F_ti_loc = np.zeros( ( binNum_loc, len( F_str ), ts + 1 ) )

            # Loop over bins
            #for ib in range( binNum_loc ):
            # Loop over form factors
            #    for ff, iff in fncs.zipXandIndex( F_str ):
                        
            #        F_ti_loc[ ib, iff ] = Z * ratio_loc[ ib, ip, 0, 0, : ] \
            #                              / kineFactor_loc[ ib, ip, 0, 0, iff ] \
            #                              / ratio_fit_err[ ip, 0, 0 ] ** 2
            
            # Gather F_ti
            # F_ti[ b, ff, t ]

            F_ti = np.zeros( ( binNum, ) + F_ti_loc.shape[ 1: ] )
    
            comm.Gatherv( F_ti_loc,
                          [ F_ti,
                            recvCount \
                            * np.prod( F_ti_loc.shape[ 1: ] ),
                            recvOffset \
                            * np.prod( F_ti_loc.shape[ 1: ] ),
                            MPI.DOUBLE ],
                          root=0 )

            if rank == 0:

                # Average over bins

                F_ti_avg = np.average( F_ti, axis=0 )
                F_ti_err = fncs.calcError( F_ti, binNum )

                # Write form factor output files

                # Loop over form factors
                for ff, iff in fncs.zipXandIndex( F_str ):
                    
                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_ti_{}_{}_tsink{}"
                                           + "_{:+}_{:+}_{:+}_Qsq{}"
                                           + "_{}configs_binSize{}",
                                           ff, particle, flav_str[ iflav ],
                                           ts, p[ 0 ], p[ 1 ], p[ 2 ], 1,
                                           configNum, binSize )

                    rw.writeAvgDataFile( output_filename,
                                         F_ti_avg[ iff ],
                                         F_ti_err[ iff ] )

                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_ti_{}_{}_tsink{}"
                                           + "_psq{}_Qsq{}"
                                           + "_{}configs_binSize{}",
                                           ff, particle, flav_str[ iflav ],
                                           ts, pSq_fin, 1,
                                           configNum, binSize )

                    rw.writeAvgDataFile( output_filename,
                                         F_ti_avg[ iff ],
                                         F_ti_err[ iff ] )

                # End loop over form factors
            # End 1st process
        # End p^2 = 0

        # Gather kineFactor
        # kineFactor[ b, p, q, r, [ F1, F2 ] ]

        #kineFactor = np.zeros( ( binNum, ) + kineFactor_loc.shape[ 1: ] )
    
        #comm.Allgatherv( kineFactor_loc,
        #                 [ kineFactor,
        #                   recvCount \
        #                   * np.prod( kineFactor_loc.shape[ 1: ] ),
        #                   recvOffset \
        #                   * np.prod( kineFactor_loc.shape[ 1: ] ),
        #                   MPI.DOUBLE ] )

        # Calculate form factors from ratio fits
        # and kinematic factors using SVD
        # F_loc[ b_loc, ts, flav, qs, ff ]

        F_loc[ :, its, iflav, :, : ] \
            = pq.calcFormFactors_SVD( kineFactor_loc,
                                      ratio_fit,
                                      ratio_fit_err,
                                      Qsq_where[ iflav ],
                                      formFactor,
                                      pSq_fin,
                                      mpi_confs_info )
        
    # End loop over flavor

    mpi_fncs.mpiPrint( "Calculated form factors for "
                       + "tsink={}".format( ts ),
                       mpi_confs_info )

# End loop over tsink


#################
# 2-state fit #
#################


# F_tsf_loc[ b_loc, flav, qsq, [ F1, F2 ] ]

F_tsf_loc = np.zeros( ( binNum_loc, flavNum,
                        QsqNum, formFactorNum ),
                      dtype=float, order='c' )

# Number of three-point functions to neglect on source and sink sides

neglect = 3

# Qsq_where_good[ flav, qs ]

Qsq_where_good = np.full( ( flavNum, QsqNum ), True, dtype=bool )

mpi_fncs.mpiPrint( "Performing 2-state fit...",
                   mpi_confs_info, flush=True )

# Loop over flavors
for iflav in range( flavNum ):
        
    # Smear index for this flavor
    
    ismr = ismr_flav[ iflav ]

    # Perform 2-state fit
    # tsfParams[ b, p, q, r, [ A00, A01, A10, A11 ] ]

    if dispRel:
        
        # Calculate 2-state fit parameters using dispersion relation
        # to calculate E0 = sqrt( mEff^2 + p^2 )

        tsfParams \
            = fit.twoStateFit_threep_momTransfer( threep_jk_loc[ :,
                                                                 iflav ],
                                                  tsink,
                                                  mEff_plat[ ismr ],
                                                  E1[ ismr ],
                                                  p_fin[ iflav ],
                                                  q_threep[ iflav ],
                                                  qSq[ ismr ],
                                                  pSq_last,
                                                  neglect, L, True,
                                                  mpi_confs_info)

    else:

        # Calculate 2-state fit parameters using E0 from 
        # 2-state fit on two-point functions

        tsfParams \
            = fit.twoStateFit_threep_momTransfer(threep_jk_loc[ :,
                                                                iflav ],
                                                 tsink,
                                                 E0[ ismr ],
                                                 E1[ ismr ],
                                                 p_fin[ iflav ],
                                                 q_threep[ iflav ],
                                                 qSq[ ismr ],
                                                 pSq_last,
                                                 neglect, L, False,
                                                 mpi_confs_info)
        
    # 2-state fit parameters

    a00 = tsfParams[ ..., 0 ]
    a01 = tsfParams[ ..., 1 ]
    a10 = tsfParams[ ..., 2 ]
    a11 = tsfParams[ ..., 3 ]

    # Ratio from 2-state fit
    # Ratio_tsf_loc[ b_loc, p, q, r ]
    
    ratio_tsf_loc \
        = pq.calcFormFactorRatio_tsf( a00[ binList_loc ],
                                      c0[ ismr,
                                          binList_loc ],
                                      p_fin[ iflav ],
                                      q_threep[ iflav ],
                                      qSq[ ismr ],
                                      mpi_confs_info )

    # Multiply by renormalization factor

    if formFactor == "A20_B20":

        # Off-diagonal insertions have a different factor
        
        ratio_tsf_loc[ ..., 0 ] = Z[ 0 ] * ratio_tsf_loc[ ..., 0 ]

        ratio_tsf_loc[ ..., 1: ] = Z[ 1 ] * ratio_tsf_loc[ ..., 1: ]

    else:

        ratio_tsf_loc = Z * ratio_tsf_loc

    # Gather ratio
    # ratio_tsf[ b, p, q, r ]
        
    ratio_tsf = np.zeros( ( binNum, ) + ratio_tsf_loc.shape[ 1: ] )

    comm.Allgatherv( ratio_tsf_loc,
                     [ ratio_tsf,
                       recvCount \
                       * np.prod( ratio_tsf_loc.shape[ 1: ] ),
                       recvOffset \
                       * np.prod( ratio_tsf_loc.shape[ 1: ] ),
                       MPI.DOUBLE ] )

    # ratio_tsf_err[ p, q, ratio ]

    ratio_tsf_err = fncs.calcError( ratio_tsf, binNum )        


    ###############################
    # Calculate kinematic factors #
    ###############################


    # kineFacter_tsf_loc[ b_loc, p, q, r, [ F1, F2 ] ]
        
    kineFactor_tsf_loc = pq.kineFactor( ratio_tsf_err,
                                        formFactor,
                                        particle,
                                        flav_str[ iflav ],
                                        mEff_plat[ 0, binList_loc ],
                                        p_fin[ iflav ],
                                        q_threep[ iflav ], L,
                                        mpi_confs_info )
        
    # Gather kinematic factor
    # kineFactor_tsf[ b, p, q, r, [ F1, F2 ] ]

    kineFactor_tsf = np.zeros( ( binNum, )
                               + kineFactor_tsf_loc.shape[ 1: ] )
    
    comm.Allgatherv( kineFactor_tsf_loc,
                     [ kineFactor_tsf,
                       recvCount \
                       * np.prod( kineFactor_tsf_loc.shape[ 1: ] ),
                       recvOffset \
                       * np.prod( kineFactor_tsf_loc.shape[ 1: ] ),
                       MPI.DOUBLE ] )

    # Calculate 2-state fit form factor
    # F_tsf_loc[ b_loc, flav, qs, ff ]

    F_tsf_loc[ :, iflav ] = pq.calcFormFactors_SVD( kineFactor_tsf_loc,
                                                    ratio_tsf,
                                                    ratio_tsf_err,
                                                    Qsq_where[ iflav ],
                                                    formFactor,
                                                    pSq_fin,
                                                    mpi_confs_info )

# End loop over flavor

# Gather 2-state fit form factors
# F_tsf[ b, flav, qsq, [ F1, F2 ] ]

# F_tsf[ b, flav, qsq, [ F1, F2 ] ]

if rank == 0:

    F_tsf = np.zeros( ( binNum, flavNum,
                        QsqNum, formFactorNum ),
                      dtype=float, order='c' )
    
else:

    F_tsf = np.array( [] )

comm.Gatherv( F_tsf_loc,
              [ F_tsf,
                recvCount * np.prod( F_tsf_loc.shape[ 1: ] ),
                recvOffset * np.prod( F_tsf_loc.shape[ 1: ] ),
                MPI.DOUBLE ],
              root=0 )

if rank == 0:
          
    F_tsf_err = fncs.calcError( F_tsf, binNum )

    F_tsf_err = np.array( [ F_tsf_err ] * binNum )
    F_tsf_err = F_tsf_err.reshape( F_tsf.shape )
        
    # Loop over flavors
    for iflav in range( flavNum ):

        # Loop over Q^2 index
        for iqs in range( QsqNum ):
                    
            # Q^2 are bad if they do not correspond to any
            # p^2 < pSq_last

            if not np.any( Qsq_where[ iflav ][ iqs ] ):
        
                Qsq_where_good[ :, iqs ] = False

            elif errorThreshold:

                # Q^2 are bad if dF(Q^2)/F(Q^2) > errorThreshold

                if np.any( F_tsf_err[ :, iflav, iqs, 0 ]
                           / np.abs( F_tsf[ :, iflav, iqs, 0 ] )
                           > errorThreshold ):
                    
                    Qsq_where_good[ iflav, iqs ] = False
                        
                # End if dF/F > errorThreshold
            # End errorThreshold
        # End loop over Q^2

    # End loop over flavor
# End first process


##########################################
# Get "good" form factors and write them #
##########################################


# Broadcast Qsq_where_good from first process

if rank != 0:

    Qsq_where_good = np.full( ( flavNum, QsqNum ),
                              False, dtype=bool )

comm.Bcast( Qsq_where_good, root=0 )

F_good_loc = [ [] for flav in flav_str ]

F_avg = [ [] for flav in flav_str ]
F_err = [ [] for flav in flav_str ]

F_good_tsf_loc = [ [] for flav in flav_str ]

F_tsf_avg = [ [] for flav in flav_str ]
F_tsf_err = [ [] for flav in flav_str ]

Qsq_GeV = [ [] for flav in flav_str ]
Qsq_GeV_loc = [ [] for flav in flav_str ]

Qsq_GeV_avg = [ [] for flav in flav_str ]

# Loop over flavor
for iflav in range( flavNum ):
            
    # Get results for good Q^2
    # F_good_loc[ iflav ][ b_loc, its, qs_good, [ F1, F2 ] ]

    F_good_loc[ iflav ] = F_loc[ :, :, iflav ]

    F_good_loc[ iflav ] = F_good_loc[ iflav ][ :, :,
                                               Qsq_where_good[ iflav ], : ]

    F_good_loc[ iflav ] = np.array( F_good_loc[ iflav ],
                                    order='c' )
    
    # Convert good Q^2 to GeV^2
    # Qsq_GeV[ b, qs_good ]

    Qsq_GeV[ iflav ] = Qsq[ :, Qsq_where_good[ iflav ] ] \
                       * ( 0.197 / a ) ** 2

    # Qsq_GeV_loc[ b_loc, qs_good ]

    Qsq_GeV_loc[ iflav ] = Qsq_loc[ :, Qsq_where_good[ iflav ] ] \
                           * ( 0.197 / a ) ** 2

    # Gather good form factors
    # F_good[ iflav ][ b, ts, qs, ff ]

    if rank == 0:
        
        F_good = np.zeros( ( binNum, )
                           + F_good_loc[ iflav ].shape[ 1: ],
                           dtype=float, order='c' )
            
    else:
    
        F_good = np.array( [] )

    comm.Gatherv( F_good_loc[ iflav ],
                  [ F_good,
                    recvCount * np.prod( F_good_loc[ iflav ].shape[ 1: ] ),
                    recvOffset * np.prod( F_good_loc[ iflav ].shape[ 1: ] ),
                    MPI.DOUBLE ],
                  root=0 )

    if rank == 0:

        # Average over bins
        # F_avg[ iflav ][ ts, qs_good, ff ]

        F_avg[ iflav ] = np.average( F_good, axis=0 )
        F_err[ iflav ] = fncs.calcError( F_good, binNum )

        Qsq_GeV_avg[ iflav ] = np.average( Qsq_GeV[ iflav ], axis=0 )

        # Loop over tsink
        for ts, its in fncs.zipXandIndex( tsink ):
            # Loop over form factors
            for ff, iff in fncs.zipXandIndex( F_str ):

                # Write form factors for each bin

                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_per_bin_{}_{}_tsink{}_psq{}" \
                                       + "_{}configs_binSize{}",
                                       ff, particle,
                                       flav_str[ iflav ],
                                       ts, pSq_fin,
                                       configNum, binSize )

                rw.writeDataFile_wX( output_filename, Qsq_GeV[ iflav ],
                                     F_good[ :, its, :, iff ] )

                # Write bin averaged form factors

                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_{}_{}_tsink{}_psq{}" \
                                       + "_{}configs_binSize{}",
                                       ff, particle,
                                       flav_str[ iflav ],
                                       ts, pSq_fin,
                                       configNum, binSize )

                rw.writeAvgDataFile_wX( output_filename,
                                        Qsq_GeV_avg[ iflav ],
                                        F_avg[ iflav ][ its, :, iff ],
                                        F_err[ iflav ][ its, :, iff ] )

            # End loop over form factors
        # End loop over tsink
    # End first process

    # Get results for good Q^2
    # F_good_tsf_loc[ flav ][ b, qs_good, [ F1, F2 ] ]

    F_good_tsf_loc[ iflav ] = F_tsf_loc[ :, iflav ]
    F_good_tsf_loc[ iflav ] \
        = F_good_tsf_loc[ iflav ][ :, Qsq_where_good[ iflav ], : ]
    F_good_tsf_loc[ iflav ] = np.array( F_good_tsf_loc[ iflav ],
                                        order='c' )
    
    # Gather 2-state form factors

    if rank == 0:

        F_good_tsf = np.zeros( ( binNum, )
                                        + F_good_tsf_loc[ iflav ].shape[1:],
                                        dtype=float, order='c' )

    else:

        F_good_tsf = np.array( [] )

    comm.Gatherv( F_good_tsf_loc[ iflav ],
                  [ F_good_tsf,
                    recvCount
                    * np.prod( F_good_tsf_loc[ iflav ].shape[ 1: ] ),
                    recvOffset
                    * np.prod( F_good_tsf_loc[ iflav ].shape[ 1: ] ),
                    MPI.DOUBLE ],
                  root=0 )

    if rank == 0:

        # Average over bins
        # F_tsf_avg[ flav ][ qs_good, [ F1, F2 ] ]

        F_tsf_avg[ iflav ] = np.average( F_good_tsf, axis=0 )
        F_tsf_err[ iflav ] = fncs.calcError( F_good_tsf, binNum )

        for ff, iff in fncs.zipXandIndex( F_str ):

            # Write form factors for each bin

            output_filename \
                = rw.makeFilename( output_template,
                                   "{}_per_bin_2sf_{}_{}_"
                                   + "tsink{}_{}_psq{}"
                                   + "_{}configs_binSize{}",
                                   ff, particle, flav_str[ iflav ],
                                   tsink[ 0 ], tsink[ -1 ], pSq_fin,
                                   configNum, binSize )

            rw.writeDataFile_wX( output_filename, Qsq_GeV[ iflav ],
                                 F_good_tsf[ :, :, iff ] )

            # Write bin averaged form factors

            output_filename \
                = rw.makeFilename( output_template,
                                   "{}_2sf_{}_{}_tsink{}_{}_psq{}"
                                   + "_{}configs_binSize{}",
                                   ff, particle, flav_str[ iflav ],
                                   tsink[ 0 ],
                                   tsink[ -1 ],
                                   pSq_fin,
                                   configNum, binSize )

            rw.writeAvgDataFile_wX( output_filename, Qsq_GeV_avg[ iflav ],
                                    F_tsf_avg[ iflav ][ :, iff ],
                                    F_tsf_err[ iflav ][ :, iff ] )
    
        # End loop over form factor
    # End first process
# End loop over flavors


##############################
# Flavor combination for F_H #
##############################


if formFactor == "GE_GM":

    # Get results for good Q^2
    # F_flavCombo_loc[ b_loc, ts, qs_good, [ F1, F2 ] ]
    
    if particle in [ "kaon", "nucleon" ]:

        # F_K = 2/3 F_u - 1/3 F_s
        # F_N = 2/3 F_u - 1/3 F_d

        Qsq_where_good_flavCombo \
            = np.array( Qsq_where_good[ 0 ] & Qsq_where_good[ 1 ],
                        dtype=bool )

        F_good_flavCombo = F_loc[ :, :, :,
                                  Qsq_where_good_flavCombo, : ]

        F_flavCombo_loc = 2./3. * F_good_flavCombo[ :, :, 0 ] \
                          - 1./3. * F_good_flavCombo[ :, :, 1 ]
        
    else: # particle == "pion"

        # F_pi = 2/3 F_u - 1/3 F_d = F_u
            
        F_flavCombo_loc = F_good_loc[ 0 ]

        Qsq_where_good_flavCombo = Qsq_where_good[ 0 ]
            
    # End pion

    F_flavCombo_loc = np.array( F_flavCombo_loc,
                                order='c' )

    # Convert Q^2 to GeV^2
            
    Qsq_GeV_flavCombo \
        = Qsq[ :, Qsq_where_good_flavCombo ] * ( 0.197 / a ) ** 2
    
    Qsq_GeV_flavCombo_loc \
        = Qsq_loc[ :, Qsq_where_good_flavCombo ] * ( 0.197 / a ) ** 2
    
    # Gather flavor combo form factors
    # F_flavCombo[ b, ts, qs_good, ff ]

    if rank == 0:

        F_flavCombo = np.zeros( ( binNum, ) + F_flavCombo_loc.shape[ 1: ],
                                dtype=float, order='c' )

    else:

        F_flavCombo = np.array( [] )

    comm.Gatherv( F_flavCombo_loc,
                  [ F_flavCombo,
                    recvCount * np.prod( F_flavCombo_loc.shape[ 1: ] ),
                    recvOffset * np.prod( F_flavCombo_loc.shape[ 1: ] ),
                    MPI.DOUBLE ],
                  root=0 )

    if rank == 0:

        # Average over bins
        # F_flavCombo_avg[ ts, qs_good, ff ]

        F_flavCombo_avg = np.average( F_flavCombo, axis=0 )
        F_flavCombo_err = fncs.calcError( F_flavCombo, binNum )
        
        Qsq_GeV_flavCombo_avg = np.average( Qsq_GeV_flavCombo, axis=0 )

        # Write output

        # Loop over tsink
        for ts, its in fncs.zipXandIndex( tsink ):
            # Loop over form factors
            for ff, iff in fncs.zipXandIndex( F_str ):

                # Write form factors for each bin

                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_per_bin_{}_tsink{}_psq{}" \
                                       + "_{}configs_binSize{}",
                                       ff, particle,
                                       ts, pSq_fin,
                                       configNum, binSize )

                rw.writeDataFile_wX( output_filename, Qsq_GeV_flavCombo,
                                     F_flavCombo[ :, its, :, iff ] )

                # Write bin-averaged form factors

                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_{}_tsink{}_psq{}" \
                                       + "_{}configs_binSize{}",
                                       ff, particle,
                                       ts, pSq_fin,
                                       configNum, binSize )

                rw.writeAvgDataFile_wX(output_filename,Qsq_GeV_flavCombo_avg,
                                       F_flavCombo_avg[ its, :, iff ],
                                       F_flavCombo_err[ its, :, iff ] )
            
            # End loop over form factor
        # End loop over tsink
    # End first process

    # Flavor combination for 2-state F_H

    if particle in [ "kaon", "nucleon" ]:

        # Get results for good Q^2
        # F_flavCombo_tsf[ b, qs_good, [ F1, F2 ] ]
           
        F_good_flavCombo_tsf = F_tsf_loc[ :, :,
                                          Qsq_where_good_flavCombo, : ]

        # F_K = 2/3 F_u - 1/3 F_s
        # F_N = 2/3 F_u - 1/3 F_d
        
        F_flavCombo_tsf_loc = 2./3. * F_good_flavCombo_tsf[ :, 0 ] \
                              - 1./3. * F_good_flavCombo_tsf[ :, 1 ]
        
    else: # particle == "pion"

        # F_pi = 2/3 F_u - 1/3 F_d = F_u
            
        F_flavCombo_tsf_loc = F_good_tsf_loc[ 0 ]

    # End pion

    F_flavCombo_tsf_loc = np.array( F_flavCombo_tsf_loc,
                                    order='c' )

    # Gather flavor combo form factors

    if rank == 0:

        F_flavCombo_tsf \
            = np.zeros( ( binNum, )
                        + F_flavCombo_tsf_loc.shape[1:],
                        dtype=float, order='c' )

    else:

        F_flavCombo_tsf = np.array( [] )

    comm.Gatherv( F_flavCombo_tsf_loc,
                  [ F_flavCombo_tsf,
                    recvCount
                    * np.prod( F_flavCombo_tsf_loc.shape[ 1: ] ),
                    recvOffset
                    * np.prod( F_flavCombo_tsf_loc.shape[ 1: ] ),
                    MPI.DOUBLE ],
                  root=0 )

    if rank == 0:

        # Average over bins
        
        F_flavCombo_tsf_avg = np.average( F_flavCombo_tsf, axis=0 )
        F_flavCombo_tsf_err = fncs.calcError( F_flavCombo_tsf, binNum )
                      
        # Write output
                
        # Loop over form factors
        for ff, iff in fncs.zipXandIndex( F_str ):

            # Write form factors for each bin
                
            output_filename \
                = rw.makeFilename( output_template,
                                   "{}_per_bin_2sf_{}_tsink{}_{}_psq{}"
                                   + "_{}configs_binSize{}",
                                   ff, particle,
                                   tsink[ 0 ], tsink[ -1 ],
                                   pSq_fin,
                                   configNum, binSize )
            
            rw.writeDataFile_wX( output_filename, Qsq_GeV_flavCombo,
                                 F_flavCombo_tsf[ :, :, iff ] )

            # Write bin-averaged form factors

            output_filename \
                = rw.makeFilename( output_template,
                                   "{}_2sf_{}_tsink{}_{}_psq{}"
                                   + "_{}configs_binSize{}",
                                   ff, particle,
                                   tsink[ 0 ], tsink[ -1 ],
                                   pSq_fin,
                                   configNum, binSize )

            rw.writeAvgDataFile_wX( output_filename, Qsq_GeV_flavCombo_avg,
                                    F_flavCombo_tsf_avg[ :, iff ],
                                    F_flavCombo_tsf_err[ :, iff ] )
            
        # End loop over form factor
    # End first process
# End GE_GM


#################################################
# Fit the form factors to a monopole distribution #
#################################################


# Broadcast F_avg and F_err
# F_avg[ iflav ][ ts, qs, ff ]
# F_err[ iflav ][ ts, qs, ff ]
# F_tsf_avg[ iflav ][ qs, ff ]
# F_tsf_err[ iflav ][ qs, ff ]

if rank != 0:

    # Loop over flavor
    for iflav in range( flavNum ):
            
        F_avg[ iflav ] = np.zeros( F_good_loc[ iflav ].shape[ 1: ] )
        F_err[ iflav ] = np.zeros( F_good_loc[ iflav ].shape[ 1: ] )

        F_tsf_avg[ iflav ] = np.zeros( F_good_tsf_loc[ iflav ].shape[ 1: ] )
        F_tsf_err[ iflav ] = np.zeros( F_good_tsf_loc[ iflav ].shape[ 1: ] )

    # End loop over flavor
# End not first process
    
# Loop over flavor
for iflav in range( flavNum ):
            
    comm.Bcast( F_avg[ iflav ], root=0 )
    comm.Bcast( F_err[ iflav ], root=0 )

    comm.Bcast( F_tsf_avg[ iflav ], root=0 )
    comm.Bcast( F_tsf_err[ iflav ], root=0 )

    if formFactor == "BT10":

        # Remove elements where form factors are zero

        F_nonzero_where = F_tsf_avg[ iflav ] != 0

        Qsq_GeV_loc[ iflav ] \
            = Qsq_GeV_loc[ iflav ][ :, F_nonzero_where[ :, 0 ] ]
            
        QsqNum_good = Qsq_GeV_loc[ iflav ].shape[ -1 ]

        F_good_loc[ iflav ] \
            = F_good_loc[ iflav ][ :, :, F_nonzero_where ]
        F_good_loc[ iflav ] \
            = F_good_loc[ iflav ].reshape( binNum_loc,
                                           tsinkNum,
                                           QsqNum_good, 1 )
            
        F_err[ iflav ] = F_err[ iflav ][ :, F_nonzero_where ]
        F_err[ iflav ] \
            = F_err[ iflav ].reshape( tsinkNum,
                                      QsqNum_good, 1 )
            
        F_avg[ iflav ] = F_avg[ iflav ][ :, F_nonzero_where ]
        F_avg[ iflav ] \
            = F_avg[ iflav ].reshape( tsinkNum,
                                      QsqNum_good, 1 )

        F_good_tsf_loc[ iflav ] \
            = F_good_tsf_loc[ iflav ][ :, F_nonzero_where ]
        F_good_tsf_loc[ iflav ] \
            = F_good_tsf_loc[ iflav ].reshape( binNum_loc,
                                               QsqNum_good, 1 )

        F_tsf_err[ iflav ] = F_tsf_err[ iflav ][ F_nonzero_where ]
        F_tsf_err[ iflav ] \
            = F_tsf_err[ iflav ].reshape( QsqNum_good, 1 )

        F_tsf_avg[ iflav ] = F_tsf_avg[ iflav ][ F_nonzero_where ]
        F_tsf_avg[ iflav ] \
            = F_tsf_avg[ iflav ].reshape( QsqNum_good, 1 )

    # End BT10

    # Set M_monopole guess

    M0 = -1.0 if formFactor == "BT10" else None
    F0=None

    # Loop over number of parameters
    for paramNum_monopole in 1, 2:
        # Loop over tsink
        for ts, its in fncs.zipXandIndex( tsink ):
            # Loop over form factors
            for ff, iff in fncs.zipXandIndex( F_str ):
            
                # Fit form factors to monopole
                # fitParams_monopole_loc[ b_loc, param ]

                fitParams_monopole_loc, chiSq_monopole_loc \
                    =fit.fitFormFactor_monopole(F_good_loc[iflav][:,its,:,iff],
                                              F_err[ iflav ][ its, :, iff ],
                                              Qsq_GeV_loc[ iflav ],
                                              paramNum_monopole,
                                              mpi_confs_info,
                                              Qsq_last=Qsq_last_monopole,
                                              M0=M0, F0=F0 )
                               
                # Gather monopole fit parameters to first rank
                # fitParams_monopole[ b, param ]

                if rank == 0:

                    fitParams_monopole = np.zeros( ( binNum, 2 ) )
                
                else:

                    fitParams_monopole = []

                comm.Gatherv( fitParams_monopole_loc,
                              [ fitParams_monopole,
                                recvCount
                                * np.prod( fitParams_monopole_loc.shape[1:] ),
                                recvOffset
                                * np.prod( fitParams_monopole_loc.shape[1:] ),
                                MPI.DOUBLE ],
                              root=0 )

                if rank == 0:

                    M_monopole = fitParams_monopole[ :, 0 ]
                    F0_monopole = fitParams_monopole[ :, 1 ]

                    # Calculate r^2

                    rSq = pq.rSq_fm( M_monopole )

                    # Set last Q^2 in curve

                    if Qsq_last_monopole:

                        Qsq_last_monopole_curve = Qsq_last_monopole
        
                    else:

                        Qsq_last_monopole_curve = Qsq_GeV_avg[ iflav ][ -1 ]

                    # Calculate monopole curve
                    # curve_monopole[ b ]

                    curve_monopole, Qsq_curve \
                        = fit.calcMonopoleCurve( M_monopole, F0_monopole,
                                               Qsq_last_monopole_curve )

                    # Average over bins
                    
                    fitParams_monopole_avg \
                        = np.average( fitParams_monopole, axis=0 )
                    fitParams_monopole_err = fncs.calcError( fitParams_monopole,
                                                           binNum )
                    
                    rSq_avg = np.average( rSq, axis=0 )
                    rSq_err = fncs.calcError( rSq, binNum )

                    curve_monopole_avg = np.average( curve_monopole, axis=0 )
                    curve_monopole_err = fncs.calcError( curve_monopole, binNum )

                    # Write monopole fit parameter file for each bin
                    
                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_monopoleFitParams_per_bin"
                                           + "_{}_{}_{}params_tsink{}_psq{}"
                                           + "_{}configs_binSize{}",
                                           ff, particle, flav_str[ iflav ],
                                           paramNum_monopole,
                                           ts, pSq_fin,
                                           configNum, binSize )

                    rw.write2ValueDataFile( output_filename,
                                            fitParams_monopole[ :, 0 ],
                                            fitParams_monopole[ :, 1 ] )
                    
                    # Write average fit parameter file and r^2
                    
                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_monopoleFitParams"
                                           + "_{}_{}_{}params_tsink{}_psq{}"
                                           + "_{}configs_binSize{}",
                                           ff, particle, flav_str[ iflav ],
                                           paramNum_monopole,
                                           ts, pSq_fin,
                                           configNum, binSize )

                    rw.writeMonopoleFitParamsFile( output_filename,
                                                 fitParams_monopole_avg,
                                                 fitParams_monopole_err,
                                                 rSq_avg, rSq_err )
                
                    # Write monopole fit curve
                    
                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_monopole_curve_{}_{}"
                                           + "_{}params_tsink{}_psq{}"
                                           + "_{}configs_binSize{}",
                                           ff, particle, flav_str[ iflav ],
                                           paramNum_monopole, ts, pSq_fin,
                                           configNum, binSize )

                    rw.writeAvgDataFile_wX( output_filename,
                                            Qsq_curve,
                                            curve_monopole_avg,
                                            curve_monopole_err )

                # End first process
            # End loop over form factor
        # End loop over tsink

        # Loop over form factors

        for ff, iff in fncs.zipXandIndex( F_str ):
            
            # Fit form factors to monopole
            # fitParams_monopole_loc[ b_loc, param ]
            
            fitParams_monopole_loc, chiSq_monopole_loc \
                = fit.fitFormFactor_monopole( F_good_tsf_loc[iflav][...,iff],
                                            F_tsf_err[ iflav ][ :, iff ],
                                            Qsq_GeV_loc[ iflav ],
                                            paramNum_monopole,
                                            mpi_confs_info,
                                            Qsq_last=Qsq_last_monopole,
                                            M0=M0 )
            
            # Gather monopole fit parameters to first rank
            # fitParams_monopole[ b, param ]
                
            if rank == 0:

                fitParams_monopole = np.zeros( ( binNum, 2 ) )

            else:

                fitParams_monopole = []
                
            comm.Gatherv( fitParams_monopole_loc,
                          [ fitParams_monopole,
                            recvCount \
                            * np.prod( fitParams_monopole_loc.shape[ 1: ] ),
                            recvOffset \
                            * np.prod( fitParams_monopole_loc.shape[ 1: ] ),
                            MPI.DOUBLE ],
                          root=0 )

            if rank == 0:

                M_monopole = fitParams_monopole[ :, 0 ]
                F0_monopole = fitParams_monopole[ :, 1 ]

                # Calculate r^2

                rSq = pq.rSq_fm( M_monopole )

                # Calculate monopole curve
                # curve_monopole[ b ]

                curve_monopole, Qsq_curve \
                    = fit.calcMonopoleCurve( M_monopole, F0_monopole,
                                           Qsq_last_monopole_curve )

                # Average over bins
                    
                fitParams_monopole_avg = np.average( fitParams_monopole, axis=0 )
                fitParams_monopole_err = fncs.calcError( fitParams_monopole,
                                                       binNum )

                rSq_avg = np.average( rSq, axis=0 )
                rSq_err = fncs.calcError( rSq, binNum )

                curve_monopole_avg = np.average( curve_monopole, axis=0 )
                curve_monopole_err = fncs.calcError( curve_monopole, binNum )

                # Write monopole fit parameter file for each bin
                
                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_monopoleFitParams_per_bin_2sf"
                                       + "_{}_{}_{}params_tsink{}_{}"
                                       + "_psq{}_{}configs_binSize{}",
                                       ff, particle, flav_str[ iflav ],
                                       paramNum_monopole,
                                       tsink[ 0 ], tsink[ -1 ], pSq_fin,
                                       configNum, binSize )

                rw.write2ValueDataFile( output_filename,
                                        fitParams_monopole[ :, 0 ],
                                        fitParams_monopole[ :, 1 ] )
                    
                # Write average fit parameter file and r^2
                    
                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_monopoleFitParams_2sf"
                                       + "_{}_{}_{}params_tsink{}_{}"
                                       + "_psq{}_{}configs_binSize{}",
                                       ff, particle, flav_str[ iflav ],
                                       paramNum_monopole,
                                       tsink[ 0 ], tsink[ -1 ], pSq_fin,
                                       configNum, binSize )

                rw.writeMonopoleFitParamsFile( output_filename,
                                             fitParams_monopole_avg,
                                             fitParams_monopole_err,
                                             rSq_avg, rSq_err )

                # Write monopole fit curve
                    
                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_monopole_curve_2sf"
                                       + "_{}_{}_{}params_tsink{}_{}"
                                       + "_psq{}_{}configs_binSize{}",
                                       ff, particle, flav_str[ iflav ],
                                       paramNum_monopole,
                                       tsink[ 0 ], tsink[ -1 ], pSq_fin,
                                       configNum, binSize )

                rw.writeAvgDataFile_wX( output_filename,
                                        Qsq_curve,
                                        curve_monopole_avg,
                                        curve_monopole_err )

            # End first process
        # End loop over form factor
    # End loop over parameter number
# End loop over flavor

if formFactor == "GE_GM":

    # Broadcast F_flavCombo_avg and F_flavCombo_err
    # F_flavCombo_avg[ ts, qs, ff ]
    # F_flavCombo_err[ ts, qs, ff ]
    # F_flavCombo_tsf_avg[ qs, ff ]
    # F_flavCombo_tsf_err[ qs, ff ]

    if rank != 0:

        F_flavCombo_avg = np.zeros( F_flavCombo_loc.shape[ 1: ] )
        F_flavCombo_err = np.zeros( F_flavCombo_loc.shape[ 1: ] )

        F_flavCombo_tsf_avg \
            = np.zeros( F_flavCombo_tsf_loc.shape[ 1: ] )
        F_flavCombo_tsf_err \
            = np.zeros( F_flavCombo_tsf_loc.shape[ 1: ] )

    # End not first process
    
    comm.Bcast( F_flavCombo_avg, root=0 )
    comm.Bcast( F_flavCombo_err, root=0 )

    comm.Bcast( F_flavCombo_tsf_avg, root=0 )
    comm.Bcast( F_flavCombo_tsf_err, root=0 )

    # Flavor combination for F_H

    # Loop over number of parameters
    for paramNum_monopole in 1, 2:
        # Loop over tsink
        for ts, its in fncs.zipXandIndex( tsink ):
            # Loop over form factors
            for ff, iff in fncs.zipXandIndex( F_str ):
            
                # Fit form factors to monopole
                # fitParams_monopole_flavCombo[ b, param ]

                fitParams_monopole_loc, chiSq_monopole \
                    = fit.fitFormFactor_monopole(F_flavCombo_loc[:,its,:,iff],
                                               F_flavCombo_err[its,:,iff],
                                               Qsq_GeV_flavCombo_loc,
                                               paramNum_monopole,
                                               mpi_confs_info,
                                               Qsq_last=Qsq_last_monopole )
                
                # Gather monopole fit parameters to first rank
                # fitParams_monopole[ b, param ]
                
                if rank == 0:

                    fitParams_monopole = np.zeros( ( binNum, 2 ) )

                else:

                    fitParams_monopole = []
                
                comm.Gatherv( fitParams_monopole_loc,
                              [ fitParams_monopole,
                                recvCount \
                                * np.prod( fitParams_monopole_loc.shape[1:] ),
                                recvOffset \
                                * np.prod( fitParams_monopole_loc.shape[1:] ),
                                MPI.DOUBLE ],
                              root=0 )

                if rank == 0:
                
                    M_monopole = fitParams_monopole[ :, 0 ]
                    F0_monopole = fitParams_monopole[ :, 1 ]

                    # Write monopole fit parameter file for each bin
                    
                    # Calculate r^2

                    rSq = pq.rSq_fm( M_monopole )
                    
                    # Set last Q^2 in curve

                    if Qsq_last_monopole:

                        Qsq_last_monopole_curve = Qsq_last_monopole
        
                    else:

                        Qsq_last_monopole_curve = Qsq_GeV_flavCombo_avg[ -1 ]

                    # Calculate monopole curve
                    # curve_monopole[ b ]

                    curve_monopole, Qsq_curve \
                        = fit.calcMonopoleCurve( M_monopole, F0_monopole,
                                               Qsq_last_monopole_curve )

                    # Average over bins
                    
                    fitParams_monopole_avg \
                        = np.average( fitParams_monopole, axis=0 )
                    fitParams_monopole_err \
                        = fncs.calcError( fitParams_monopole, binNum )
                
                    rSq_avg = np.average( rSq, axis=0 )
                    rSq_err = fncs.calcError( rSq, binNum )

                    curve_monopole_avg = np.average( curve_monopole, axis=0 )
                    curve_monopole_err = fncs.calcError( curve_monopole, binNum )

                    # Write monopole fit parameter file for each bin                
                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_monopoleFitParams_per_bin_{}"
                                           + "_{}params_tsink{}_psq{}"
                                           + "_{}configs_binSize{}",
                                           ff, particle,
                                           paramNum_monopole, ts, pSq_fin,
                                           configNum, binSize )

                    rw.write2ValueDataFile( output_filename,
                                            fitParams_monopole[ :, 0 ],
                                            fitParams_monopole[ :, 1 ] )

                    # Write average fit parameter file and r^2
                    
                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_monopoleFitParams"
                                           + "_{}_{}params_tsink{}_psq{}"
                                           + "_{}configs_binSize{}",
                                           ff, particle,
                                           paramNum_monopole,
                                           ts, pSq_fin,
                                           configNum, binSize )
                    
                    rw.writeMonopoleFitParamsFile(output_filename,
                                                fitParams_monopole_avg,
                                                fitParams_monopole_err,
                                                rSq_avg,
                                                rSq_err)
                    
                    # Write monopole fit curve
                    
                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_monopole_curve_{}"
                                           + "_{}params_tsink{}_psq{}"
                                           + "_{}configs_binSize{}",
                                           ff, particle,
                                           paramNum_monopole, ts, pSq_fin,
                                           configNum, binSize )
                    
                    rw.writeAvgDataFile_wX( output_filename,
                                            Qsq_curve,
                                            curve_monopole_avg,
                                            curve_monopole_err )

                # End first rank
            # End loop over form factor
        # End loop over tsink

        # Loop over form factors
        for ff, iff in fncs.zipXandIndex( F_str ):
            
            # Fit form factors to monopole
            # fitParams_monopole[ b, param ]

            fitParams_monopole_loc, chiSq_monopole \
                = fit.fitFormFactor_monopole( F_flavCombo_tsf_loc[ ..., iff ],
                                            F_flavCombo_tsf_err[ :, iff ],
                                            Qsq_GeV_flavCombo_loc,
                                            paramNum_monopole,
                                            mpi_confs_info,
                                            Qsq_last=Qsq_last_monopole )
            
            # Gather monopole fit parameters to first rank
            # fitParams_monopole[ b, param ]
                
            if rank == 0:

                fitParams_monopole = np.zeros( ( binNum, 2 ) )

            else:

                fitParams_monopole = []
                
            comm.Gatherv( fitParams_monopole_loc,
                          [ fitParams_monopole,
                            recvCount \
                            * np.prod( fitParams_monopole_loc.shape[ 1: ] ),
                            recvOffset \
                            * np.prod( fitParams_monopole_loc.shape[ 1: ] ),
                            MPI.DOUBLE ],
                          root=0 )

            if rank == 0:

                M_monopole = fitParams_monopole[ :, 0 ]
                F0_monopole = fitParams_monopole[ :, 1 ]
                               
                # Calculate r^2
                
                rSq = pq.rSq_fm( M_monopole )

                # Calculate monopole curve
                # curve_monopole[ b ]

                curve_monopole, Qsq_curve \
                    = fit.calcMonopoleCurve( M_monopole, F0_monopole,
                                           Qsq_last_monopole_curve )

                # Average over bins
                
                fitParams_monopole_avg \
                    = np.average( fitParams_monopole, axis=0 )
                fitParams_monopole_err \
                    = fncs.calcError( fitParams_monopole, binNum )
                
                rSq_avg = np.average( rSq, axis=0 )
                rSq_err = fncs.calcError( rSq, binNum )

                curve_monopole_avg = np.average( curve_monopole, axis=0 )
                curve_monopole_err = fncs.calcError( curve_monopole, binNum )

                # Write monopole fit parameter file for each bin
                    
                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_monopoleFitParams_per_bin_2sf"
                                       + "_{}_{}params_tsink{}_{}_psq{}"
                                       + "_{}configs_binSize{}",
                                       ff, particle, paramNum_monopole,
                                       tsink[ 0 ], tsink[ -1 ],
                                       pSq_fin, configNum, binSize )

                rw.write2ValueDataFile( output_filename,
                                        fitParams_monopole[ :, 0 ],
                                        fitParams_monopole[ :, 1 ] )

                # Write average fit parameter file and r^2
                    
                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_monopoleFitParams_2sf"
                                       + "_{}_{}params_tsink{}_{}_psq{}"
                                       + "_{}configs_binSize{}",
                                       ff, particle,
                                       paramNum_monopole,
                                       tsink[ 0 ], tsink[ -1 ],
                                       pSq_fin, configNum, binSize )
                
                rw.writeMonopoleFitParamsFile(output_filename,
                                            fitParams_monopole_avg,
                                            fitParams_monopole_err,
                                            rSq_avg,
                                            rSq_err)
                    
                # Write monopole fit curve
                    
                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_monopole_curve_2sf"
                                       + "_{}_{}params_tsink{}_{}"
                                       + "_psq{}_{}configs_binSize{}",
                                       ff, particle, paramNum_monopole,
                                       tsink[ 0 ], tsink[ -1 ],
                                       pSq_fin, configNum, binSize )

                rw.writeAvgDataFile_wX( output_filename,
                                        Qsq_curve,
                                        curve_monopole_avg,
                                        curve_monopole_err )

            # End first process
        # End loop over form factor
    # End loop over parameter number
# End if GE_GM

exit()
