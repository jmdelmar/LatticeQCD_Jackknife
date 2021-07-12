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

parser.add_argument( "-tsf", "--two_state_fit", action='store_true',
                     help="Performs the two-state fit on "
                     + "three-point functions if supplied" )

parser.add_argument( "--tsf_fit_start", action='store', type=int,
                     help="If given, will perform two-state fit on " 
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

# Set whether or not to check that the two-state fit on 
# the two-point functions and plateau fit on the 
# effective mass meet the condition
# | E_2sf - E_plat | = dE_plat / 2

if args.tsf_fit_start and args.plat_fit_start:

    checkFit = False

else:

    checkFit = True

srcNum = args.source_number

tsf = args.two_state_fit

pSq_fin = args.threep_final_momentum_squared

formFactor = args.form_factor

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

    Z = 0.445
    
elif formFactor == "A20_B20":

    Z = [ 1.123, 1.134 ]

elif formFactor == "A30_B30":

    Z = 1.34

elif formFactor == "A40_B40_C40":

    Z = 1.668

# Momentum transfer list

q, qNum, qSq, qSqNum, qSq_start, qSq_end, qSq_where \
    = [ [ [] for smr in smear_str_list ] for qq in range( 7 ) ]

for smr, ismr in zip( smear_str_list, range( smearNum ) ):

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

# Cut q off at iqSq_last

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
    # End loop over final momenta
# End loop over tsink

# threep_jk[ ts, p, b_loc, flav, q, ratio, t ]
# -> threep_jk[ ts, flav, b_loc, p, q, ratio, t ]

threep_jk_loc = np.moveaxis( threep_jk_loc, [ 1, 3 ], [ 3, 1 ] )

# Change sign of strange part of final momentum
# because conjugate is taken after phase

p_fin = [ p_fin, -1 * p_fin ]
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

Qsq = np.zeros( ( binNum, QsqNum ), dtype=float, order='c' )

# Gather Q^2

comm.Gatherv( Qsq_loc,
              [ Qsq,
                recvCount \
                * np.prod( Qsq_loc.shape[ 1: ] ),
                recvOffset \
                * np.prod( Qsq_loc.shape[ 1: ] ),
                MPI.DOUBLE ],
              root=0 )

# Loop over tsink
for ts, its in zip( tsink, range( tsinkNum ) ):

    # F[ flav, b, qsq, [ F1, F2 ] ]

    if rank == 0:
                
        F = np.zeros( ( flavNum, binNum,
                        QsqNum, formFactorNum ),
                      dtype=float, order='c' )
        #F = np.zeros( ( flavNum, binNum, QsqNum, 4, 2 ),
        #              dtype=float, order='c' )
            
    else:

        F = np.array( [ [] for flav in flav_str ] )

    # Qsq_where_good[ flav, qs ]

    Qsq_where_good = np.full( ( flavNum, QsqNum ), False, dtype=bool )
    #Qsq_where_good = np.full( ( flavNum, QsqNum, 4 ), False, dtype=bool )

    # Calculate ratio

    # Loop over flavor
    for iflav in range( flavNum ):

        # Smear index for this flavor

        ismr = ismr_flav[ iflav ]

        # ratio_loc[ b_loc, p, q, proj*curr, t ]
        
        if particle == "nucleon":
            
            # Calculate ratio with two-point functions

            ratio_loc \
                = pq.calcFormFactorRatio( threep_jk_loc[ its, iflav ],
                                          twop_jk[ binList_loc,
                                                   ismr,
                                                   :, : ],
                                          ts )
                    
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

        if rank == 0:


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
                                       formFactor,
                                       particle,
                                       flav_str[ iflav ],
                                       ts, p[ 0 ], p[ 1 ], p[ 2 ],
                                       configNum, binSize )

                rw.writeAvgDataFile( output_filename,
                                     ratio_avg[ ip, 0, 0, 5:10 ],
                                     ratio_err[ ip, 0, 0, 5:10 ] )

        # ratio_err[ p, q, ratio ]

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

        if rank == 0:

            # Average over bins

            ratio_fit_avg = np.average( ratio_fit, axis=0 )
            ratio_fit_err = fncs.calcError( ratio_fit, binNum )

            # Write ratio_fits for each final momentum

            for p, ip in fncs.zipXandIndex( p_fin[ iflav ] ):

                #print(ratio_fit[:,ip,0,0])
                #print(ratio_fit_avg[ip,0,0])

                output_filename \
                    = rw.makeFilename( output_template,
                                       "{:}_ratio_fit_{:}_{:}_tsink{:}" \
                                       + "_{:+}_{:+}_{:+}_Qsq0" \
                                       + "_{:}configs_binSize{:}",
                                       formFactor,
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
        
        # Calculate F as a function of ti for Q^2=1,2

        #for iq in range( 1, 3 ):

        #F_ti_loc = pq.calcFormFactors_ti( ratio_loc, kineFactor_loc,
        #                                      Qsq_where[ iflav

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

        # Expected sign of form factor to remove erroneous data

        if formFactor == "GE_GM":

            ratioSign = -1.0 if flav_str[ iflav ] == "s" else 1.0

        else:
                     
            ratioSign = 1.0

        # Calculate form factors from ratio fits
        # and kinematic factors using SVD
        # F_loc[ b_loc, qs, ff ]

        F_loc, Qsq_where_good[ iflav ], \
            = pq.calcFormFactors_SVD( kineFactor_loc,
                                      ratio_fit,
                                      ratio_fit_err,
                                      Qsq_where[ iflav ],
                                      formFactor,
                                      ratioSign,
                                      pSq_fin,
                                      mpi_confs_info )

        #F_loc = np.zeros( ( binNum_loc, QsqNum, 2 ) )

        #for iqs in range(QsqNum):
        
        #    ratio_Qsq = ratio_fit_loc[ :, Qsq_where[ iflav ][ iqs ], : ]
        
        #    kineFactor_Qsq \
        #        = kineFactor_loc[ :, Qsq_where[ iflav ][ iqs ], :, : ]

        #    F_tmp = ratio_Qsq[...,0]/kineFactor_Qsq[...,0,0]

        #    F_loc[ :, iqs, 0 ] = np.average( F_tmp, axis=1 )

        # Gather form factors

        comm.Gatherv( F_loc,
                      [ F[ iflav ],
                        recvCount \
                        * np.prod( F_loc.shape[ 1: ] ),
                        recvOffset \
                        * np.prod( F_loc.shape[ 1: ] ),
                        MPI.DOUBLE ],
                      root=0 )        

        #decomp = np.zeros( ( binNum, ) + decomp_loc.shape[ 1: ],
        #                   dtype=float, order='c' )
    
        #comm.Allgatherv( decomp_loc,
        #                 [ decomp,
        #                   recvCount \
        #                   * np.prod( decomp_loc.shape[ 1: ] ),
        #                   recvOffset \
        #                   * np.prod( decomp_loc.shape[ 1: ] ),
        #                   MPI.DOUBLE ] )

        if rank == 0:

            #curr_str = [ "g0", "gx", "gy", "gz" ]

            # This is for formatting so that we can easily change
            # to loop over the insertion currents
            if True:
                #for ic in range( 4 ):

                F_err = fncs.calcError( F[ iflav ], binNum )

                F_err = np.array( [ F_err ] * binNum )
                F_err = F_err.reshape( ( binNum, )
                                       + F.shape[ 2: ] )

                # Determine which Q^2 are good, i.e., 
                # the error is < 25%

                #CJL:HERE

                errorThreshold = 0.6 if formFactor == "FS" else 0.25

                # Loop over Q^2 index
                for iqs in range( QsqNum ):

                    if Qsq_where_good[ iflav, iqs ]:

                        if np.any( F_err[ :, iqs, 0 ]
                                   / np.abs( F[ iflav, :, iqs, 0 ] )
                                   > errorThreshold ):
                            
                            #| ( np.abs( F[ iflav, :, iqs, 0 ] )
                            #> 1.5 ) 
                            Qsq_where_good[ iflav, iqs ] = False

                # End loop over Q^2

                F_good = F[ iflav ]
                F_good = F_good[ :, Qsq_where_good[ iflav ], : ]

                F_good_err = F_err[ :, Qsq_where_good[ iflav ], : ]

                if formFactor == "BT10":

                    F_good_firstNonzero \
                        = F_good[ :, np.where( F_good < 0 )[ 1 ][ 0 ], 0 ]
                    
                    igood_firstNonzero_0 \
                        = np.where( F[ iflav, 0, :, 0 ]
                                    == F_good_firstNonzero[ 0 ] )[ 0 ][ 0 ]

                    for ib in range( 1, binNum ):

                        igood_firstNonzero \
                            = np.where( F[ iflav, ib, :, 0 ] 
                                        == F_good_firstNonzero[ ib ] )[ 0 ][ 0 ]

                        if igood_firstNonzero != igood_firstNonzero_0:
                            
                            warning_template = "Warning (formFactors.py):" \
                                               + "first non-zero form factor " \
                                               + "on bin {} is at index {}, " \
                                               + "which differs from the first " \
                                               + "bin where the first non-zero " \
                                               + "form factor is at index {}"

                            print( warning_template.format( ib, igood_firstNonzero,
                                                            igood_firstNonzero_0 ) )

                    # End loop over bins

                    igood_firstNonzero = igood_firstNonzero_0

                    F_good_last = F_good[ :, igood_firstNonzero, 0 ]
                    F_good_err_last = F_good_err[ :, igood_firstNonzero, 0 ]

                    Qsq_range = range( igood_firstNonzero, QsqNum )

                else:

                    F_good_last = F_good[ :, 0, 0 ]
                    F_good_err_last = F_good_err[ :, 0, 0 ]

                    Qsq_range = range( QsqNum )

                # Loop over Q^2 index
                for iqs in Qsq_range:

                    if Qsq_where_good[ iflav, iqs ]:

                        if np.any( np.abs( F[ iflav, :, iqs, 0 ] ) >
                                   np.abs( F_good_last ) + F_good_err_last ):
                            
                            #| ( np.abs( F[ iflav, :, iqs, 0 ] )
                            #> 1.5 ) 
                            Qsq_where_good[ iflav, iqs ] = False

                        else:

                            F_good_last = F[ iflav, :, iqs, 0 ]
                            F_good_err_last = F_err[ :, iqs, 0 ]

                # End loop over Q^2

                #mpi_fncs.mpiPrint(F_err[ ..., 0 ]
                #                  / np.abs( F[ iflav, ..., 0 ] ),
                #                  mpi_confs_info)
                
                # Get results for good Q^2
                # F[ flav, b, qs_good, [ F1, F2 ] ]

                F_good = F[ iflav ]
                F_good = F_good[ :, Qsq_where_good[ iflav ], : ]
                #F_good = F[ :, :, Qsq_where_good[ iflav, :, ic ], ic, : ]

                # Average over bins and convert to GeV^2

                Qsq_GeV = Qsq[ :, Qsq_where_good[ iflav ] ] \
                          * ( 0.197 / a ) ** 2

                Qsq_GeV_avg = np.average( Qsq_GeV, axis=0 )

                #decomp_avg = np.average(decomp,axis=0)

                #output_filename \
                #    = rw.makeFilename( output_template,
                #                       "decomp_A20_B20_{}_tsink{}_psq{}" \
                #                       + "_{}configs_binSize{}",
                #                       particle,
                #                       ts, pSq_fin,
                #                       configNum, binSize )            

                #rw.writeSVDOutputFile( output_filename,
                #                       decomp_avg,
                #                       q_threep )

                # Average over bins

                F_avg = np.average( F_good, axis=0 )
                F_err = fncs.calcError( F_good, binNum )

                # Write form factor output files

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

                    rw.writeDataFile_wX( output_filename, Qsq_GeV,
                                         F_good[ :, :, iff ] )

                    # Write bin averaged form factors

                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_{}_{}_tsink{}_psq{}" \
                                           + "_{}configs_binSize{}",
                                           ff, particle,
                                           flav_str[ iflav ],
                                           ts, pSq_fin,
                                           configNum, binSize )

                    #output_filename \
                    #    = rw.makeFilename( output_template,
                    #                       "{}_{}_{}_{}_tsink{}_psq{}" \
                    #                       + "_{}configs_binSize{}",
                    #                       ff,
                    #                       particle,
                    #                       flav_str[iflav],
                    #                       curr_str[ ic ],
                    #                       ts, pSq_fin,
                    #                       configNum, binSize )

                    rw.writeAvgDataFile_wX( output_filename, Qsq_GeV_avg,
                                            F_avg[ :, iff ],
                                            F_err[ :, iff ] )
            # End loop over insertion currents (for testing)
        # End first process


        #################################################
        # Fit the form factors to a dipole distribution #
        #################################################


        # Broadcast Qsq_where_good from first process

        comm.Bcast( Qsq_where_good[ iflav ], root=0 )

        # Get form factors at good Q^2

        F_good_loc = F_loc[ :, Qsq_where_good[ iflav ], : ]

        # Convert good Qsq^2 to GeV^2

        Qsq_GeV_loc = Qsq_loc[ :, Qsq_where_good[ iflav ] ] \
                      * ( 0.197 / a ) ** 2

        # Broadcast F_err
        # F_err[ qs, ff ]

        if rank != 0:

            F_err = np.zeros( F_good_loc.shape[ 1: ] )

        comm.Bcast( F_err, root=0 )

        # Loop over number of parameters

        for paramNum_dipole in 1, 2:

            # Loop over form factors

            for ff, iff in fncs.zipXandIndex( F_str ):
            
                # Fit form factors to dipole
                # fitParams_dipole_loc[ b_loc, param ]

                fitParams_dipole_loc, chiSq_dipole_loc \
                    = fit.fitFormFactor_dipole( F_good_loc[ ..., iff ],
                                                F_err[ :, iff ],
                                                Qsq_GeV_loc,
                                                paramNum_dipole,
                                                mpi_confs_info )
                               
                # Gather dipole fit parameters to first rank
                # fitParams_dipole[ b, param ]

                if rank == 0:

                    fitParams_dipole = np.zeros( ( binNum, 2 ) )

                else:

                    fitParams_dipole = []

                # End not first process

                comm.Gatherv( fitParams_dipole_loc,
                              [ fitParams_dipole,
                                recvCount \
                                * np.prod( fitParams_dipole_loc.shape[ 1: ] ),
                                recvOffset \
                                * np.prod( fitParams_dipole_loc.shape[ 1: ] ),
                                MPI.DOUBLE ],
                              root=0 )

                if rank == 0:

                    # Calculate r^2

                    rSq = 6. / fitParams_dipole[ :, 0 ] ** 2

                    # Average over bins
                    
                    fitParams_dipole_avg = np.average( fitParams_dipole, axis=0 )
                    fitParams_dipole_err = fncs.calcError( fitParams_dipole,
                                                           binNum )

                    rSq_avg = np.average( rSq, axis=0 )
                    rSq_err = fncs.calcError( rSq, binNum )

                    # Write dipole fit parameter file for each bin
                    
                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_dipoleFitParams_per_bin"
                                           + "_{}_{}_{}params_tsink{}_psq{}"
                                           + "_{}configs_binSize{}",
                                           ff, particle, flav_str[ iflav ],
                                           paramNum_dipole,
                                           ts, pSq_fin,
                                           configNum, binSize )

                    rw.write2ValueDataFile( output_filename,
                                            fitParams_dipole[ :, 0 ],
                                            fitParams_dipole[ :, 1 ] )
                    
                    # Write average fit parameter file and r^2
                    
                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_dipoleFitParams"
                                           + "_{}_{}_{}params_tsink{}_psq{}"
                                           + "_{}configs_binSize{}",
                                           ff, particle, flav_str[ iflav ],
                                           paramNum_dipole,
                                           ts, pSq_fin,
                                           configNum, binSize )

                    rw.writeDipoleFitParamsFile( output_filename,
                                                 fitParams_dipole_avg,
                                                 fitParams_dipole_err,
                                                 rSq_avg, rSq_err )
                    
                # End first process
            # End loop over form factor
        # End loop over parameter number
    # End loop over flavor
    

    ##############################
    # Flavor combination for F_H #
    ##############################


    if formFactor == "GE_GM" and rank == 0:

        # Get results for good Q^2
        # F[ flav, b, qs_good, [ F1, F2 ] ]

        if particle in [ "kaon", "nucleon" ]:

            # F_K = 2/3 F_u - 1/3 F_s
            # F_N = 2/3 F_u - 1/3 F_d

            Qsq_where_good_flavCombo = Qsq_where_good[ 0 ] & Qsq_where_good[ 1 ]

            F_good_flavCombo = F[ :, :, Qsq_where_good_flavCombo, : ]

            F_flavCombo = 2./3. * F_good_flavCombo[ 0 ] \
                          - 1./3. * F_good_flavCombo[ 1 ]
        
        else: # particle == "pion"

            # F_pi = 2/3 F_u - 1/3 F_d = F_u

            F_flavCombo = F_good

            Qsq_where_good_flavCombo = Qsq_where_good[ 0 ]

        # End pion

        # Convert Q^2 to GeV^2
        
        Qsq_GeV_flavCombo \
            = Qsq[ :, Qsq_where_good_flavCombo ] * ( 0.197 / a ) ** 2
        
        # Average over bins
        
        F_flavCombo_avg = np.average( F_flavCombo, axis=0 )
        F_flavCombo_err = fncs.calcError( F_flavCombo, binNum )
        
        Qsq_GeV_flavCombo_avg = np.average( Qsq_GeV_flavCombo, axis=0 )

        # Write output

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
                                 F_flavCombo[ :, :, iff ] )

            # Write bin-averaged orm factors

            output_filename \
                = rw.makeFilename( output_template,
                                   "{}_{}_tsink{}_psq{}" \
                                   + "_{}configs_binSize{}",
                                   ff, particle,
                                   ts, pSq_fin,
                                   configNum, binSize )

            rw.writeAvgDataFile_wX( output_filename, Qsq_GeV_flavCombo_avg,
                                    F_flavCombo_avg[ :, iff ],
                                    F_flavCombo_err[ :, iff ] )
            
        # End loop over form factor


        #################################################
        # Fit the form factors to a dipole distribution #
        #################################################


        # Loop over number of parameters

        for paramNum_dipole in 1, 2:

            # Loop over form factors

            for ff, iff in fncs.zipXandIndex( F_str ):
            
                # Fit form factors to dipole
                # fitParams_dipole_flavCombo[ b, param ]

                fitParams_dipole_flavCombo, chiSq_dipole_flavCombo \
                    = fit.fitFormFactor_dipole( F_flavCombo[ ..., iff ],
                                                F_flavCombo_err[ :, iff ],
                                                Qsq_GeV_flavCombo,
                                                paramNum_dipole,
                                                mpi_confs_info )
                               
                m_dipole = fitParams_dipole_flavCombo[ :, 0 ]
                F0_dipole = fitParams_dipole_flavCombo[ :, 1 ]

                # Write dipole fit parameter file for each bin
                    
                # Calculate r^2

                rSq_flavCombo = 6. / fitParams_dipole_flavCombo[ :, 0 ] ** 2

                # Average over bins
                    
                fitParams_dipole_flavCombo_avg \
                    = np.average( fitParams_dipole_flavCombo, axis=0 )
                fitParams_dipole_flavCombo_err \
                    = fncs.calcError( fitParams_dipole_flavCombo, binNum )
                
                rSq_flavCombo_avg = np.average( rSq_flavCombo, axis=0 )
                rSq_flavCombo_err = fncs.calcError( rSq_flavCombo, binNum )

                # Write dipole fit parameter file for each bin                

                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_dipoleFitParams_per_bin_{}"
                                       + "_{}params_tsink{}_psq{}"
                                       + "_{}configs_binSize{}",
                                       ff, particle,
                                       paramNum_dipole, ts, pSq_fin,
                                       configNum, binSize )

                rw.write2ValueDataFile( output_filename,
                                        fitParams_dipole_flavCombo[ :, 0 ],
                                        fitParams_dipole_flavCombo[ :, 1 ] )

                # Write average fit parameter file and r^2
                    
                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_dipoleFitParams"
                                       + "_{}_{}params_tsink{}_psq{}"
                                       + "_{}configs_binSize{}",
                                       ff, particle,
                                       paramNum_dipole,
                                       ts, pSq_fin,
                                       configNum, binSize )
                
                rw.writeDipoleFitParamsFile(output_filename,
                                            fitParams_dipole_flavCombo_avg,
                                            fitParams_dipole_flavCombo_err,
                                            rSq_flavCombo_avg,
                                            rSq_flavCombo_err)
                    
                # Calculate dipole curve
                # curve_dipole[ b ]

                curve_dipole, Qsq_curve \
                    = fit.calcDipoleCurve( m_dipole, F0_dipole,
                                           Qsq_GeV_flavCombo_avg[ -1 ] )

                curve_dipole_avg = np.average( curve_dipole, axis=0 )
                curve_dipole_err = fncs.calcError( curve_dipole, binNum )

                # Write dipole fit curve
                    
                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_dipole_curve_{}"
                                       + "_{}params_tsink{}_psq{}"
                                       + "_{}configs_binSize{}",
                                       ff, particle,
                                       paramNum_dipole, ts, pSq_fin,
                                       configNum, binSize )

                rw.writeAvgDataFile_wX( output_filename,
                                        Qsq_curve,
                                        curve_dipole_avg,
                                        curve_dipole_err )

            # End loop over form factor
        # End loop over parameter number
    # End if GE_GM and first rank
# End loop over tsink


#################
# Two-state fit #
#################


if tsf:

    mpi_fncs.mpiPrint( "Will perform the two-state fit", mpi_confs_info )

    # Number of three-point functions to neglect on source and sink sides

    neglect = 3

    # F_tsf[ flav, b, qsq, [ F1, F2 ] ]

    if rank == 0:

        F_tsf = np.zeros( ( flavNum, binNum,
                            QsqNum, formFactorNum ),
                          dtype=float, order='c' )
        
    else:

        F_tsf = np.array( [ [] for flav in flav_str ] )

    # Qsq_where_good_tsf[ flav, qs ]

    Qsq_where_good_tsf = np.full( ( flavNum, QsqNum ), False, dtype=bool )

    # Loop over flavors
    for iflav in range( flavNum ):

        # Smear index for this flavor

        ismr = ismr_flav[ iflav ]

        # Perform two-state fit
        # tsfParams[ b, p, q, r, [ A00, A01, A10, A11 ] ]

        if dispRel:

            # Calculate two-state fit parameters using dispersion relation
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
                                                      neglect, L, True,
                                                      mpi_confs_info)

        else:

            # Calculate two-state fit parameters using E0 from 
            # two-state fit on two-point functions

            tsfParams \
                = fit.twoStateFit_threep_momTransfer(threep_jk_loc[ :,
                                                                    iflav ],
                                                     tsink,
                                                     E0[ ismr ],
                                                     E1[ ismr ],
                                                     p_fin[ iflav ],
                                                     q_threep[ iflav ],
                                                     qSq[ ismr ],
                                                     neglect, L, False,
                                                     mpi_confs_info)
        
        # Two-state fit parameters

        a00 = tsfParams[ ..., 0 ]
        a01 = tsfParams[ ..., 1 ]
        a10 = tsfParams[ ..., 2 ]
        a11 = tsfParams[ ..., 3 ]

        # Ratio from two-state fit
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

        # Gather two-state fit form factor
        # F_tsf_loc[ b_loc, qs, [ F1, F2(, F3 ) ] ]

        F_tsf_loc, Qsq_where_good_tsf[ iflav ], \
            = pq.calcFormFactors_SVD( kineFactor_tsf_loc,
                                      ratio_tsf,
                                      ratio_tsf_err,
                                      Qsq_where[ iflav ],
                                      formFactor,
                                      ratioSign,
                                      pSq_fin,
                                      mpi_confs_info )

        comm.Gatherv( F_tsf_loc,
                      [ F_tsf[ iflav ],
                        recvCount \
                        * np.prod( F_tsf_loc.shape[ 1: ] ),
                        recvOffset \
                        * np.prod( F_tsf_loc.shape[ 1: ] ),
                        MPI.DOUBLE ],
                      root=0 )        

        if rank == 0:
          
            F_tsf_err = fncs.calcError( F_tsf[ iflav ], binNum )

            F_tsf_err = np.array( [ F_tsf_err ] * binNum )
            F_tsf_err = F_tsf_err.reshape( ( binNum, )
                                   + F_tsf.shape[ 2: ] )

            # Loop over Q^2 index
            for iqs in range( QsqNum ):
                
                if Qsq_where_good_tsf[ iflav, iqs ]:

                    if np.any( F_tsf_err[ :, iqs, 0 ]
                               / np.abs( F_tsf[ iflav, :, iqs, 0 ] )
                               > errorThreshold ):
                            
                        Qsq_where_good_tsf[ iflav, iqs ] = False

            F_good_tsf = F_tsf[ iflav ]
            F_good_tsf = F_good_tsf[ :, Qsq_where_good_tsf[ iflav ], : ]

            F_good_tsf_err = F_tsf_err[ :, Qsq_where_good[ iflav ], : ]

            if formFactor == "BT10":

                F_good_firstNonzero \
                    = F_good_tsf[ :, np.where( F_good_tsf < 0 )[ 1 ][ 0 ], 0 ]
                    
                igood_firstNonzero_0 \
                    = np.where( F_tsf[ iflav, 0, :, 0 ]
                                == F_good_firstNonzero[ 0 ] )[ 0 ][ 0 ]

                for ib in range( 1, binNum ):

                    igood_firstNonzero \
                        = np.where( F_tsf[ iflav, ib, :, 0 ] 
                                    == F_good_firstNonzero[ ib ] )[ 0 ][ 0 ]
                    
                    if igood_firstNonzero != igood_firstNonzero_0:
                            
                        warning_template = "Warning (formFactors.py):" \
                                           + "first non-zero form factor " \
                                           + "on bin {} is at index {}, " \
                                           + "which differs from the first " \
                                           + "bin where the first non-zero " \
                                           + "form factor is at index {}"
                        
                        print( warning_template.format( ib, igood_firstNonzero,
                                                        igood_firstNonzero_0 ) )

                # End loop over bins

                igood_firstNonzero = igood_firstNonzero_0

                F_good_last = F_good_tsf[ :, igood_firstNonzero, 0 ]
                F_good_err_last = F_good_tsf_err[ :, igood_firstNonzero, 0 ]
                    
                Qsq_range = range( igood_firstNonzero, QsqNum )

            else:

                F_good_last = F_good_tsf[ :, 0, 0 ]
                F_good_err_last = F_good_tsf_err[ :, 0, 0 ]

                Qsq_range = range( QsqNum )

            # Loop over Q^2 index
            for iqs in Qsq_range:
                
                if Qsq_where_good_tsf[ iflav, iqs ]:

                    if np.any( np.abs( F_tsf[ iflav, :, iqs, 0 ] ) >
                               np.abs( F_good_last ) + F_good_err_last ):
                            
                        Qsq_where_good_tsf[ iflav, iqs ] = False

                    else:

                        F_good_last = F_tsf[ iflav, :, iqs, 0 ]
                        F_good_err_last = F_tsf_err[ :, iqs, 0 ]

            # Get results for good Q^2
            # F_good_tsf[ flav, b, qs_good, [ F1, F2 ] ]

            F_good_tsf = F_tsf[ iflav ]
            F_good_tsf = F_good_tsf[ :, Qsq_where_good_tsf[ iflav ], : ]

            Qsq_GeV = Qsq[ :, Qsq_where_good_tsf[ iflav, : ] ] \
                      * ( 0.197 / a ) ** 2

            Qsq_GeV_avg = np.average( Qsq_GeV, axis=0 )

            # Average over bins

            F_tsf_avg = np.average( F_good_tsf, axis=0 )
            F_tsf_err = fncs.calcError( F_good_tsf, binNum )

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

                rw.writeDataFile_wX( output_filename, Qsq_GeV,
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

                rw.writeAvgDataFile_wX( output_filename, Qsq_GeV_avg,
                                        F_tsf_avg[ :, iff ],
                                        F_tsf_err[ :, iff ] )
    
        # End first process


        #################################################
        # Fit the form factors to a dipole distribution #
        #################################################


        # Broadcast Qsq_where_good from first process

        comm.Bcast( Qsq_where_good_tsf[ iflav ], root=0 )

        # Get form factors at good Q^2

        F_good_tsf_loc = F_tsf_loc[ :, Qsq_where_good_tsf[ iflav ], : ]

        # Convert good Qsq^2 to GeV^2

        Qsq_GeV_loc \
            = Qsq_loc[ :, Qsq_where_good_tsf[ iflav ] ] * ( 0.197 / a ) ** 2

        # Broadcast F_err
        # F_err[ qs, ff ]

        if rank != 0:

            F_tsf_err = np.zeros( F_good_tsf_loc.shape[ 1: ] )

        comm.Bcast( F_tsf_err, root=0 )

        # Loop over number of parameters

        for paramNum_dipole in 1, 2:

            # Loop over form factors

            for ff, iff in fncs.zipXandIndex( F_str ):
            
                # Fit form factors to dipole
                # fitParams_dipole_loc[ b_loc, param ]

                fitParams_dipole_loc, chiSq_dipole_loc \
                    = fit.fitFormFactor_dipole( F_good_tsf_loc[ ..., iff ],
                                                F_tsf_err[ :, iff ],
                                                Qsq_GeV_loc,
                                                paramNum_dipole,
                                                mpi_confs_info )
                               
                # Gather dipole fit parameters to first rank
                # fitParams_dipole[ b, param ]

                if rank == 0:

                    fitParams_dipole = np.zeros( ( binNum, 2 ) )

                else:

                    fitParams_dipole = []

                # End not first process

                comm.Gatherv( fitParams_dipole_loc,
                              [ fitParams_dipole,
                                recvCount \
                                * np.prod( fitParams_dipole_loc.shape[ 1: ] ),
                                recvOffset \
                                * np.prod( fitParams_dipole_loc.shape[ 1: ] ),
                                MPI.DOUBLE ],
                              root=0 )

                if rank == 0:

                    # Calculate r^2

                    rSq = 6. / fitParams_dipole[ :, 0 ] ** 2

                    # Average over bins
                    
                    fitParams_dipole_avg = np.average( fitParams_dipole, axis=0 )
                    fitParams_dipole_err = fncs.calcError( fitParams_dipole,
                                                           binNum )

                    rSq_avg = np.average( rSq, axis=0 )
                    rSq_err = fncs.calcError( rSq, binNum )

                    # Write dipole fit parameter file for each bin
                    
                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_dipoleFitParams_per_bin"
                                           + "_{}_{}_{}params_2sf_tsink{}_{}"
                                           + "_psq{}_{}configs_binSize{}",
                                           ff, particle, flav_str[ iflav ],
                                           paramNum_dipole,
                                           tsink[ 0 ], tsink[ -1 ], pSq_fin,
                                           configNum, binSize )

                    rw.write2ValueDataFile( output_filename,
                                            fitParams_dipole[ :, 0 ],
                                            fitParams_dipole[ :, 1 ] )
                    
                    # Write average fit parameter file and r^2
                    
                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_dipoleFitParams"
                                           + "_{}_{}_{}params_2sf_tsink{}_{}"
                                           + "_psq{}_{}configs_binSize{}",
                                           ff, particle, flav_str[ iflav ],
                                           paramNum_dipole,
                                           tsink[ 0 ], tsink[ -1 ], pSq_fin,
                                           configNum, binSize )

                    rw.writeDipoleFitParamsFile( output_filename,
                                                 fitParams_dipole_avg,
                                                 fitParams_dipole_err,
                                                 rSq_avg, rSq_err )

                # End first process
            # End loop over form factor
        # End loop over parameter number
    # End loop over flavor


    ##############################
    # Flavor combination for F_H #
    ##############################


    if formFactor == "GE_GM" and rank == 0:

        # Write the flavor combined form factors
                
        if particle in [ "kaon", "nucleon" ]:

            # Get results for good Q^2
            # F_tsf[ flav, b, qs_good, [ F1, F2 ] ]
           
            Qsq_where_good_flavCombo_tsf = Qsq_where_good_tsf[ 0 ] \
                    & Qsq_where_good_tsf[ 1 ]

            F_good_flavCombo_tsf = F_tsf[ :, :,
                                          Qsq_where_good_flavCombo_tsf,
                                          : ]

            # F_K = 2/3 F_u - 1/3 F_s
            # F_N = 2/3 F_u - 1/3 F_d

            F_flavCombo_tsf = 2./3. * F_good_flavCombo_tsf[ 0 ] \
                              - 1./3. * F_good_flavCombo_tsf[ 1 ]
        
        else: # particle == "pion"

            # F_pi = 2/3 F_u - 1/3 F_d = F_u
            
            F_flavCombo_tsf = F_good_tsf
           
            Qsq_where_good_flavCombo_tsf = Qsq_where_good_tsf[ 0 ]

        # End pion

        # Convert Q^2 to GeV^2

        Qsq_GeV_flavCombo \
            = Qsq[ :, Qsq_where_good_flavCombo_tsf ] * ( 0.197 / a ) ** 2
        
        # Average over bins
        
        F_flavCombo_tsf_avg = np.average( F_flavCombo_tsf, axis=0 )
        F_flavCombo_tsf_err = fncs.calcError( F_flavCombo_tsf, binNum )
                      
        Qsq_GeV_flavCombo_avg = np.average( Qsq_GeV_flavCombo, axis=0 )

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


        #################################################
        # Fit the form factors to a dipole distribution #
        #################################################


        # Loop over number of parameters

        for paramNum_dipole in 1, 2:

            # Loop over form factors

            for ff, iff in fncs.zipXandIndex( F_str ):
            
                # Fit form factors to dipole
                # fitParams_dipole_flavCombo[ b, param ]

                fitParams_dipole_flavCombo, chiSq_dipole_flavCombo \
                    = fit.fitFormFactor_dipole( F_flavCombo_tsf[ ..., iff ],
                                                F_flavCombo_tsf_err[ :,
                                                                     iff ],
                                                Qsq_GeV_flavCombo,
                                                paramNum_dipole,
                                                mpi_confs_info )
                
                m_dipole = fitParams_dipole_flavCombo[ :, 0 ]
                F0_dipole = fitParams_dipole_flavCombo[ :, 1 ]
                               
                # Calculate r^2
                
                rSq_flavCombo = 6. / fitParams_dipole_flavCombo[ :, 0 ] ** 2

                # Average over bins
                    
                fitParams_dipole_flavCombo_avg \
                    = np.average( fitParams_dipole_flavCombo, axis=0 )
                fitParams_dipole_flavCombo_err \
                    = fncs.calcError( fitParams_dipole_flavCombo, binNum )
                
                rSq_flavCombo_avg = np.average( rSq_flavCombo, axis=0 )
                rSq_flavCombo_err = fncs.calcError( rSq_flavCombo, binNum )

                # Write dipole fit parameter file for each bin
                    
                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_dipoleFitParams_per_bin"
                                       + "_{}_{}params_2sf_tsink{}_{}_psq{}"
                                       + "_{}configs_binSize{}",
                                       ff, particle, paramNum_dipole,
                                       tsink[ 0 ], tsink[ -1 ],
                                       pSq_fin, configNum, binSize )

                rw.write2ValueDataFile( output_filename,
                                        fitParams_dipole_flavCombo[ :, 0 ],
                                        fitParams_dipole_flavCombo[ :, 1 ] )

                # Write average fit parameter file and r^2
                    
                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_dipoleFitParams"
                                       + "_{}_{}params_2sf_tsink{}_{}_psq{}"
                                       + "_{}configs_binSize{}",
                                       ff, particle,
                                       paramNum_dipole,
                                       tsink[ 0 ], tsink[ -1 ],
                                       pSq_fin, configNum, binSize )
                
                rw.writeDipoleFitParamsFile(output_filename,
                                            fitParams_dipole_flavCombo_avg,
                                            fitParams_dipole_flavCombo_err,
                                            rSq_flavCombo_avg,
                                            rSq_flavCombo_err)
                    
                # Calculate dipole curve
                # curve_dipole[ b ]

                curve_dipole, Qsq_curve \
                    = fit.calcDipoleCurve( m_dipole, F0_dipole,
                                           Qsq_GeV_flavCombo_avg[ -1 ] )

                curve_dipole_avg = np.average( curve_dipole, axis=0 )
                curve_dipole_err = fncs.calcError( curve_dipole, binNum )

                # Write dipole fit curve
                    
                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_dipole_curve_{}"
                                       + "_{}params_2sf_tsink{}_{}_psq{}"
                                       + "_{}configs_binSize{}",
                                       ff, particle, paramNum_dipole,
                                       tsink[ 0 ], tsink[ -1 ],
                                       pSq_fin, configNum, binSize )

                rw.writeAvgDataFile_wX( output_filename,
                                        Qsq_curve,
                                        curve_dipole_avg,
                                        curve_dipole_err )

            # End loop over form factor
        # End loop over parameter number
    # End if GE_GM
# End if tsf


exit()
