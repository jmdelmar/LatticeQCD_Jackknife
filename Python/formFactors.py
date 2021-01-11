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

iqSq_last = 32

np.set_printoptions(threshold=sys.maxsize)

a = 0.093
L = 32.0

particle_list = fncs.particleList()

form_factor_list = fncs.formFactorList()

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Calculate form factors for " \
                              + "non-zero momentum transfer" )

parser.add_argument( "threep_dir", action='store', type=str )

parser.add_argument( "threep_tokens", action='store', \
                     type=lambda s: [str(token) for token in s.split(',')], \
                     help="Comma seperated list of filename tokens. " \
                     + "CPU: part before tsink, part before momentum " \
                     + "boost components. GPU: part before momentum " \
                     + "boost components, part after momentum boost; " \
                     + "* for configuration number." )

parser.add_argument( "twop_dir", action='store', 
                     help="Comma seperated list of twop directories"
                     + " with directory for mEff first and then flavor(s)",
                     type=lambda s: [str(item) for item in s.split(',')] )

parser.add_argument( "twop_template", action='store',
                     help="Comma seperated list of twop filename templates"
                     + " with same order as 'twop_dir'",
                     type=lambda s: [str(item) for item in s.split(',')] )

parser.add_argument( "particle", action='store', \
                     help="Particle to calculate for. " \
                     + "Should be 'pion', 'kaon', or 'nucleon'.", \
                     type=str )

parser.add_argument( 't_sink', action='store', \
                     help="Comma seperated list of t sink's", \
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "threep_final_momentum_squared", \
                     action='store', type=int )

parser.add_argument( "form_factor", action='store', type=str,
                     help="Form_Factor to calculate. Must be one of "
                     + ", ".join( form_factor_list ) )

parser.add_argument( "binSize", action='store', type=int )

parser.add_argument( "-o", "--output_template", action='store', \
                     type=str, default="./*.dat" )

parser.add_argument( "-sn", "--source_number", action='store', type=int, \
                     help="Number of sources correlators were averaged " \
                     + "over", default=16 )

parser.add_argument( "-tsf", "--two_state_fit", action='store_true',
                     help="Performs the two-state fit if supplied" )

parser.add_argument( "--tsf_fit_start", action='store', type=int,
                     help="If given, will perform two-state fit on " 
                     + "two-point functions or effective energy "
                     + "starting at given t value, otherwise, will "
                     + "use lowest t value which satisfies condition.",
                     default=1 )

parser.add_argument( "--plat_fit_start", action='store', type=int,
                     help="If given, will perform plateau fit on effective "
                     + "mass starting at given t value, otherwise, will "
                     + "use lowest t value which satisfies condition.",
                     default=11 )

#parser.add_argument( "-f", "--data_format", action='store', \
#                     help="Data format. Should be 'gpu', " \
#                     + "'cpu', or 'ASCII'.", \
#                     type=str, default="gpu" )

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

tsf_fitStart = args.tsf_fit_start
plat_fitStart = args.plat_fit_start

srcNum = args.source_number

tsf = args.two_state_fit

pSq_fin = args.threep_final_momentum_squared

formFactor = args.form_factor

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
configList = mpi_confs_info[ 'configList' ]
configList_loc = mpi_confs_info[ 'configList_loc' ]
recvCount = mpi_confs_info[ 'recvCount' ]
recvOffset = mpi_confs_info[ 'recvOffset' ]

# Check inputs

assert particle in particle_list, \
    "Error: Particle not supported. " \
    + "Supported particles: " + str( particle_list )

#assert dataFormat in format_list, \
#    "Error: Data format not supported. " \
#    + "Supported formats: " + str( format_list )

assert configNum % binSize == 0, "Number of configurations " \
    + str( configNum ) + " not evenly divided by bin size " \
    + str( binSize ) + "."

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

elif formFactor == "A20_B20":

    Z = 1.123

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
                                   srcNum, pSq_fin, dataFormat_twop[ ismr ],
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

#q = np.array( q, dtype=int )
q = -1 * np.array( q, dtype=int )
qSq = np.array( qSq )
qSq_start = np.array( qSq_start )
qSq_end = np.array( qSq_end )
qSq_where = np.array( qSq_where )

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
                            srcNum, pSq_fin,
                            dataFormat_threep,
                            mpi_confs_info )

finalMomentaNum = len( p_fin )

############################
# Read Two-point Functions #
############################

# Zero momentum two-point functions
# twop[ smr, c, q, t ]

twop_q = [ [] for smr in smear_str_list ]

for smr, ismr in zip( smear_str_list, range( smearNum ) ):

    twop_template_smr = twop_template[ ismr ].format( smr )

    twop_q[ ismr ] = rw.readTwopFile( twopDir[ ismr ],
                                      twop_template_smr,
                                      configList_loc, 
                                      configNum, q[ ismr ], qSq[ ismr ],
                                      qSq_start[ ismr ], qSq_end[ ismr ], 
                                      particle, srcNum, pSq_fin,
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


c0 = np.zeros( ( smearNum, binNum, qSqNum ) )
c1 = np.zeros( ( smearNum, binNum, qSqNum ) )
E1 = np.zeros( ( smearNum, binNum, qSqNum ) )
mEff_plat = np.zeros( ( smearNum, binNum ) )

for ismr in range( smearNum ):

    # Calculate the plateau fit of the ground state effective mass
    # to use in the dispersion relation
    # CJL: Change this to use function which compares two fits
    # and use fit start for rest
    
    mEff_plat[ ismr ], chiSq \
        = fit.fitPlateau_parallel( effEnergy[ :, ismr, 0, : ],
                                   plat_fitStart, rangeEnd, 
                                   mpi_confs_info )

    # Fit the effective energy or two-point functions 
    
    for iq in range( qSqNum ):
        fitParams, chiSq \
            = fit.twoStateFit_twop_dispersionRelation( twop[ :, ismr, iq, : ],
                                                       tsf_fitStart, rangeEnd, 
                                                       mEff_plat[ ismr ],
                                                       qSq[ ismr, iq ], L,
                                                       mpi_confs_info )

        c0[ ismr, :, iq ] = fitParams[ :, 0 ]
        c1[ ismr, :, iq ] = fitParams[ :, 1 ]
        E1[ ismr, :, iq ] = fitParams[ :, 2 ]

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

    # Write c0 for each bin

    if particle == "kaon":

        for smr, ismr in zip( smear_str_list, range( smearNum ) ):

            c0_outputFilename = rw.makeFilename( output_template,
                                                 "c0_per_bin_{}{}_psq{}" \
                                                 + "_{}configs_binSize{}",
                                                 particle, smr, pSq_fin,
                                                 configNum, binSize )
            
            rw.writeDataFile_wX( c0_outputFilename, qSq[ ismr ], 
                                 c0[ ismr ] )

    else:

        c0_outputFilename = rw.makeFilename( output_template,
                                             "c0_per_bin_{}_psq{}" \
                                             + "_{}configs_binSize{}",
                                             particle, pSq_fin,
                                             configNum, binSize )
        
        rw.writeDataFile_wX( c0_outputFilename, qSq[ 0 ], 
                             c0[ 0 ] )

# End first process

if particle == "pion" or particle == "nucleon":

    ismr_flav = [ 0 ]

else:

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
                                               srcNum, pSq_fin, dataFormat_threep,
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

    q_threep = -q_threep[ 0 ][ 0 ]

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
                                        formFactor, srcNum,
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

# threep_jk[ ts, p, b, flav, q, ratio, t ]
# -> threep_jk[ ts, flav, b, p, q, ratio, t ]

threep_jk_loc = np.moveaxis( threep_jk_loc, [ 1, 3 ], [ 3, 1 ] )

# Change sign of up part of final momentum

p_fin = [ -1 * p_fin, p_fin ]


####################
# Calculate ratios #
####################


# Calculate Q^2 = (p_f - p_i)^2 - (E_f - E_i)^2

Qsq_where = [ [] for flav in flav_str ]

# Loop over flavor
for iflav in range( flavNum ):

    # Qsq_loc[ b_loc, qs ], QsqNum, Qsq_where[ flav, qsq, p, q ]

    Qsq_loc, QsqNum, Qsq_where[ iflav ] \
        = pq.calcQsq( p_fin[ iflav ], q_threep,
                      mEff_plat[ 0, binList_loc ],
                      L, mpi_confs_info )

Qsq = np.zeros( ( binNum, QsqNum ), dtype=float, order='c' )

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
                
        F = np.zeros( ( flavNum, binNum, QsqNum, 2 ),
                      dtype=float, order='c' )
        #F = np.zeros( ( flavNum, binNum, QsqNum, 4, 2 ),
        #              dtype=float, order='c' )
            
    else:

        F = np.array( [ [] for flav in flav_str ] )

    # Qsq_good[ flav, qs ]

    Qsq_good = np.full( ( flavNum, QsqNum ), False, dtype=bool )
    #Qsq_good = np.full( ( flavNum, QsqNum, 4 ), False, dtype=bool )

    # Loop over flavor
    for iflav in range( flavNum ):

        # ratio_loc[ b_loc, p, q, proj*curr, t ]
        
        if particle == "nucleon":
            
            ratio_loc \
                = pq.calcFormFactorRatio( threep_jk_loc[ its, iflav ],
                                          twop_jk[ binList_loc,
                                                   ismr_flav[ ismr ],
                                                   :, : ],
                                          ts )
                    
        else: # particle is meson

            ratio_loc \
                =pq.calcFormFactorRatio_twopFit(threep_jk_loc[ its,
                                                               iflav ],
                                                c0[ ismr_flav[iflav],
                                                    binList_loc ],
                                                mEff_plat[ ismr_flav[iflav],
                                                           binList_loc ],
                                                ts, p_fin[ iflav ],
                                                q_threep,
                                                qSq[ ismr_flav[ iflav ] ],
                                                L, mpi_confs_info)
            
        # End if meson

        # ratio[ b, p, q, proj*curr, t ]

        ratio = np.zeros( ( binNum, ) + ratio_loc.shape[ 1: ] )

        comm.Allgatherv( ratio_loc,
                         [ ratio,
                           recvCount \
                           * np.prod( ratio_loc.shape[ 1: ] ),
                           recvOffset \
                           * np.prod( ratio_loc.shape[ 1: ] ),
                           MPI.DOUBLE ] )

        # ratio_err[ p, q, ratio ]

        ratio_err = fncs.calcError( ratio, binNum )

        # ratio_fit_loc[ b_loc, p, q, ratio ]

        ratio_fit_loc = fit.fitFormFactor( ratio_loc,
                                           ratio_err,
                                           ts, 2 )

        # ratio_fit[ b, p, q, proj*curr ]

        ratio_fit = np.zeros( ( binNum, ) + ratio_fit_loc.shape[ 1: ] )

        comm.Allgatherv( ratio_fit_loc,
                         [ ratio_fit,
                           recvCount \
                           * np.prod( ratio_fit_loc.shape[ 1: ] ),
                           recvOffset \
                           * np.prod( ratio_fit_loc.shape[ 1: ] ),
                           MPI.DOUBLE ] )

        # ratio_fit_err[ p, q, ratio ]

        ratio_fit_err = fncs.calcError( ratio_fit, binNum )        


        ###############################
        # Calculate kinematic factors #
        ###############################


        # kineFacter[ b_loc, p, q, r, [ F1, F2 ] ]
        
        kineFactor_loc = pq.kineFactor( ratio_fit_err,
                                        formFactor,
                                        particle,
                                        flav_str[ iflav ],
                                        mEff_plat[ 0, binList_loc ],
                                        p_fin[ iflav ], q_threep, L,
                                        mpi_confs_info )

        chargeSign = -1.0 if particle == "kaon" \
                     and flav_str[ iflav ] == "s" else 1.0

        #mpi_fncs.mpiPrint(flav_str[iflav],mpi_confs_info)

        # F_loc[ b_loc, qs, [ F1, F2 ] ]

        F_loc, Qsq_good[ iflav ] \
            = pq.calcFormFactors_SVD( kineFactor_loc,
                                      ratio_fit,
                                      ratio_fit_err,
                                      Qsq_where[ iflav ],
                                      chargeSign,
                                      mpi_confs_info )
        
        comm.Gatherv( F_loc,
                      [ F[ iflav ],
                        recvCount \
                        * np.prod( F_loc.shape[ 1: ] ),
                        recvOffset \
                        * np.prod( F_loc.shape[ 1: ] ),
                        MPI.DOUBLE ],
                      root=0 )

        if rank == 0:
            
            curr_str = [ "g0", "gx", "gy", "gz" ]

            if True:
            #for ic in range( 4 ):

                # Get results for good Q^2
                # F[ b, qs_good, [ F1, F2 ] ]

                F_cp = F[ :, :, Qsq_good[ iflav ], : ]
                #F_cp = F[ :, :, Qsq_good[ iflav, :, ic ], ic, : ]

                Qsq_GeV = np.average( Qsq[ :,
                                                Qsq_good[ iflav, : ] ],
                                      axis=0 ) \
                    * ( 0.197 / a ) ** 2
                #Qsq_GeV = np.average( Qsq[ :,
                #                                Qsq_good[ iflav, :, ic ] ],
                #                      axis=0 ) \
                #    * ( 0.197 / a ) ** 2
        
                #F_cp = np.array( F_cp )
                #Qsq_GeV = np.array( Qsq_GeV )

                # Average over bins

                F_avg = np.average( F_cp[ iflav ], axis=0 )
                F_err = fncs.calcError( F_cp[ iflav ], binNum )

                if formFactor == "GE_GM":
                            
                    if particle == "nucleon":

                        F_str = [ "GE", "GM" ]
                    
                    else:
                
                        F_str = [ "GE" ]
                
                elif formFactor == "A20_B20":

                    F_str = [ "A20", "B20" ]

                for ff, iff in fncs.zipXandIndex( F_str ):

                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_{}_{}_tsink{}_psq{}" \
                                           + "_{}configs_binSize{}",
                                           ff, particle,
                                           flav_str[ iflav ],
                                           ts, pSq_fin,
                                           configNum, binSize )
                    """
                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_{}_{}_{}_tsink{}_psq{}" \
                                           + "_{}configs_binSize{}",
                                           ff,
                                           particle,
                                           flav_str[iflav],
                                           curr_str[ ic ],
                                           ts, pSq_fin,
                                           configNum, binSize )
                    """
                    rw.writeAvgDataFile_wX( output_filename, Qsq_GeV,
                                            F_avg[ :, iff ],
                                            F_err[ :, iff ] )
            
    # End loop over flavor
if False:
    if rank == 0:
                
        # Get results for good Q^2
        # F[ b, qs_good, [ F1, F2 ] ]

        if particle == "kaon":

            Qsq_good = Qsq_good[ 0 ] & Qsq_good[ 1 ]

            F = F[ :, :, Qsq_good, : ]

            F = 2./3. * F[ 0 ] - 1./3. * F[ 1 ]
        
        else:

            Qsq_good = Qsq_good[ 0 ]

            F = F[ :, :, Qsq_good, : ]

            F = F[ 0 ]

        # End pion

        Qsq_GeV = np.average( Qsq[ :, Qsq_good ], axis=0 ) \
                  * ( 0.197 / a ) ** 2
        
        # Average over bins
        
        F_avg = np.average( F, axis=0 )
        F_err = fncs.calcError( F, binNum )
                      

        ################
        # Write output #
        ################

                
        if formFactor == "GE_GM":
                            
            if particle == "nucleon":

                F_str = [ "GE", "GM" ]
                    
            else:
                
                F_str = [ "GE" ]
                    
        elif formFactor == "A20_B20":

            F_str = [ "A20", "B20" ]

        for ff, iff in fncs.zipXandIndex( F_str ):

            output_filename \
                = rw.makeFilename( output_template,
                                   "{}_{}_tsink{}_psq{}" \
                                   + "_{}configs_binSize{}",
                                   ff, particle,
                                   ts, pSq_fin,
                                   configNum, binSize )
            """
            output_filename \
            = rw.makeFilename( output_template,
            "{}_{}_{}_{}_tsink{}_psq{}" \
            + "_{}configs_binSize{}",
            ff,
            curr_str[ ic ],
            particle,
            flav_str[iflav],
            ts, pSq_fin,
            configNum, binSize )
            """                    
            rw.writeAvgDataFile_wX( output_filename, Qsq_GeV,
                                    F_avg[ :, iff ],
                                    F_err[ :, iff ] )
            
        # End loop over form factor
    # End first rank
# End loop over tsink

exit()
