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

lastQsqIndex = 11

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

momSq = args.threep_final_momentum_squared

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

    currNum = 4

projNum = len( projector )
flavNum = len( flav_str )

currNum = fncs.setCurrentNumber( formFactor, mpi_confs_info )

# Set data format for twop and threep files 
# and smear strings based on 
# particle and p^2

dataFormat_twop, dataFormat_threep, twop_boost_template \
    = fncs.setDataFormat( particle, momSq )

smear_str_list, smear_str_list_boost, smearNum, smearNum_boost \
    = fncs.setSmearString( particle, momSq )

# CJL: I'll need to change something for boosted frame to use right smear

# Momentum transfer list

Q, QNum, Qsq, QsqNum, Qsq_start, Qsq_end, Qsq_where \
    = [ [ [] for smr in smear_str_list ] for qq in range( 7 ) ]

for smr, ismr in zip( smear_str_list, range( smearNum ) ):

    twop_template_smr = twop_template[ ismr ].format( smr )

    Q[ ismr ], QNum[ ismr ], Qsq[ ismr ], QsqNum[ ismr ], \
    Qsq_start[ ismr ], Qsq_end[ ismr ], Qsq_where[ ismr ] \
    = rw.readMomentumTransferList( twopDir[ ismr ],
                                   twop_template_smr,
                                   [configList[ 0 ]], particle, 
                                   srcNum, momSq, dataFormat_twop[ ismr ],
                                   args.momentum_transfer_list, 
                                   mpi_confs_info )
    
    Qsq[ ismr ] = Qsq[ ismr ][ :lastQsqIndex ]
    QsqNum[ ismr ] = len( Qsq[ ismr ] )
    Qsq_start[ ismr ] = Qsq_start[ ismr ][ :lastQsqIndex ]
    Qsq_end[ ismr ] = Qsq_end[ ismr ][ :lastQsqIndex ]
    Qsq_where[ ismr ] = Qsq_where[ ismr ][ :Qsq_end[ ismr ][ -1 ] + 1 ]

# Make sure all files have same number of Q and Q^2

for qsqN in QNum:
    if qsqN != QNum[ 0 ]:

        mpi_fncs.mpiPrintError( "Error (formFactors.py): " \
                           + "two-point function files " \
                           + "do not have the same number " \
                           + "of Q. Number of Q's are: " \
                           + ",".join( QNum ) + ".",
                           mpi_confs_info )

for qsqN in QsqNum:
    if qsqN != QsqNum[ 0 ]:

        mpi_fncs.mpiPrintError( "Error (formFactors.py): " \
                           + "two-point function files " \
                           + "do not have the same number " \
                           + "of Q^2. Number of Q^2's are: " \
                           + ",".join( QsqNum ) + ".",
                           mpi_confs_info )

QNum = QNum[ 0 ]
QsqNum = QsqNum[ 0 ]

Q = np.array( Q )
Qsq = np.array( Qsq )
Qsq_start = np.array( Qsq_start )
Qsq_end = np.array( Qsq_end )
Qsq_where = np.array( Qsq_where )

# Read final momentum list

finalMomList = rw.readMomentaList( twopDir[ 0 ],
                                   twop_template[ 0 ],
                                   configList_loc[ 0 ],
                                   particle,
                                   srcNum, momSq,
                                   dataFormat_threep,
                                   mpi_confs_info )

finalMomNum = len( finalMomList )

############################
# Read Two-point Functions #
############################

# Zero momentum two-point functions
# twop[ smr, c, Q, t ]

twop_Q = [ [] for smr in smear_str_list ]

for smr, ismr in zip( smear_str_list, range( smearNum ) ):

    twop_template_smr = twop_template[ ismr ].format( smr )

    twop_Q[ ismr ] = rw.readTwopFile( twopDir[ ismr ],
                                      twop_template_smr,
                                      configList_loc, 
                                      configNum, Q[ ismr ], Qsq[ ismr ],
                                      Qsq_start[ ismr ], Qsq_end[ ismr ], 
                                      particle, srcNum, momSq,
                                      dataFormat_twop[ ismr ],
                                      mpi_confs_info )

# End loop over smears

twop_Q = np.array( twop_Q )

# Time dimension length

T = twop_Q.shape[ -1 ]

# Time dimension length after fold

T_fold = T // 2 + 1

rangeEnd = T // 2 - 1

# Move configurations to first dimension
# twop[ smr, c, Q, t ] -> twop[ c, smr, Q, t ]

twop_Q = np.moveaxis( twop_Q, 1, 0 )


##########################################
# Jackknife and fold two-point functions #
##########################################


if binNum_loc:

    # twop_jk_loc[ b_loc, smr, Q, t ]

    twop_jk_loc = fncs.jackknifeBinSubset( twop_Q,
                                           binSize,
                                           binList_loc )

    # twop_jk_loc[ b_loc, smr, Q, t ]

    twop_fold_loc = fncs.fold( twop_jk_loc )


    #############################################
    # Average two-point functions over momentum #
    #############################################

    
    # twop_loc[ b_loc, smr, Qsq, t ]

    twop_loc = np.zeros( ( binNum_loc, smearNum, QsqNum, T_fold ) )

    for ismr in range( smearNum ):

        twop_loc[ :, ismr, :, : ] \
            = fncs.averageOverQsq( twop_fold_loc[ :, ismr, :, : ],
                                   Qsq_start[ ismr ],
                                   Qsq_end[ ismr ] )
        

    ####################
    # Effective Energy #
    ####################


    # effEnergy_loc[ b_loc, smr, Qsq, t ]

    effEnergy_loc = pq.mEffFromSymTwop( twop_loc )

else:

    twop_fold_loc = np.array( [] )
    twop_loc = np.array( [] )
    effEnergy_loc = np.array( [] )


####################################################
# Gather two-point functions  and effective energy #
####################################################


# twop[ b, smr, Qsq, t ]
# effEnergy[ b, smr, Qsq, t ]

twop = np.zeros( ( binNum, smearNum, QsqNum, T_fold ) )
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

if rank == 0:

    iQsq_test = 1

    twop_test_avg = np.average( twop[:,:,iQsq_test,:], axis=0 )
    twop_test_err = fncs.calcError( twop[:,:,iQsq_test,:],
                                    binNum, axis=0 )

    for smr, ismr in zip( smear_str_list, range( smearNum ) ):

        filename_test_avg = rw.makeFilename( output_template,
                                             "twop_test_avg_{}_Qsq{}_" \
                                             + "{}configs_binSize{}",
                                             particle, Qsq[ ismr, iQsq_test ],
                                             configNum, binSize )

        rw.writeAvgDataFile( filename_test_avg,
                             twop_test_avg[ ismr ],
                             twop_test_err[ ismr ] )


###########################
# Fit two-point functions #
###########################


c0 = np.zeros( ( smearNum, binNum, QsqNum ) )
E0 = np.zeros( ( smearNum, binNum, QsqNum ) )
mEff_plat = np.zeros( ( smearNum, binNum ) )

for ismr in range( smearNum ):

    # Calculate the plateau fit of the ground state effective mass
    # to use in the dispersion relation
    
    mEff_plat[ ismr ], chiSq \
        = fit.fitPlateau_parallel( effEnergy[ :, ismr, 0, : ],
                                   plat_fitStart, rangeEnd, 
                                   mpi_confs_info )

    # Fit the effective energy or two-point functions 
    
    for iq in range( QsqNum ):
        """
        try:
            
            fitResults \
                = fit.effEnergyTwopFit( effEnergy[ :, ismr, iq, : ],
                                        twop[ :, ismr, iq, : ], rangeEnd,
                                        Qsq[ ismr, iq ], L, tsf,
                                        mpi_confs_info,
                                        plat_t_low_range=[plat_fitStart],
                                        tsf_t_low_range=[tsf_fitStart],
                                        checkFit=checkFit,
                                        fitType="twop" )
        
        except fit.lqcdjk_BadFitError as error: # Bad twop fit
    
            mpi_fncs.mpiPrint( error, mpi_confs_info )
            mpi_fncs.mpiPrint( " Will try fit on effective mass.", 
                               mpi_confs_info )

            try:
        
                fitResults \
                    = fit.effEnergyTwopFit( effEnergy[ :, ismr, iq, : ], 
                                            twop[ :, ismr, iq, : ], rangeEnd,
                                            Qsq[ ismr, iq ], L, tsf,
                                            mpi_confs_info,
                                            plat_t_low_range=[plat_fitStart],
                                            tsf_t_low_range=[tsf_fitStart],
                                            checkFit=checkFit,
                                            fitType="effEnergy" )

            except fit.lqcdjk_BadFitError as error: # Bad effEnergy fit

                mpi_fncs.mpiPrintError( "ERROR (lqcdjk_fitting.mEffTwopFit):"
                                      + str( error ), mpi_confs_info )

            # End bad effEnergy fit
        # End bad twop fit
        """
        fitParams, chiSq \
            = fit.twoStateFit_twop_dispersionRelation( twop[ :, ismr, iq, : ],
                                                       tsf_fitStart, rangeEnd, 
                                                       mEff_plat[ ismr ],
                                                       Qsq[ ismr, iq ], L,
                                                       mpi_confs_info )

        """
        fitParams_avg = np.average( fitParams, axis=0 )
        fitParams_err = fncs.calcError( fitParams, binNum )

        mpi_fncs.mpiPrint(Qsq[ismr,iq],mpi_confs_info)
        mpi_fncs.mpiPrint((fitParams_avg,fitParams_err),mpi_confs_info)

        if ismr == 0 and iq == 0:

            # Get fit results for ismr, iq = 0

            chiSq = fitResults[ 1 ]
            mEff_plat = fitResults[ 2 ]
            rangeStart = fitResults[ 3 ]
            plat_rangeStart = fitResults[ 4 ]
            fitType = fitResults[ 5 ]

            if tsf:

                if fitType == "twop":

                    E0_mEff = fitParams[ :, 2 ]
                    E1_mEff = fitParams[ :, 3 ]
            
                elif fitType == "effEnergy":

                    #c = fitParams[ :, 0 ]
                    E0_mEff = fitParams[ :, 1 ]
                    E1_mEff = fitParams[ :, 2 ]
        
            else: # One-state fit

                #c = fitParams[ :, 0 ]
                E0_mEff = fitParams[ :, 1 ]
        
            # End if one-state fit
        # End if ismr, iq == 0

        # Get two-point fit parameters

        if tsf:

            if fitType == "twop":

                c0[ ismr, :, iq ] = fitParams[ :, 0 ]
                E0[ ismr, :, iq ] = fitParams[ :, 2 ]

            elif fitType == "effEnergy":

                c0[ ismr, :, iq ] = fitParams[ :, 0 ]
                E0[ ismr, :, iq ] = fitParams[ :, 1 ]

        else: # One-state fit
            
            c0[ ismr, :, iq ] = fitParams[ :, 0 ]
            E0[ ismr, :, iq ] = fitParams[ :, 1 ]
        """
        c0[ ismr, :, iq ] = fitParams[ :, 0 ]

        mpi_fncs.mpiPrint( "Fit two-point functions at " \
                           + "Q^2={}".format( Qsq[ ismr, iq ] ),
                           mpi_confs_info )

    # End loop over Q^2
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
    """
    E0_mEff_avg = np.average( E0_mEff, axis=0 )
    E0_mEff_err = fncs.calcError( E0_mEff, binNum )

    if tsf:

        E1_mEff_avg = np.average( E1_mEff, axis=0 )
        E1_mEff_err = fncs.calcError( E1_mEff, binNum )
    
    chiSq_avg = np.average( chiSq, axis=0 )
    chiSq_err = fncs.calcError( chiSq, binNum )
    """
    # Write output files

    avgOutputFilename = rw.makeFilename( output_template, 
                                         "mEff_avg_{}" \
                                         + "_{}configs_binSize{}",
                                         particle,
                                         configNum, binSize )
    rw.writeAvgDataFile( avgOutputFilename, mEff_avg, mEff_err )

    #mEff_fit_str = "2s" + str( rangeStart ) \
    #              + ".2e" + str( rangeEnd )
    mEff_plat_str = "2s" + str( plat_fitStart ) \
                    + ".2e" + str( rangeEnd )
    """
    mEff_fit_outputFilename = rw.makeFilename( output_template,
                                               "mEff_{}_{}"
                                               + "_{}_{}" \
                                               + "_{}configs" \
                                               + "_binSize{}",
                                               fitType,
                                               "2sf" if tsf else "1sf",
                                               particle, 
                                               mEff_fit_str,
                                               configNum, 
                                               binSize )

    rw.writeFitDataFile( mEff_fit_outputFilename, E0_mEff_avg,
                         E0_mEff_err, rangeStart, rangeEnd )
        
    chiSqOutputFilename = rw.makeFilename( output_template,
                                           "mEff_{}_{}_chiSq" \
                                           + "_{}_{}" \
                                           + "_{}configs" \
                                           + "_binSize{}",
                                           fitType,
                                           "2sf" if tsf else "1sf",
                                           particle, 
                                           mEff_fit_str,
                                           configNum, binSize )

    rw.writeFitDataFile( chiSqOutputFilename, chiSq_avg,
                         chiSq_err, rangeStart, rangeEnd )
    """
    mEff_outputFilename = rw.makeFilename( output_template,
                                           "mEff_plat_{}_{}" \
                                           + "_{}configs_binSize{}",
                                           particle, mEff_plat_str, 
                                           configNum, binSize )

    rw.writeFitDataFile( mEff_outputFilename, mEff_plat_avg,
                         mEff_plat_err, plat_fitStart, rangeEnd )

# End first process

if particle == "pion" or particle == "nucleon":

    ismr_flav = [ 0 ]

else:

    if momSq == 0:

        ismr_flav = [ 1, 2 ]

    else:

        ismr_flav = [ 0, 1 ]


########################################
# Get Q list for three-point functions #
########################################


Q_threep= [ [ [] for p in finalMomList ]
            for ts in tsink ]

# Loop over tsink
for ts, its in zip( tsink, range( tsinkNum ) ) :
    # Loop over final momenta
    for p, ip in zip( finalMomList, range( finalMomNum ) ):

        if dataFormat_threep == "gpu":            

            threep_template = "{}{}{}".format( threep_tokens[0],
                                               particle,
                                               threep_tokens[1] )

        elif dataFormat_threep == "cpu":

            threep_template = "{:}{:}{:}{:+}_{:+}_{:+}.{:}.h5"
            threep_template = threep_template.format( threep_tokens[ 0 ],
                                                      ts,
                                                      threep_tokens[ 1 ],
                                                      p[ 0 ], p[ 1 ],
                                                      p[ 2 ], "up" )
        
        Q_threep[ its ][ ip ], \
            QNum_threep, \
            Qsq_threep, \
            QsqNum_threep, \
            Qsq_start_threep, \
            Qsq_end_threep, \
            Qsq_where_threep \
            = rw.readMomentumTransferList( threepDir, threep_template, 
                                          [configList[ 0 ]], particle, 
                                          srcNum, momSq, dataFormat_threep,
                                          args.momentum_transfer_list, 
                                          mpi_confs_info )

        if not np.array_equal( Q_threep[ its ][ ip ],
                               Q_threep[ 0 ][ 0 ] ):

            mpi_fncs.mpiPrintError( "Error (formFactors.py): " \
                               + "list of Q in three-point " \
                               + "funtion files are not consistant " \
                               + "across tsink or final momentum.",
                               mpi_confs_info )

Q_threep = Q_threep[ 0 ][ 0 ]


##############################
# Read three-point functions #
##############################


# threep_jk[ ts, p, b, flav, Q, proj*curr, t ]
threep_jk = np.zeros( ( tsinkNum, finalMomNum,
                        binNum, flavNum,
                        QNum_threep,
                        projNum * currNum,
                        T ) )

# Loop over tsink
for ts, its in zip( tsink, range( tsinkNum ) ) :
    # Loop over final momenta
    for p, ip in zip( finalMomList, range( finalMomNum ) ):

        # threep[ conf, flav, Q, proj*curr, t ]

        threep = rw.readFormFactorFile( threepDir, threep_tokens,
                                        formFactor, srcNum,
                                        Qsq, QNum, ts, projector,
                                        p, T, particle,
                                        dataFormat_threep,
                                        mpi_confs_info )

        ###################################
        # Jackkinfe three-point functions #
        ###################################

        # If bin on this process
        if binNum_loc:

            # Jackknife
            # threep_jk_loc[ b, flav, Q, proj*curr, t ]

            threep_jk_loc \
                = fncs.jackknifeBinSubset( threep,
                                           binSize,
                                           binList_loc )

        # End if bin on local process

        comm.Allgatherv( threep_jk_loc,
                         [ threep_jk[ its, ip ],
                           recvCount \
                           * np.prod( threep_jk_loc.shape[ 1: ] ),
                           recvOffset \
                           * np.prod( threep_jk_loc.shape[ 1: ] ),
                           MPI.DOUBLE ] )
            
    # End loop over final momenta
# End loop over tsink

Qsq_threep = Qsq_threep[ :lastQsqIndex ]
QsqNum_threep = len( Qsq_threep )
Qsq_start_threep = Qsq_start_threep[ :lastQsqIndex ]
Qsq_end_threep = Qsq_end_threep[ :lastQsqIndex ]
Qsq_where_threep = Qsq_where_threep[ :Qsq_end_threep[ -1 ] + 1 ]

Q_threep = Q_threep[ :Qsq_end_threep[ -1 ] + 1 ]
QNum_threep = len( Q_threep )

threep_jk = threep_jk[ :, :, :, :, :Qsq_end_threep[ -1 ] + 1, :, : ]

if rank == 0:

    for flav, iflav in zip( flav_str, range(flavNum) ):

        threep_Q_test_avg = np.average( threep_jk[0,0,:,iflav,
                                                  Qsq_start_threep[iQsq_test]\
                                                  :Qsq_end_threep[iQsq_test]
                                                  +1,0,:],
                                        axis=0 )
        threep_Q_test_err = fncs.calcError( threep_jk[0,0,:,iflav,
                                                      Qsq_start_threep[iQsq_test]\
                                                      :Qsq_end_threep[iQsq_test]+1,0,:],
                                            binNum )
        
        filename_test_avg = rw.makeFilename( output_template,
                                             "threep_test_avg_{}_{}_Qsq{}_"
                                            + "{}configs_binSize{}",
                                             particle, flav,
                                             Qsq_threep[ iQsq_test ],
                                             configNum, binSize )
        
        Q_test = Q_threep[Qsq_start_threep[iQsq_test] \
                          : Qsq_end_threep[iQsq_test]+1]
        
        Q_str = [ [] for q in Q_test ]

        for q, iq in zip( Q_test, range( len(Q_test) ) ):
            
            Q_str[ iq ] = " ".join( "{:>2}".format( qq ) for qq in q )

            rw.writeAvgFormFactorFile( filename_test_avg,
                                       Q_str,
                                       threep_Q_test_avg,
                                       threep_Q_test_err )


####################
# Calculate ratios #
####################


# ratio[ ts, p, b, flav, Q, proj*curr, t ]

#ratio = np.zeros( threep_jk.shape )

if rank == 0:

    Qsq_GeV = pq.convertQsqToGeV( Qsq_threep, mEff_plat_avg, a, L )

    # Loop over tsink
    for ts, its in zip( tsink, range( tsinkNum ) ):
        # Loop over final momenta
        for ip in range( finalMomNum ):
            # Loop over flavor
            for iflav in range( flavNum ):

                # ratio[ b, Q, proj*curr, t ]

                if particle == "nucleon":

                    ratio \
                        = pq.calcRatio_Q( threep_jk[ its, ip, :, iflav ],
                                          twop_jk[ :, ismr_flav[ ismr ],
                                                   :, : ],
                                          ts )
                    
                else: # particle is meson

                    ratio \
                        =pq.calcRatio_Q_twopFit(threep_jk[ its, ip,
                                                           :, iflav ],
                                                c0[ ismr_flav[iflav] ],
                                                mEff_plat[ ismr_flav[iflav] ],
                                                ts, Qsq_threep, L,
                                                Qsq_where[ismr_flav[iflav]],
                                                mpi_confs_info)
                
                # End if meson
                
                #print(Qsq_where_threep[np.where(np.isnan(ratio))[1][0]])

                ratio_Q_test_avg = np.average( ratio[:,Qsq_start_threep[iQsq_test]\
                                                     :Qsq_end_threep[iQsq_test]+1,0,:],
                                               axis=0 )
                ratio_Q_test_err = fncs.calcError( ratio[:,Qsq_start_threep[iQsq_test]\
                                                         :Qsq_end_threep[iQsq_test]+1,0,:],
                                                   binNum )

                filename_test_avg = rw.makeFilename( output_template,
                                                     "ratio_test_avg_{}_{}"
                                                     + "_Qsq{}_"
                                                     + "{}configs_binSize{}",
                                                     particle, flav_str[iflav],
                                                     Qsq_threep[ iQsq_test ],
                                                     configNum, binSize )

                Q_test = Q_threep[Qsq_start_threep[iQsq_test] \
                                 : Qsq_end_threep[iQsq_test]+1]

                Q_str = [ [] for q in Q_test ]

                for q, iq in zip( Q_test, range( len(Q_test) ) ):

                    Q_str[ iq ] = " ".join( "{:>2}".format( qq ) for qq in q )

                rw.writeAvgFormFactorFile( filename_test_avg,
                                           Q_str,
                                           ratio_Q_test_avg,
                                           ratio_Q_test_err )

                ratio_err = fncs.calcError( ratio, binNum )

                # ratio_fit[ b, Q, ratio ]
                
                ratio_fit = fit.fitFormFactor( ratio,
                                               ratio_err,
                                               ts, 2 )

                # ratio_fit_err[ Q, ratio ]

                ratio_fit_err = fncs.calcError( ratio_fit, binNum )        
            
                ###############################
                # Calculate kinematic factors #
                ###############################

                # kineFacter[ b, Q, r, [ GE, GM ] ]

                if formFactor == "GE_GM":

                    kineFactor = pq.kineFactor_GE_GM( ratio_fit_err,
                                                      particle,
                                                      flav_str[ iflav ],
                                                      mEff_plat[ 0 ],
                                                      Q_threep, L )
                
                GE = np.zeros( ( QsqNum, binNum ) )
                GM = np.zeros( ( QsqNum, binNum ) )
                """
                kineFactor_curr = [ kineFactor[ ..., 0, : ], 
                                    kineFactor[ ..., 1:, : ] ]

                ratio_fit_curr = [ ratio_fit[ ..., 0 ],
                                   ratio_fit[ ..., 1: ] ]

                ratio_fit_curr_err = [ ratio_fit_err[ ..., 0 ],
                                       ratio_fit_err[ ..., 1: ] ]

                curr_str = [ "g0", "gi" ]
                currNum_curr = [ 1, 3 ]

                for ic in range( 2 ):
                """
                gE = np.zeros( ( QsqNum, binNum ) )
                gM = np.zeros( ( QsqNum, binNum ) )
                
                for qsq in range( QsqNum ):
                    """
                    kineFactor_Qsq \
                        = kineFactor_curr[ic][:,
                                              Qsq_start_threep[ qsq ]
                                              :Qsq_end_threep[ qsq ] 
                                              + 1 ].reshape(binNum,
                                                            (Qsq_start_threep[qsq]
                                                             - Qsq_end_threep[qsq]
                                                             + 1 )
                                                            * projNum
                                                            * currNum_curr[ ic ],
                                                            2 )

                        if ic == 0:

                            gE[ qsq ] \
                                = np.average( ratio_fit_curr[ ic ][ :,
                                                                    Qsq_start_threep[qsq] 
                                                                    : Qsq_end_threep[qsq]
                                                                    + 1 ]
                                              / kineFactor_Qsq[ :, :, 0 ],
                                              axis=-1 )
                            #/ ratio_fit_curr_err[ ic ][ Qsq_start_threep[qsq] 
                            #: Qsq_end_threep[qsq]
                            #+ 1 ],

                            gM[ qsq ] \
                                = np.average( ratio_fit_curr[ ic ][ :,
                                                                    Qsq_start_threep[qsq] 
                                                                    : Qsq_end_threep[qsq]
                                                                    + 1 ]
                                              / kineFactor_Qsq[ :, :, 1 ]
                                              / ratio_fit_curr_err[ ic ][ Qsq_start_threep[qsq] 
                                                                          : Qsq_end_threep[qsq]
                                                                          + 1 ],
                                              axis=-1 )
                    
                        else:

                            # Loop over Q for Q^2
                            for iq in range(Qsq_start_threep[qsq],Qsq_end_threep[qsq]+1):
                            
                            #print(Q_threep[iq])
                            #print(kineFactor[0,iq,:,0])

                            if ic == 0:
                                
                                averageSum += ratio_fit[:,iq,0] \
                                              / kineFactor[:,iq,0,0]                                        
                        
                                averageFactor += 1

                            else:

                                # Loop over ratios
                                #for ir in range( currNum_curr[ ic ] ):
                                for ir in range( 3 ):

                                    if Q_threep[iq,ir] != 0:
                            
                                        averageSum += ratio_fit_curr[ic][:,iq,ir] \
                                        / kineFactor[:,iq,ir+1,0]                                        
                                        
                                        averageFactor += 1
                            
                                        averageSum += ratio_fit[:,iq,ir+1] \
                                                      / kineFactor[:,iq,ir+1,0]                                        
                                
                                        averageFactor += 1

                                    # End if non-zero momentum
                                # End loop over ratio
                            # End loop over Q

                            #print(Q_threep[Qsq_start_threep[qsq]:Qsq_end_threep[qsq]+1])
                            #print(averageFactor)

                            if averageFactor != 0:

                                gE[ qsq] = averageSum / averageFactor
                        
                            else:

                                gE[ qsq ] = 0.0
                    """
                    ###############
                    # Perform SVD #
                    ############### 

                    kineFactor_Qsq \
                        = kineFactor[ :,
                                      Qsq_start_threep[ qsq ]
                                      :Qsq_end_threep[ qsq ] + 1,
                                      ... ].reshape( binNum,
                                                     (Qsq_start_threep[qsq]
                                                      - Qsq_end_threep[qsq]
                                                      + 1 )
                                                     * projNum
                                                     * currNum, 2 )
                    """
                            if ic == 0:

                                gE[ qsq ] \
                                    = np.average( ratio_fit_curr[ ic ][ :,
                                                                        Qsq_start_threep[qsq] 
                                                                        : Qsq_end_threep[qsq]
                                                                    + 1 ]
                                                  / kineFactor_Qsq[ :, :, 0 ],
                                                  axis=-1 )
                                #/ ratio_fit_curr_err[ ic ][ Qsq_start_threep[qsq] 
                            #: Qsq_end_threep[qsq]
                            #+ 1 ],

                            gM[ qsq ] \
                                = np.average( ratio_fit_curr[ ic ][ :,
                                                                    Qsq_start_threep[qsq] 
                                                                    : Qsq_end_threep[qsq]
                                                                    + 1 ]
                                              / kineFactor_Qsq[ :, :, 1 ]
                                              / ratio_fit_curr_err[ ic ][ Qsq_start_threep[qsq] 
                                                                          : Qsq_end_threep[qsq]
                                                                          + 1 ],
                                              axis=-1 )
                    
                            else:
                    
                            kineFactor_Qsq \
                                = kineFactor_curr[ic][:,
                                                      Qsq_start_threep[ qsq ]
                                                      :Qsq_end_threep[ qsq ] 
                                                      + 1 ].reshape(binNum,
                                                                    (Qsq_start_threep[qsq]
                                                                     - Qsq_end_threep[qsq]
                                                                     + 1 )
                                                                    * projNum
                                                                    * currNum_curr[ ic ],
                                                                    2 )
                    """
                    u, s, vT = np.linalg.svd( kineFactor_Qsq,
                                              full_matrices=False )
                            
                    ##############################
                    # Calculate ( v s^-1 u^T )^T #
                    ##############################
                            
                    uT = np.transpose( u, ( 0, 2, 1 ) )
                    v = np.transpose( vT, ( 0, 2, 1 ) )
                    
                    smat = np.zeros( ( u.shape[-1], vT.shape[-2] ) )
                    smat_inv = np.zeros( ( binNum, ) \
                                         + np.transpose( smat ).shape )
                    
                    for b in range( binNum ):

                        smat[ :vT.shape[ -2 ], \
                              :vT.shape[ -2 ] ] = np.diag( s[ b ] )
                                
                        smat_inv[ b ] = np.linalg.pinv( smat )
                        
                    # End loop over bins

                    # decomp[ b, Q, ratio, [ GE, GM ] ]
                        
                    decomp = np.transpose( v @ smat_inv @ uT,
                                           ( 0, 2, 1 ) )

                    decomp = decomp.reshape( binNum,
                                             Qsq_end_threep[ qsq ] \
                                             - Qsq_start_threep[ qsq ] \
                                             + 1,
                                             projNum * currNum, 2 )
                    """
                            decomp = decomp.reshape( binNum,
                                                     Qsq_end_threep[ qsq ] \
                                                     - Qsq_start_threep[ qsq ] \
                                                     + 1,
                                                     projNum * currNum_curr[ ic ],
                                                     2 )

                            gE[ qsq ], gM[ qsq ] \
                                = pq.calc_gE_gM( decomp,
                                                 ratio_fit_curr[ ic ],
                                                 ratio_fit_curr_err[ ic ],
                                                 Qsq_start_threep[ qsq ],
                                                 Qsq_end_threep[ qsq ] )

                    """
                    gE[ qsq ], gM[ qsq ] \
                                = pq.calc_gE_gM( decomp,
                                                 ratio_fit,
                                                 ratio_fit_err,
                                                 Qsq_start_threep[ qsq ],
                                                 Qsq_end_threep[ qsq ] )

                        # End if ic != 0
                # End loop over Q^2
                
                # Average over bins
                
                gE_avg = np.average( gE, axis=-1 )
                gE_err = fncs.calcError( gE, binNum, axis=-1 )
                    
                gM_avg = np.average( gM, axis=-1 )
                gM_err = fncs.calcError( gM, binNum, axis=-1 )
                
                ################
                # Write output #
                ################
                    
                """
                    for i in range(4):
                    
                    output_filename \
                    = rw.makeFilename( output_template,
                    "{}_{}_GE_tsink{}_g{}" \
                    + "_{}configs_binSize{}",
                    particle,
                    flav_str[iflav],
                    ts, i, configNum, binSize )

                    rw.writeAvgDataFile_wX( output_filename, Qsq_GeV,
                    gE_avg[ i ], gE_err[ i ] )
                 
                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_{}_GE_{}_tsink{}" \
                                           + "_{}configs_binSize{}",
                                           particle,
                                           flav_str[iflav],
                                           curr_str[ ic ],
                                           ts, configNum, binSize )
                    
                    rw.writeAvgDataFile_wX( output_filename, Qsq_GeV,
                                            gE_avg, gE_err )
                """
                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_{}_GE_tsink{}" \
                                       + "_{}configs_binSize{}",
                                       particle,
                                       flav_str[iflav],
                                       ts, configNum, binSize )
                
                rw.writeAvgDataFile_wX( output_filename, Qsq_GeV,
                                        gE_avg, gE_err )
                 
                if particle == "nucleon":
                        
                    output_filename \
                        = rw.makeFilename( output_template,
                                           "{}_{}_GM_tsink{}" \
                                           + "_{}configs_binSize{}",
                                           particle,
                                           flav_str[iflav],
                                           ts, configNum, binSize )
                        
                    rw.writeAvgDataFile_wX( output_filename, Qsq_GeV,
                                            gM_avg, gM_err )
                    
                # End loop over current

            # End loop over flavor
        # End loop over p
    # End loop over tsink
# End if first process

exit()
