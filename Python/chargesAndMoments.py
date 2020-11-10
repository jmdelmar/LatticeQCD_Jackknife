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

particle_list = fncs.particleList()

format_list = fncs.dataFormatList()

mellin_list = fncs.mellinMomentList()
GE_list = fncs.GEList()

ratio_list = mellin_list + GE_list

#########################
# Parse input arguments #
#########################


parser = argp.ArgumentParser( description="Calculate physical "
                              + "quantities at zero momentum "
                              + "transfer from the ratio of "
                              + "three- and two-point functions." )

parser.add_argument( "threep_dir", action='store', type=str )

parser.add_argument( "threep_tokens", action='store',
                     type=lambda s: [str(token) for token in s.split(',')],
                     help="Comma seperated list of filename tokens. "
                     + "CPU: part before tsink, part before momentum "
                     + "boost components. GPU: part before momentum "
                     + "boost components, part after momentum boost; "
                     + "* for configuration number." )

parser.add_argument( "twop_dir", action='store', 
                     help="Comma seperated list of twop directories"
                     + " with directory for mEff first and then flavor(s)",
                     type=lambda s: [str(item) for item in s.split(',')] )

parser.add_argument( "twop_template", action='store',
                     help="Comma seperated list of twop filename templates"
                     + " with same order as 'twop_dir'",
                     type=lambda s: [str(item) for item in s.split(',')] )

parser.add_argument( "particle", action='store',
                     help="Particle to calculate for. "
                     + "Should be 'pion' or 'kaon'.", type=str )

parser.add_argument( 't_sink', action='store',
                     help="Comma seperated list of t sink's",
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "mom_squared", action='store', type=int )

parser.add_argument( "ratio", action='store', type=str,
                     help="Ratio to calculate. Must be one of "
                     + ", ".join( ratio_list ) )

parser.add_argument( "binSize", action='store', type=int )

parser.add_argument( "-o", "--output_template", action='store',
                     type=str, default="./*.dat" )

parser.add_argument( "-sn", "--source_number", action='store', type=int,
                     help="Number of sources correlators were averaged "
                     + "over", default=16 )

parser.add_argument( "-tsf", "--two_state_fit", action='store_true',
                     help="Performs the two-state fit if supplied" )

parser.add_argument( "--tsf_fit_start", action='store', type=int,
                     help="If given, will perform two-state fit on " 
                     + "two-point functions or effective energy "
                     + "starting at given t value, otherwise, will "
                     + "use lowest t value which satisfies condition." )

parser.add_argument( "--plat_fit_start", action='store', type=int,
                     help="If given, will perform plateau fit on effective "
                     + "mass starting at given t value, otherwise, will "
                     + "use lowest t value which satisfies condition." )

#parser.add_argument( "-f", "--data_format", action='store',
#                     help="Data format. Should be 'gpu' or 'cpu'.",
#                     type=str, default="gpu" )

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

#dataFormat = args.data_format

momSq = args.mom_squared

whichRatio = args.ratio

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

#assert dataFormat in format_list, \
#    "Error: Data format not supported. " \
#    + "Supported particles: " + ", ".join( format_list )

assert whichRatio in ratio_list, \
    "Error: Ratio " + whichRatio + " not supported. " \
    + "Supported ratios: " + ", ".join( ratio_list )


assert particle in particle_list, \
    "Error: Particle not supported. " \
    + "Supported particles: " + ", ".join( particle_list )

# Set flavor strings.
# If pion, will not access strange.

if whichRatio in GE_list:

    flav_str = [ "" ]

elif whichRatio in mellin_list:

    if particle == "pion":

        flav_str = [ "_u" ]

    elif particle == "kaon":

        flav_str = [ "_u", "_s" ]

    elif particle == "nucleon":

        flav_str = [ "_IV", "_IS" ]

flavNum = len( flav_str )

# Set data format for twop and threep files 
# and smear strings based on 
# particle and p^2

dataFormat_twop, dataFormat_threep, twop_boost_template \
    = fncs.setDataFormat( particle, momSq )

smear_str_list, smear_str_list_boost, smearNum, smearNum_boost \
    = fncs.setSmearString( particle, momSq )

if whichRatio == "avgX":

    Z = 1.123

elif whichRatio == "avgX2":

    Z = 1.34

elif whichRatio == "avgX3":

    Z = 1.0

elif whichRatio == "GE0_local":

    Z = 0.715

elif whichRatio == "GE0_noether":

    Z = 1.0

# Read momentum list

momList = rw.readMomentaList( twopDir[ 0 ], 
                              twop_template[ 0 ].format(smear_str_list[ 0 ]),
                              configList_loc[ 0 ], particle,
                              srcNum, momSq, dataFormat_twop[ 0 ], 
                              mpi_confs_info )

momBoostNum = len( momList )


############################
# Read Two-point Functions #
############################


# Zero momentum two-point functions
# twop[ smr, c, t ]

twop = [ [] for smr in smear_str_list ]

for smr, ismr in zip( smear_str_list, range( smearNum ) ):

    twop_template_smr = twop_template[ ismr ].format( smr )

    twop[ ismr ] = rw.readTwopFile_zeroQ( twopDir[ ismr ], configList_loc, 
                                          configNum,
                                          twop_template_smr,
                                          srcNum, 0, 
                                          dataFormat_twop[ ismr ],
                                          mpi_confs_info )

twop = np.array( twop )

# Time dimension length

T = twop.shape[ -1 ]

# Time dimension length after fold

T_fold = T // 2 + 1

rangeEnd = T // 2 - 1


##########################################
# Jackknife and fold two-point functions #
##########################################


twop_fold = np.zeros( ( smearNum, binNum, T_fold ) )
mEff = np.zeros( ( smearNum, binNum, T_fold ) )

for ismr in range( smearNum ):

    if binNum_loc:

        twop_jk_loc = fncs.jackknifeBinSubset( twop[ ismr ],
                                               binSize,
                                               binList_loc )

        # twop_fold[ b, t ]

        twop_fold_loc = fncs.fold( twop_jk_loc )

        # mEff[ b, t ]

        mEff_loc = pq.mEffFromSymTwop( twop_fold_loc )

    else:

        twop_jk_loc = np.array( [] )
        twop_fold_loc = np.array( [] )
        mEff_loc = np.array( [] )

    comm.Allgatherv( twop_fold_loc, 
                     [ twop_fold[ ismr ], 
                       recvCount * T_fold,
                       recvOffset * T_fold, 
                       MPI.DOUBLE ] )
    comm.Allgatherv( mEff_loc, 
                     [ mEff[ ismr ], 
                       recvCount * T_fold, 
                       recvOffset * T_fold, 
                       MPI.DOUBLE ] )


##################
# Effective mass #
##################


fitResults = [ [] for smr in smear_str_list ]
mEff_fit = np.zeros( ( smearNum, binNum ) )
mEff_fit_avg = np.zeros( smearNum )

for ismr in range( smearNum ):

    if rank == 0 and ismr == 0:

        # mEff_avg[ t ]

        mEff_avg = np.average( mEff[ ismr ], axis=-2 )
        mEff_err = fncs.calcError( mEff[ ismr ], binNum, axis=-2 )
    
        avgOutputFilename = rw.makeFilename( output_template, 
                                             "mEff_avg_{}_psq{}" \
                                             + "_{}configs_binSize{}",
                                             particle, 0, 
                                             configNum, binSize )
        rw.writeAvgDataFile( avgOutputFilename, mEff_avg, mEff_err )

    # End if first rank

    # Fit the effective mass and two-point functions 

    try:
    
        fitResults[ ismr ] \
            = fit.effEnergyTwopFit( mEff[ ismr ], twop_fold[ ismr ],
                                    rangeEnd, 0, L, tsf,
                                    mpi_confs_info,
                                    plat_t_low_range=[plat_fitStart],
                                    checkFit=checkFit,
                                    fitType="twop",
                                    tsf_t_low_range=[tsf_fitStart] )
    
    except fit.lqcdjk_BadFitError as error: # Bad twop fit
    
        mpi_fncs.mpiPrint( error, mpi_confs_info )
        mpi_fncs.mpiPrint( " Will try fit on effective mass.", 
                           mpi_confs_info )

        try:
        
            fitResults[ ismr ] \
                = fit.effEnergyTwopFit( mEff[ ismr ], 
                                        twop_fold[ ismr ],
                                        rangeEnd, 0, L, tsf,
                                        mpi_confs_info,
                                        tsf_t_low_range=[tsf_fitStart],
                                        plat_t_low_range=[plat_fitStart],
                                        checkFit=checkFit,
                                        fitType="effEnergy" )
            
        except fit.lqcdjk_BadFitError as error: # Bad effEnergy fit

            mpi_fncs.mpiPrintError( "ERROR (lqcdjk_fitting.mEffTwopFit):"
                                  + str( error ), mpi_confs_info )

        # End bad effEnergy fit
    # End bad twop fit

    fitParams = fitResults[ ismr ][ 0 ]
    chiSq = fitResults[ ismr ][ 1 ]

    mEff_fit[ ismr ] = fitResults[ ismr ][ 2 ]

    rangeStart = fitResults[ ismr ][ 3 ]
    mEff_rangeStart = fitResults[ ismr ][ 4 ]

    fitType = fitResults[ ismr ][ 5 ]
    
    if rank == 0 and ismr == 0:

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
                                                   "mEff_{}_1sf_chiSq_{}_{}"
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
    
        mEff_fit_avg[ ismr ] = np.average( mEff_fit[ ismr ], axis=0 )
        mEff_fit_err = fncs.calcError( mEff_fit[ ismr ], binNum )
        
        mEff_plat_fit_str = "2s" + str( mEff_rangeStart ) \
                            + ".2e" + str( rangeEnd )

        mEff_outputFilename = rw.makeFilename( output_template,
                                               "mEff_plat_fit_{}_{}" \
                                               + "_psq{}_{}configs_binSize{}",
                                               particle, mEff_plat_fit_str, 
                                               0, configNum, binSize )
        rw.writeFitDataFile( mEff_outputFilename, mEff_fit_avg[ ismr ],
                             mEff_fit_err, mEff_rangeStart, rangeEnd )

    # End not first process

mEff_fit = comm.bcast( mEff_fit, root=0 )
mEff_fit_avg = comm.bcast( mEff_fit_avg, root=0 )

# End loop over smear


###############################
# Boosted two-point functions #
###############################


if momSq > 0:

    twop_boost_fold_p = np.zeros( ( smearNum_boost, 
                                    momBoostNum, 
                                    binNum, T_fold ) )

    # Loop over smear
    for smr, ismr in zip( smear_str_list_boost, range( smearNum_boost ) ):

        twop_boost = np.zeros( ( momBoostNum, 
                                 binNum, T ) )
    
        twop_template_smr = twop_boost_template[ ismr ].format( smr )

        twop_boost = rw.readTwopFile_zeroQ( twopDir[ ismr ], 
                                            configList_loc, 
                                            configNum,
                                            twop_template_smr, 
                                            srcNum, momSq,
                                            dataFormat_twop[ ismr ], 
                                            mpi_confs_info )

        # Loop over momenta
        for imom in range( momBoostNum ):

            if binNum_loc:
                
                twop_boost_jk_loc = fncs.jackknifeBinSubset( twop_boost[imom],
                                                             binSize,
                                                             binList_loc )

                twop_boost_fold_loc = fncs.fold( twop_boost_jk_loc )

            else:

                twop_boost_fold_loc = np.array( [] )

            comm.Allgatherv( twop_boost_fold_loc, 
                             [ twop_boost_fold_p[ ismr, imom ],
                               recvCount * T_fold,
                               recvOffset * T_fold, 
                               MPI.DOUBLE ] )

        # End loop over momenta
    # End loop over smear
# End if non-zero momentum boost


###########################
# Fit two-point functions #
###########################


if momSq > 0:

    c0 = np.zeros( ( smearNum_boost, binNum ) )
    c1 = np.zeros( ( smearNum_boost, binNum ) )
    E0 = np.zeros( ( smearNum_boost, binNum ) )
    E1 = np.zeros( ( smearNum_boost, binNum ) )

    smear_to_loop_over = zip( smear_str_list_boost, range( smearNum_boost ) )
    twopFit_str = [ [] for smr in smear_str_list_boost ]

else:

    c0 = np.zeros( ( smearNum, binNum ) )
    c1 = np.zeros( ( smearNum, binNum ) )
    E0 = np.zeros( ( smearNum, binNum ) )
    E1 = np.zeros( ( smearNum, binNum ) )

    smear_to_loop_over = zip( smear_str_list, range( smearNum ) )
    twopFit_str = [ [] for smr in smear_str_list ]

# Loop over smear
for smr, ismr in smear_to_loop_over:

    if momSq > 0: # Boosted two-point functions

        # Average over momenta

        twop_to_fit = np.average( twop_boost_fold_p[ ismr ], axis=0 )
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

            fitParams_twop = fitResults_twop[ 0 ]
            twop_rangeStart = fitResults_twop[ 3 ]

            mpi_fncs.mpiPrint( fitResults_twop[ 4 ], mpi_confs_info )
            
        except fit.lqcdjk_BadFitError as error:
        
            mpi_fncs.mpiPrint( error + " Will try fit on effective energy." )

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
        
                mpi_fncs.mpiPrintError( "ERROR (lqcdjk_fitting.mEffTwopFit):"
                                      + str( error ), mpi_confs_info )

            # End bad mEff fit

            twop_rangeStart = fitResults_tmp[ 3 ]
            
            twop_rangeStart = comm.bcast( twop_rangeStart, root=0 )
        
            E_guess = np.sqrt( mEff_fit_avg[ ismr ] ** 2 
                               + ( 2.0 * np.pi / L ) ** 2 * momSq )

            fitParams_twop,chiSq=fit.twoStateFit_twop( twop_to_fit,
                                                       twop_rangeStart,
                                                       rangeEnd, 
                                                       E_guess, T,
                                                       mpi_confs_info )

        # End bad twop fit

    else: # Zero momentum two-point functions

        twop_to_fit = twop_fold[ ismr ]

        if fitType == "effEnergy":

            try:

                fitResults_twop \
                    = fit.effEnergyTwopFit( mEff[ ismr ], 
                                            twop_to_fit[ ismr ],
                                            rangeEnd, momSq, L, tsf,
                                            mpi_confs_info,
                                            tsf_t_low_range=[tsf_fitStart],
                                            plat_t_low_range=[plat_fitStart],
                                            checkFit=checkFit,
                                            fitType="twop" )

                twop_rangeStart = fitResults_twop[ 3 ]
                fitParams_twop = fitResults_twop[ 0 ]
            
            except fit.lqcdjk_BadFitError as error:
        
                mpi_fncs.mpiPrint( error, mpi_confs_info )
                mpi_fncs.mpiPrint( " Will use fit start from fitting "
                                   + "effective energy.", mpi_confs_info )

                twop_rangeStart = rangeStart

                if tsf:

                    E_guess = mEff_fit_avg[ ismr ]

                    fitParams_twop, chiSq \
                        = fit.twoStateFit_twop( twop_to_fit,
                                                twop_rangeStart,
                                                rangeEnd, 
                                                E_guess, T,
                                                mpi_confs_info )
            
                else: # One-state fit

                    fitParams_twop,chiSq \
                        = fit.oneStateFit_twop( twop_to_fit,
                                                twop_rangeStart,
                                                rangeEnd, T )

        elif fitType == "twop":

            fitParams_twop = fitResults[ ismr ][ 0 ]
            twop_rangeStart = fitResults[ ismr ][ 3 ]

    # End if zero momentum boost

    if tsf:

        c0[ ismr ] = np.asarray( fitParams_twop[ :, 0 ],
                                 order='c', dtype=float )
        c1[ ismr ] = np.asarray( fitParams_twop[ :, 1 ],
                                 order='c', dtype=float )
        E0[ ismr ] = np.asarray( fitParams_twop[ :, 2 ],
                                 order='c', dtype=float )
        E1[ ismr ] = np.asarray( fitParams_twop[ :, 3 ],
                                 order='c', dtype=float )

    else: # One-state Fit
    
        c0[ ismr ] = np.asarray( fitParams_twop[ :, 0 ],
                                 order='c', dtype=float )
        E0[ ismr ] = np.asarray( fitParams_twop[ :, 1 ],
                                 order='c', dtype=float )

    # End one-state fit


    ######################################################
    # Write the two-point functions and their fit curves #
    ######################################################


    if rank == 0:

        if tsf:
        
            twop_curve, ts_twop \
                = fit.calcTwopTwoStateCurve( c0[ ismr ], c1[ ismr ], 
                                             E0[ ismr ], E1[ ismr ], T,
                                             twop_rangeStart, 
                                             rangeEnd )
            
            twopFit_str[ ismr ] = "2s" + str( twop_rangeStart ) \
                                  + ".2e" + str( rangeEnd )

            twopParams_filename = rw.makeFilename( output_template,
                                                   "twop_2sf_params_per_bin"
                                                   + "_{}_{}_psq{}_"
                                                   + "{}configs_binSize{}{}",
                                                   particle, twopFit_str[ ismr ],
                                                   momSq, configNum, 
                                                   binSize, smr )

            rw.writeDataFile( twopParams_filename, fitParams_twop )

            fitParams_twop_avg \
                = np.concatenate( ( [ 0, 0, 0 ], \
                                    np.average( fitParams_twop, \
                                                axis=0 ) ) )
            fitParams_twop_err \
                = np.concatenate( ( [ 0, 0, 0 ], \
                                    fncs.calcError( fitParams_twop, \
                                                    binNum ) ) )

            fitParams_twopOutputFilename \
                = rw.makeFilename( output_template,
                                   "twop_2sf_params_{}_{}" \
                                   + "_psq{}_{}configs_binSize{}{}",
                                   particle, twopFit_str[ ismr ], 
                                   momSq, configNum, binSize, smr )
            rw.writeTSFParamsFile( fitParams_twopOutputFilename, \
                                   fitParams_twop_avg, fitParams_twop_err )
            
        else: # One-state fit

            twop_curve, ts_twop \
                = fit.calcTwopOneStateCurve( c0[ ismr ], E0[ ismr ], T,
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
                                            + "_{}configs_binSize{}{}",
                                            particle, momSq,
                                            configNum, binSize, smr )

        rw.writeAvgDataFile( twop_outFilename, twop_avg, twop_err )

        twop_curve_outFilename = rw.makeFilename( output_template,
                                                  "twop_2sf_curve_{}_psq{}" \
                                                  + "_{}configs_binSize{}{}",
                                                  particle, momSq,
                                                  configNum, binSize, smr )
        
        rw.writeAvgDataFile_wX( twop_curve_outFilename, ts_twop,
                                twop_curve_avg, twop_curve_err )
        
    # End first process
# End loop over smear

if particle == "pion" or particle == "nucleon":

    ismr_flav = [ 0 ]

else:

    if momSq == 0:

        ismr_flav = [ 1, 2 ]

    else:

        ismr_flav = [ 0, 1 ]


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

        if whichRatio in mellin_list:

            threep_p = rw.getMellinMomentThreep( threepDir, 
                                                 configList_loc,
                                                 configNum, 
                                                 threep_tokens,
                                                 srcNum, ts, momList[ imom ],
                                                 particle, 
                                                 dataFormat_threep, 
                                                 whichRatio, L, T, 
                                                 mpi_confs_info )

        elif whichRatio in GE_list:

            threep_p = rw.readEMFile( threepDir, 
                                      configList_loc,
                                      configNum,
                                      threep_tokens, 
                                      srcNum, ts, momList[ imom ], 
                                      particle, dataFormat_threep, 
                                      whichRatio, T, 
                                      mpi_confs_info )

        # Loop over flavor
        for iflav in range( flavNum ):

            # Jackknife

            threep_p_jk_loc \
                = fncs.jackknifeBinSubset( threep_p[ iflav ],
                                           binSize, binList_loc )

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
# Calculate ratio #
####################


if rank == 0:

    # ratio[ flav, ts, b, t ]

    ratio = np.zeros( ( flavNum, tsinkNum, binNum, T ) )

    # Loop over flavor
    for iflav in range( flavNum ):
        # Loop over tsink
        for ts, its in zip( tsink, range( tsinkNum ) ) :

            if whichRatio in mellin_list:

                ratio[ iflav, its ] \
                    = Z * pq.calcMellin_twopFit( threep_jk[ iflav, its ],
                                                 ts, mEff_fit[ 0 ],
                                                 momSq, L, 
                                                 c0[ ismr_flav[ iflav ] ],
                                                 E0[ ismr_flav[ iflav ] ], 
                                                 whichRatio )

            elif whichRatio in GE_list:

                ratio[ iflav, its] \
                    = Z * pq.calcMatrixElemEM_twopFit( threep_jk[iflav, its],
                                                       ts, 
                                                       c0[ ismr_flav[ iflav ] ], 
                                                       E0[ ismr_flav[ iflav ] ] ) 

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
            
            # Write threep output file
    
            threep_outFilename = rw.makeFilename( output_template,
                                                  "{}_threep_{}{}_tsink{}" \
                                                  + "_psq{}_{}configs_binSize{}",
                                                  whichRatio, particle, 
                                                  flav_str[ iflav ], ts,
                                                  momSq, configNum, binSize )
            rw.writeAvgDataFile( threep_outFilename,
                                 threep_avg[ iflav, its ],
                                 threep_err[ iflav, its ] )

            # Write ratio output file

            ratio_outFilename \
                = rw.makeFilename( output_template,
                                   "{}_{}{}_tsink{}" \
                                   + "_psq{}_{}configs_binSize{}",
                                   whichRatio, particle, 
                                   flav_str[ iflav ], ts,
                                   momSq, configNum, binSize )
            rw.writeAvgDataFile( ratio_outFilename, ratio_avg[ iflav, its ],
                                 ratio_err[ iflav, its ] )

            # Fit plateau

            rangeStart_plat = [ ts // 2 - 1, ts // 2 - 2,
                                ts // 2 - 3, ts // 2 - 4 ]
            
            fitEnd_plat = [ ts // 2 + 1, ts // 2 + 2,
                            ts // 2 + 3, ts // 2 + 4 ]

            # Loop over fit ranges
            for irange in range( len( rangeStart_plat ) ):

                ratio_fit, chiSq = fit.fitPlateau( ratio[ iflav, its ],
                                                    ratio_err[iflav, its ],
                                                    rangeStart_plat[ irange ],
                                                    fitEnd_plat[ irange ] )

                # Print fit per bin

                ratio_fit_outFilename \
                    = rw.makeFilename( output_template,\
                                       "{}_plat_fit_per_bin_{}_{}" \
                                       + "_{}{}_tsink{}" \
                                       + "_psq{}_{}configs_binSize{}",
                                       whichRatio, 
                                       rangeStart_plat[ irange ],
                                       fitEnd_plat[ irange ],
                                       particle, flav_str[ iflav ], 
                                       ts, momSq, configNum, binSize )

                rw.writeAvgDataFile( ratio_fit_outFilename, 
                                     ratio_fit, 
                                     np.zeros( ratio_fit.shape ) )
                
                # Average over bins

                ratio_fit_avg = np.average( ratio_fit )
                ratio_fit_err = fncs.calcError( ratio_fit, 
                                                 binNum )
                
                # Write output files

                ratio_fit_outFilename \
                    = rw.makeFilename( output_template,\
                                       "{}_plat_fit_{}_{}" \
                                       + "_{}{}_tsink{}" \
                                       + "_psq{}_{}configs_binSize{}",
                                       whichRatio, 
                                       rangeStart_plat[ irange ],
                                       fitEnd_plat[ irange ],
                                       particle, flav_str[ iflav ], 
                                       ts, momSq, configNum, binSize )
                rw.writeFitDataFile( ratio_fit_outFilename, ratio_fit_avg,
                                     ratio_fit_err, rangeStart_plat[ irange ],
                                     fitEnd_plat[ irange ] )
            
            # End loop over fit ranges
        # End loop over tsink
    # End loop over flavor
# End first process

##################
# Two-state Fit  #
##################

if tsf:
# CJL:HERE
    mpi_fncs.mpiPrint( "Will perform the two-state fit", mpi_confs_info )

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
                                                E0[ ismr_flav[ iflav ] ],
                                                E1[ ismr_flav[ iflav ] ],
                                                T, mpi_confs_info )

            if rank == 0:

                a00 = fitParams_threep[ :, 0 ]
                a01 = fitParams_threep[ :, 1 ]
                a11 = fitParams_threep[ :, 2 ]
          
                fitParams = np.stack( ( a00, a01, a11,
                                        c0[ ismr_flav[ iflav ] ],
                                        c1[ ismr_flav[ iflav ] ],
                                        E0[ ismr_flav[ iflav ] ],
                                        E1[ ismr_flav[ iflav ] ] ),
                                      axis=1 )
                
                # ratio[ b ]

                if whichRatio =="avgX":

                    ratio = pq.calcAvgX_twoStateFit( a00,
                                                     c0[ ismr_flav[ iflav ] ],
                                                     mEff_fit[ 0 ], 
                                                     momSq, L, Z )

                elif whichRatio in mellin_list:
                
                    ratio = -Z * a00 / c0[ ismr_flav[ iflav ] ]

                elif whichRatio in GE_list:

                    ratio = Z * a00 / c0[ ismr_flav[ iflav ] ]

                # Average over bins
                    
                ratio_avg = np.average( ratio )
                ratio_err = fncs.calcError( ratio, binNum )
    
                fitParams_avg = np.average( fitParams, axis=0 )
                fitParams_err = fncs.calcError( fitParams, binNum )
                
                chiSq_avg = np.average( chiSq, axis=0 )
                chiSq_err = fncs.calcError( chiSq, binNum )
                
                tsf_threep_range_str = twopFit_str[ ismr_flav[ iflav ] ] + ".3n" \
                                       + str( neglect )

                # Print fit per bin

                ratioOutputFilename \
                    = rw.makeFilename( output_template,
                                       "{}_2sf_per_bin_{}{}_{}_{}" \
                                       + "_psq{}_{}configs_binSize{}",
                                       whichRatio, 
                                       particle, flav_str[ iflav ],
                                       tsf_threep_range_str,
                                       ts_range_str,
                                       momSq, configNum, binSize )

                rw.writeAvgDataFile( ratioOutputFilename, 
                                     ratio,
                                     np.zeros( ratio.shape ) )
                
                # Write ratio output file
            
                ratioOutputFilename \
                    = rw.makeFilename( output_template,
                                       "{}_2sf_{}{}_{}_{}" \
                                       + "_psq{}_{}configs_binSize{}",
                                       whichRatio, 
                                       particle, flav_str[ iflav ],
                                       tsf_threep_range_str,
                                       ts_range_str,
                                       momSq, configNum, binSize )
                rw.writeFitDataFile( ratioOutputFilename,
                                     ratio_avg, ratio_err, 0, 0 )
                
                # Write chi^2 output file
            
                chiSqOutputFilename \
                    = rw.makeFilename( output_template,
                                       "{}_2sf_threep_chiSq_{}{}_{}_{}" \
                                       + "_psq{}_{}configs_binSize{}",
                                       whichRatio, 
                                       particle, flav_str[ iflav ],
                                       tsf_threep_range_str,
                                       ts_range_str,
                                       momSq, configNum, binSize )
        
                rw.writeFitDataFile( chiSqOutputFilename,
                                     chiSq_avg, chiSq_err, 0, 0 )
                
                # Write tsf paramater output file

                ratioParamsOutputFilename \
                    = rw.makeFilename( output_template,
                                       "{}_2sf_params_{}{}_{}_{}" \
                                       + "_psq{}_{}configs_binSize{}",
                                       whichRatio, 
                                       particle, flav_str[ iflav ],
                                       tsf_threep_range_str,
                                       ts_range_str,
                                       momSq, configNum, binSize )

                rw.writeTSFParamsFile( ratioParamsOutputFilename,
                                       fitParams_avg, fitParams_err )
            
                if whichRatio in mellin_list:

                    # Calculate curve with constant tsink
                
                    # curve[ b, ts, t ]
                    # ti[ ts, t ]

                    threep_curve, ti_threep \
                        = fit.calcThreepTwoStateCurve( a00, a01,
                                                       a11, E0[ ismr_flav[ iflav ] ],
                                                       E1[ ismr_flav[ iflav ] ], T,
                                                       tsink,
                                                       ti_to_fit,
                                                       neglect )

                    ratio_curve_const_ts, ti_ratio \
                        = fit.calcAvgXTwoStateCurve_const_ts( a00, 
                                                              a01, 
                                                              a11,
                                                              c0[ismr_flav[iflav]], 
                                                              c1[ismr_flav[iflav]], 
                                                              E0[ismr_flav[iflav]], 
                                                              E1[ismr_flav[iflav]],
                                                              mEff_fit[ 0 ],
                                                              momSq, 
                                                              L, T, 
                                                              Z, 
                                                              tsink,
                                                              ti_to_fit,
                                                              neglect,
                                                              whichRatio )
                    
                    ratio_curve_const_ti, ts_ratio \
                        = fit.calcAvgXTwoStateCurve_const_ti( a00, 
                                                              a01, a11,
                                                              c0[ismr_flav[iflav]], 
                                                              c1[ismr_flav[iflav]],
                                                              E0[ismr_flav[iflav]], 
                                                              E1[ismr_flav[iflav]],
                                                              mEff_fit[ 0 ],
                                                              momSq,
                                                              L, T, Z,
                                                              tsink[0]
                                                              -2,
                                                              tsink[-1]
                                                              +5,
                                                              whichRatio )
                    
                    threep_curve_avg = np.average( threep_curve, axis=0 )
                    threep_curve_err = fncs.calcError( threep_curve, 
                                                       binNum )
                    
                    ratio_curve_const_ts_avg \
                        = np.average( ratio_curve_const_ts, axis=0 )
                    ratio_curve_const_ts_err \
                        = fncs.calcError( ratio_curve_const_ts, binNum )
                
                    ratio_curve_const_ti_avg \
                        = np.average( ratio_curve_const_ti, axis=0 )
                    ratio_curve_const_ti_err \
                        = fncs.calcError( ratio_curve_const_ti, binNum )
                
                    curveOutputFilename \
                        = rw.makeFilename( output_template,
                                           "{}_2sf_curve_{}{}_{}_{}" \
                                           + "_psq{}_{}configs_binSize{}",
                                           whichRatio, 
                                           particle, flav_str[ iflav ],
                                           tsf_threep_range_str,
                                           ts_range_str,
                                           momSq, configNum, binSize )
                    rw.writeAvgDataFile_wX( curveOutputFilename,
                                            ts_ratio,
                                            ratio_curve_const_ti_avg,
                                            ratio_curve_const_ti_err )
                    
                    for ts in range( tsinkNum ):
            
                        threep_curveOutputFilename \
                            = rw.makeFilename( output_template,
                                               "{}_2sf_threep_curve_"
                                               + "{}{}_tsink{}_{}_{}"
                                               + "_psq{}_{}configs_binSize{}",
                                               whichRatio, 
                                               particle, flav_str[ iflav ],
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
                                               "{}_2sf_curve_"
                                               + "{}{}_tsink{}_{}_{}"
                                               + "_psq{}_{}configs_binSize{}",
                                               whichRatio, 
                                               particle, flav_str[ iflav ],
                                               tsink[ ts ],
                                               tsf_threep_range_str,
                                               ts_range_str,
                                               momSq, configNum, binSize )
                        rw.writeAvgDataFile_wX( curveOutputFilename,
                                                ti_ratio[ ts ],
                                                ratio_curve_const_ts_avg[ ts ],
                                                ratio_curve_const_ts_err[ ts ] )
                        
                    # End loop over tsink
                # End if mellin moment
            # End first process
        # End loop over number of neglected three-point functions
    # End loop over flavors
# End two-state fit
