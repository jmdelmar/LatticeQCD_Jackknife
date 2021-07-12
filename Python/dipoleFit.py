import sys
import time
import numpy as np
import argparse as argp
from scipy.optimize import curve_fit
import functions as fncs
import readWrite as rw
import physQuants as pq
import lqcdjk_fitting as fit

# Set whether or not to use the dispersion
# relation E = sqrt( m^2 + p^2 ) when fitting
# two-point functions

dispRel = True

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

parser.add_argument( "data_dir", action='store', type=str,
                     help="Directory containing form factor data." )

parser.add_argument( "formFactor_filename_template", action='store',
                     type=lambda s: [str(item) for item in s.split(',')],
                     help="Comma seperated list of the form factor filename "
                     "template for each momentum frame." )

parser.add_argument( "formFactor_tsf_filename_template", action='store',
                     type=lambda s: [str(item) for item in s.split(',')],
                     help="Comma seperated list of the 2-state fit form factor "
                     + "filename template for each momentum frame." )

parser.add_argument( "form_factor", action='store', type=str,
                     help="Form factor to calculate. Must be one of "
                     + ", ".join( form_factor_list ) )

parser.add_argument( "particle", action='store',
                     help="Particle to calculate form facors for. "
                     + "Must be one of"
                     + ", ".join( particle_list ),
                     type=str )

parser.add_argument( 't_sink', action='store',
                     help="Comma seperated list of t sink's",
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "boosted_momentum_squared", action='store',
                     type=int, 
                     help="Momentum squared of the boosted frame." )

parser.add_argument( "binNum", action='store', type=int,
                     help="Size of bins for jackknife resampling." )

parser.add_argument( "--Qsq_last_rest", action='store',
                     type=float, default=100.,
                     help="Number of  Q^2 to include in the dipole fit "
                     + "for the rest frame." )

parser.add_argument( "--Qsq_last_boost", action='store',
                     type=float, default=100.,
                     help="Number of  Q^2 to include in the dipole fit "
                     + "for the boosted frame." )

parser.add_argument( "-o", "--output_template", action='store',
                     type=str, default="./*.dat",
                     help="Template for output files. '*' will be "
                     + "replaced with text depending on output data." )

# Parse

args = parser.parse_args()

#########
# Setup #
#########

# Input directories and filename templates

dataDir = args.data_dir

formFactor_filename_template = args.formFactor_filename_template

formFactor_tsf_filename_template = args.formFactor_tsf_filename_template

# Info on what to analyze

Qsq_last_rest = args.Qsq_last_rest
Qsq_last_boost = args.Qsq_last_boost

formFactor = args.form_factor

particle = args.particle

tsink = args.t_sink
tsinkNum = len( tsink )
ts_range_str = "tsink" + str(tsink[0]) + "_" + str(tsink[-1])

psq = args.boosted_momentum_squared

# Other info

binNum = args.binNum

output_template = args.output_template

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

# Check inputs

# Check that particle is valid

assert particle in particle_list, \
    "Error: Particle not supported. " \
    + "Supported particles: " + str( particle_list )

# Check that three-point functions are in boosted frame
# if we are calculating the 2- or 3-derivative form factors

if formFactor in [ "A30_B30", "A40_B40_C40" ] and pSq_fin == 0:

    errorMessage = "Error (formFactors.py): {} form factors cannot be " \
                   + "calculated in the rest frame."

    mpi_fncs.mpiPrintError( errorMessage.format( formFactor ),
                            mpi_confs_info )

# Set projectors and flavors based on particle

if formFactor == "GE_GM":

    if particle == "pion":

        flav_str = [ "", "_u" ]

    elif particle == "kaon":

        flav_str = [ "", "_u", "_s" ]

elif particle == "pion":

    flav_str = [ "_u" ]

elif particle == "kaon":

    flav_str = [ "_u", "_s" ]
    
elif particle == "nucleon":

    print( "Error (dipoleFit.py): nucleon not supported." )

    exit()

flavNum = len( flav_str )

# Loop over form factors
for ff, iff in fncs.zipXandIndex( F_str ):
    # Loop over flavors
    for flav, iflav in fncs.zipXandIndex( flav_str ):
        # Loop over tsink
        for ts, its in fncs.zipXandIndex( tsink ):
            

            #######################
            # Read the data files #
            #######################


            # Momentum transfer squared in the rest frame
            # Qsq_rest[ b, q ]

            # Form factor in the rest frame
            # F_rest[ b, q ]

            F_filename_rest \
                = dataDir + "/" \
                + formFactor_filename_template[ 0 ].format( ff,
                                                            particle,
                                                            flav, ts,
                                                            0 )
            
            Qsq_rest, F_rest \
                = rw.readFormFactorFile_ASCII( F_filename_rest, binNum )

            # Momentum transfer squared in the boosted frame
            # Qsq_boost[ b, q ]

            # Form factor in the boosted frame
            # F_boost[ b, q ]

            F_filename_boost \
                = dataDir + "/" \
                + formFactor_filename_template[ 1 ].format( ff,
                                                            particle,
                                                            flav, ts,
                                                            psq )
            
            Qsq_boost, F_boost \
                = rw.readFormFactorFile_ASCII( F_filename_boost, binNum )

            
            ##################################################
            # Combine and sort form factors from both frames #
            ##################################################

            
            F_rest = F_rest[ Qsq_rest <= Qsq_last_rest  ]
            F_rest = F_rest.reshape( binNum, F_rest.size // binNum )

            F_boost = F_boost[ Qsq_boost <= Qsq_last_boost ]
            F_boost = F_boost.reshape( binNum, F_boost.size // binNum )

            Qsq_rest = Qsq_rest[ Qsq_rest <= Qsq_last_rest  ]
            Qsq_rest = Qsq_rest.reshape( binNum, Qsq_rest.size // binNum )

            Qsq_boost = Qsq_boost[ Qsq_boost <= Qsq_last_boost ]
            Qsq_boost = Qsq_boost.reshape( binNum, Qsq_boost.size // binNum )

            Qsq = np.concatenate( ( Qsq_rest, Qsq_boost ),
                                  axis=-1 )

            F = np.concatenate( ( F_rest, F_boost ),
                                axis=-1 )

            for ib in range( binNum ):

                iq_sort = np.argsort( Qsq[ ib ] )

                Qsq[ ib ] = Qsq[ ib, iq_sort ]
            
                F[ ib ] = F[ ib, iq_sort ]

            # Average over bins

            Qsq_avg = np.average( Qsq, axis=0 )
            Qsq_err = fncs.calcError( Qsq, binNum )

            F_avg = np.average( F, axis=0 )
            F_err = fncs.calcError( F, binNum )

            if formFactor == "BT10":

                # Remove elements where form factor is are zero

                F = F[ :, F_avg != 0. ]
                Qsq = Qsq[ :, F_avg != 0. ]
                Qsq_avg = Qsq_avg[ F_avg != 0. ]
                Qsq_err = Qsq_err[ F_avg != 0. ]
                F_err = F_err[ F_avg != 0. ]
            
                F_avg = F_avg[ F_avg != 0. ]

            #################################################
            # Fit the form factors to a dipole distribution #
            #################################################


            # Loop over number of parameters

            for paramNum_dipole in 1, 2:

                # Fit form factors to dipole
                # fitParams[ b, param ]

                fitParams_dipole, chiSq_dipole \
                    = fit.fitFormFactor_dipole( F, F_err,
                                                Qsq,
                                                paramNum_dipole,
                                                None )
                               
                m_dipole = fitParams_dipole[ :, 0 ]
                F0_dipole = fitParams_dipole[ :, 1 ]

                # Calculate r^2
            
                rSq = 6. / m_dipole ** 2

                # Calculate dipole curve
                # curve_dipole[ b ]

                curve_dipole, Qsq_curve \
                    = fit.calcDipoleCurve( m_dipole, F0_dipole,
                                           Qsq_avg[ -1 ] )

                # Average over bins

                fitParams_dipole_avg = np.average( fitParams_dipole, axis=0 )
                fitParams_dipole_err = fncs.calcError( fitParams_dipole,
                                                       binNum )
                
                rSq_avg = np.average( rSq, axis=0 )
                rSq_err = fncs.calcError( rSq, binNum )

                curve_dipole_avg = np.average( curve_dipole, axis=0 )
                curve_dipole_err = fncs.calcError( curve_dipole, binNum )

                
                ######################
                # Write output files #
                ######################


                # Write dipole fit parameter file for each bin
                    
                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_dipoleFitParams_per_bin_{}{}"
                                       + "_{}params_tsink{}",
                                       ff, particle, flav,
                                       paramNum_dipole, ts )

                rw.write2ValueDataFile( output_filename,
                                        fitParams_dipole[ :, 0 ],
                                        fitParams_dipole[ :, 1 ] )

                # Write average fit parameter file and r^2
                    
                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_dipoleFitParams"
                                       + "_{}{}_{}params_tsink{}",
                                       ff, particle, flav,
                                       paramNum_dipole, ts )

                rw.writeDipoleFitParamsFile( output_filename,
                                             fitParams_dipole_avg,
                                             fitParams_dipole_err,
                                             rSq_avg, rSq_err )

                # Write dipole fit curve
                    
                output_filename \
                    = rw.makeFilename( output_template,
                                       "{}_dipole_curve_{}{}"
                                       + "_{}params_tsink{}",
                                       ff, particle, flav,
                                       paramNum_dipole, ts )

                rw.writeAvgDataFile_wX( output_filename,
                                        Qsq_curve,
                                        curve_dipole_avg,
                                        curve_dipole_err )

            # End loop over parameter number
        # End loop over tsink


        #################
        # Two-state fit #
        #################

        #######################
        # Read the data files #
        #######################

        
        # Momentum transfer squared in the rest frame
        # Qsq_rest[ b, q ]

        # Form factor in the rest frame
        # F_rest[ b, q ]

        F_filename_rest \
            = dataDir + "/" \
            + formFactor_tsf_filename_template[ 0 ].format( ff,
                                                            particle,
                                                            flav,
                                                            0 )
            
        Qsq_rest, F_rest \
            = rw.readFormFactorFile_ASCII( F_filename_rest, binNum )

        # Momentum transfer squared in the boosted frame
        # Qsq_boost[ b, q ]

        # Form factor in the boosted frame
        # F_boost[ b, q ]

        F_filename_boost \
            = dataDir + "/" \
            + formFactor_tsf_filename_template[ 1 ].format( ff,
                                                            particle,
                                                            flav,
                                                            psq )

        Qsq_boost, F_boost \
            = rw.readFormFactorFile_ASCII( F_filename_boost, binNum )

            
        ##################################################
        # Combine and sort form factors from both frames #
        ##################################################


        F_rest = F_rest[ Qsq_rest <= Qsq_last_rest  ]
        F_rest = F_rest.reshape( binNum, F_rest.size // binNum )

        F_boost = F_boost[ Qsq_boost <= Qsq_last_boost ]
        F_boost = F_boost.reshape( binNum, F_boost.size // binNum )

        Qsq_rest = Qsq_rest[ Qsq_rest <= Qsq_last_rest  ]
        Qsq_rest = Qsq_rest.reshape( binNum, Qsq_rest.size // binNum )

        Qsq_boost = Qsq_boost[ Qsq_boost <= Qsq_last_boost ]
        Qsq_boost = Qsq_boost.reshape( binNum, Qsq_boost.size // binNum )

        Qsq = np.concatenate( ( Qsq_rest, Qsq_boost ),
                              axis=-1 )

        F = np.concatenate( ( F_rest, F_boost ),
                            axis=-1 )

        for ib in range( binNum ):

            iq_sort = np.argsort( Qsq[ ib ] )
            
            Qsq[ ib ] = Qsq[ ib, iq_sort ]
            
            F[ ib ] = F[ ib, iq_sort ]

        # Average over bins

        Qsq_avg = np.average( Qsq, axis=0 )
        Qsq_err = fncs.calcError( Qsq, binNum )

        F_avg = np.average( F, axis=0 )
        F_err = fncs.calcError( F, binNum )

        if formFactor == "BT10":

            # Remove elements where form factor is are zero

            F = F[ :, F_avg != 0. ]
            Qsq = Qsq[ :, F_avg != 0. ]
            Qsq_avg = Qsq_avg[ F_avg != 0. ]
            Qsq_err = Qsq_err[ F_avg != 0. ]
            F_err = F_err[ F_avg != 0. ]
            
            F_avg = F_avg[ F_avg != 0. ]


        #################################################
        # Fit the form factors to a dipole distribution #
        #################################################


        # Loop over number of parameters

        for paramNum_dipole in 1, 2:

            # Fit form factors to dipole
            # fitParams[ b, param ]
            
            fitParams_dipole, chiSq_dipole \
                = fit.fitFormFactor_dipole( F, F_err,
                                            Qsq,
                                            paramNum_dipole,
                                            None )
                               
            m_dipole = fitParams_dipole[ :, 0 ]
            F0_dipole = fitParams_dipole[ :, 1 ]
            
            # Calculate r^2

            rSq = 6. / m_dipole ** 2

            # Calculate dipole curve
            # curve_dipole[ b ]
            
            curve_dipole, Qsq_curve \
                = fit.calcDipoleCurve( m_dipole, F0_dipole,
                                       Qsq_avg[ -1 ] )
            
            # Average over bins
            
            fitParams_dipole_avg = np.average( fitParams_dipole, axis=0 )
            fitParams_dipole_err = fncs.calcError( fitParams_dipole,
                                                   binNum )
            
            rSq_avg = np.average( rSq, axis=0 )
            rSq_err = fncs.calcError( rSq, binNum )

            curve_dipole_avg = np.average( curve_dipole, axis=0 )
            curve_dipole_err = fncs.calcError( curve_dipole, binNum )


            ######################
            # Write output files #
            ######################


            # Write dipole fit parameter file for each bin

            output_filename \
                = rw.makeFilename( output_template,
                                   "{}_dipoleFitParams_per_bin_2sf_{}{}"
                                   + "_{}params",
                                   ff, particle, flav,
                                   paramNum_dipole )

            rw.write2ValueDataFile( output_filename,
                                    fitParams_dipole[ :, 0 ],
                                    fitParams_dipole[ :, 1 ] )
            
            # Write average fit parameter file and r^2
                    
            output_filename \
                = rw.makeFilename( output_template,
                                   "{}_dipoleFitParams"
                                   + "_2sf_{}{}_{}params",
                                   ff, particle, flav,
                                   paramNum_dipole )

            rw.writeDipoleFitParamsFile( output_filename,
                                         fitParams_dipole_avg,
                                         fitParams_dipole_err,
                                         rSq_avg, rSq_err )

            # Write dipole fit curve
                    
            output_filename \
                = rw.makeFilename( output_template,
                                   "{}_dipole_curve_2sf_{}{}"
                                   + "_{}params",
                                   ff, particle, flav,
                                   paramNum_dipole )

            rw.writeAvgDataFile_wX( output_filename,
                                    Qsq_curve,
                                    curve_dipole_avg,
                                    curve_dipole_err )

            

    # End loop over flavor
# End loop over form factor

exit()
