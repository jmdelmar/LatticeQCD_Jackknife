import sys
import time
import numpy as np
import argparse as argp
import functions as fncs
import readWrite as rw
import lqcdjk_fitting as fit

np.set_printoptions(threshold=sys.maxsize)

#########################
# Parse input arguments #
#########################


parser = argp.ArgumentParser( description="Calculate the ratio "
                              + "of two form factors." )

parser.add_argument( "data_dir", action='store', type=str )

parser.add_argument( "numer_params_template", action='store',
                     help="Filename template for monopole fit parameters "
                     + "for the form factor in the numerator",
                     type=str )

parser.add_argument( "denom_params_template", action='store',
                     help="Filename template for monopole fit parameters "
                     + "for the form factor in the denominator",
                     type=str )

parser.add_argument( "form_factor", action='store',
                     help="Which form factor to calculate. Used for "
                     + "input and output file names",
                     type=str )

parser.add_argument( 'particle_flavor', action='store',
                     help="Comma seperated list of particle and flavor "
                     + "in the numerator followed by the denominator. "
                     + "Used for input and output file names",
                     type=lambda s: [str(item) for item in s.split(',')] )

parser.add_argument( "Qsq_last", action='store',
                     help="Last Q^2 to be included in monopole curve "
                     + "calculation",
                     type=float )

parser.add_argument( "bin_num", action='store',
                     help="Number of bins in input files",
                     type=int )

parser.add_argument( "-o", "--output_template", action='store',
                     type=str, default="./*.dat",
                     help="Template for output files. '*' will be "
                     + "replaced with text depending on output data." )

args = parser.parse_args()


#########
# Setup #
#########


# Input directories and filename templates

data_dir = args.data_dir

numer_params_template = args.numer_params_template

denom_params_template = args.denom_params_template

output_template = args.output_template

# Info on what to analyze

part_flav = args.particle_flavor

formFactor = args.form_factor

binNum = args.bin_num

Qsq_last = args.Qsq_last


########################
# Read parameter files #
########################


numer_params_filename = numer_params_template.format( data_dir,
                                                      formFactor,
                                                      part_flav[ 0 ] )

m_numer = rw.readNthDataCol( numer_params_filename, 0 )
F0_numer = rw.readNthDataCol( numer_params_filename, 1 )

denom_params_filename = denom_params_template.format( data_dir,
                                                      formFactor,
                                                      part_flav[ 1 ] )

m_denom = rw.readNthDataCol( denom_params_filename, 0 )
F0_denom = rw.readNthDataCol( denom_params_filename, 1 )


###########################
# Calculate monopole curves #
###########################


curve_numer, Qsq = fit.calcMonopoleCurve( m_numer, F0_numer, Qsq_last )

curve_denom, Qsq = fit.calcMonopoleCurve( m_denom, F0_denom, Qsq_last )

curve_numer_avg = np.average( curve_numer, axis=0 )
curve_numer_err = fncs.calcError( curve_numer, binNum )

curve_denom_avg = np.average( curve_denom, axis=0 )
curve_denom_err = fncs.calcError( curve_denom, binNum )

###################
# Calculate ratio #
###################


ratio = curve_numer / curve_denom

ratio_avg = np.average( ratio, axis=0 )
ratio_err = fncs.calcError( ratio, binNum )


######################
# Write output files #
######################


# Numerator monopole curve

output_filename = rw.makeFilename( output_template,
                                   "{}_monopole_curve_{}",
                                   formFactor, part_flav[ 0 ] )

rw.writeAvgDataFile_wX( output_filename, Qsq,
                        curve_numer_avg, curve_numer_err )

# Denominator monopole curve

output_filename = rw.makeFilename( output_template,
                                   "{}_monopole_curve_{}",
                                   formFactor, part_flav[ 1 ] )

rw.writeAvgDataFile_wX( output_filename, Qsq,
                        curve_denom_avg, curve_denom_err )

# Ratio

output_filename = rw.makeFilename( output_template,
                                   "{}_SU3_ratio_{}_{}",
                                   formFactor, part_flav[ 0 ],
                                   part_flav[ 1 ] )

rw.writeAvgDataFile_wX( output_filename, Qsq,
                        ratio_avg, ratio_err )

