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

#scaleConvertFactor = [ 1.0,
#                       1.0,
#                       1.0 ]
scaleConvertFactor = [ 0.88006465263859557,
                       0.82124624619025055,
                       0.78205608925074033 ]
#scaleConvertFactor = [ 0.64385830344721184,
#                       0.51453823194515136,
#                       0.43934940901402392 ]

np.set_printoptions(threshold=sys.maxsize)

#########################
# Parse input arguments #
#########################


parser = argp.ArgumentParser( description="Reconstruct the PDF from "
                              + "the Mellin moments." )

parser.add_argument( "data_dir", action='store', type=str,
                     help="Path to datafile director" )

parser.add_argument( "avgX_filename_template", action='store',
                     type=str, help="<x> filename template." )

parser.add_argument( "avgX2_filename_template", action='store',
                     type=str, help="<x^2> filename template." )

parser.add_argument( "avgX3_filename_template", action='store',
                     type=str, help="<x^3> filename template." )

#parser.add_argument( "avgX_filename_template", action='store', \
#                     type=lambda s: [str(token) for token in s.split(',')], \
#                     help="Comma seperated list of <x> filename " \
#                     + "templates for pion_u, kaon_u, and kaon_s." )

#parser.add_argument( "moment_filename_template", action='store', \
#                     type=str,
#                     help="PDF moment filename " \
#                     + "template." )

#parser.add_argument( "moment_num", action='store',
#                     help="Number of PDF moments to include in fit.",
#                     type=int )

parser.add_argument( "parameter_num", action='store',
                     help="Number of fitting parameters to use.",
                     type=int )

parser.add_argument( "bin_num", action='store',
                     help="Number of bins in input files",
                     type=int )

parser.add_argument( "-o", "--output_template", action='store',
                     type=str, default="./*.dat" )

args = parser.parse_args()


#########
# Setup #
#########


# Input directories and filename templates

data_dir = args.data_dir

avgX_filename_template = args.avgX_filename_template

avgX2_filename_template = args.avgX2_filename_template

avgX3_filename_template = args.avgX3_filename_template

# Info on what to analyze

#momentNum = args.moment_num
momentNum = 2

paramNum = args.parameter_num

binNum = args.bin_num

# Other info

output_template = args.output_template

moment_list = [ "avgX", "avgX2" ]

if momentNum >= 3:

    moment_list.append( "avgX3" )

#if momentNum >= 4:

#    moment_list.append( "avgX4" )

#if momentNum >= 5:

#    moment_list.append( "avgX5" )

#if momentNum >= 6:

#    moment_list.append( "avgX6" )

particle = [ "pion", "kaon", "kaon" ]
flavor = [ "u", "u", "s" ]

particleNum = len( particle )


#######################
# Read Mellin Moments #
#######################


# moments[ part, momnt, b ]

moments = np.zeros( ( particleNum, momentNum, binNum ) )

moment_filename_template = [ avgX_filename_template,
                             avgX2_filename_template,
                             avgX3_filename_template ]

# Loop over particles
for part, flav, ipart in zip( particle, flavor, range( particleNum ) ):

    # Loop over moments
    for momnt, imomnt in fncs.zipXandIndex( moment_list ):

        moment_filename \
            = moment_filename_template[ imomnt ].format( data_dir,
                                                         part,
                                                         flav )
        
        moments[ ipart, imomnt ] = rw.readNthDataCol( moment_filename, 1 )

    # End loop over moment
# End loop over particle/flavor

if momentNum >= 4:

    moments[ 0, 3, : ] = 0.027 * np.ones( binNum )

    moments[ 1, 3, : ] = 0.021 * np.ones( binNum )

    moments[ 2, 3, : ] = 0.029 * np.ones( binNum )

# Rescale moments to mu=27GeV^2

moments[ :, 0, : ] = scaleConvertFactor[ 0 ] * moments[ :, 0, : ]
moments[ :, 1, : ] = scaleConvertFactor[ 1 ] * moments[ :, 1, : ]

if momentNum >=3:

    moments[ :, 2, : ] = scaleConvertFactor[ 2 ] * moments[ :, 2, : ]
"""
#CJL:HERE
# Multiply pion by 2 to get valence

avgX[ 0 ] = 2. * avgX[ 0 ]
avgX2[ 0 ] = 2. * avgX2[ 0 ]
avgX3[ 0 ] = 2. * avgX3[ 0 ]
"""

######################
# Fit Mellin moments #
######################


# fitParams[ part, b, [ a, b c ] ]

fitParams = np.zeros( ( particleNum, binNum, paramNum ) )

xNum = 100
curve = np.zeros( ( particleNum, binNum, xNum ) )

for part, flav, ipart in zip( particle, flavor, range( particleNum ) ):

    fitParams[ ipart ] = fit.fitMellinMoments( moments[ ipart ],
                                               paramNum )

    a = fitParams[ ipart, :, 0 ]
    b = fitParams[ ipart, :, 1 ]

    if paramNum == 3:

        c = fitParams[ ipart, :, 2 ]

    elif paramNum == 2:

        c = np.zeros( binNum )

    out_str_template = "{}: {} {}"

    a_avg = np.average( a ) 
    b_avg = np.average( b ) 
    c_avg = np.average( c ) 

    a_err = fncs.calcError( a, binNum ) 
    b_err = fncs.calcError( b, binNum ) 
    c_err = fncs.calcError( c, binNum ) 

    print( out_str_template.format( "a", a_avg, a_err ) )
    print( out_str_template.format( "b", b_avg, b_err ) )
    print( out_str_template.format( "c", c_avg, c_err ) )


    ############################################
    # Calculate curves from fitting paramaters #
    ############################################


    curve[ ipart ], x = fit.calcPDFcurve( a, b, c, xNum )

    # PDF at x=0 is nan but we want 0

    curve[ ..., 0 ] = 0

    # Write curve file

    curve_avg = np.average( curve[ ipart ], axis=0 )
    curve_err = fncs.calcError( curve[ ipart ], binNum )

    outputFilename = rw.makeFilename( output_template,
                                      "xPDF_{}_{}_{}params",
                                      part,
                                      flav,
                                      paramNum )

    rw.writeAvgDataFile_wX( outputFilename,
                            x,
                            curve_avg,
                            curve_err )

    # Write fit parameters per bin

    outputFilename \
        = rw.makeFilename( output_template,
                           "xPDF_alpha_per_bin_{}_{}_{}params",
                           part, flav,
                           paramNum )

    rw.writeAvgDataFile( outputFilename, 
                         a,
                         np.zeros( a.shape ) )
    
    outputFilename \
        = rw.makeFilename( output_template,
                           "xPDF_beta_per_bin_{}_{}_{}params",
                           part, flav,
                           paramNum )

    rw.writeAvgDataFile( outputFilename, 
                         b,
                         np.zeros( b.shape ) )

    if paramNum == 3:
    
        outputFilename \
            = rw.makeFilename( output_template,
                               "xPDF_gamma_per_bin_{}_{}_{}params",
                               part, flav,
                               paramNum )
        
        rw.writeAvgDataFile( outputFilename, 
                             c,
                             np.zeros( c.shape ) )
        
    # Write average fit parameters

    outputFilename \
        = rw.makeFilename( output_template,
                           "xPDF_params_{}_{}_{}params",
                           part, flav,
                           paramNum )

    rw.writePDFParamsFile( outputFilename, 
                           [ a_avg, b_avg, c_avg ],
                           [ a_err, b_err, c_err ] )

# End loop over flavors


######################################
# Calculate ratio of q_pi^u to q_K^u #
######################################


curve_ratio = curve[ 0 ] / curve[ 1 ]

curve_ratio_avg = np.average( curve_ratio, axis=0 )
curve_ratio_err = fncs.calcError( curve_ratio, binNum )

outputFilename = rw.makeFilename( output_template,
                                  "xPDF_pion_kaon_ratio_{}params",
                                  paramNum )

rw.writeAvgDataFile_wX( outputFilename,
                        x,
                        curve_ratio_avg,
                        curve_ratio_err )

exit()
