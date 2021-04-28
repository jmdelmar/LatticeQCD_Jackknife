import sys
import time
import numpy as np
import argparse as argp
import functions as fncs
import readWrite as rw

np.set_printoptions(threshold=sys.maxsize)

#########################
# Parse input arguments #
#########################


parser = argp.ArgumentParser( description="Calculate the ratio "
                              + "of two jackknifed quantities." )

parser.add_argument( "data_dir", action='store', type=str )

parser.add_argument( "numer_plat_template", action='store',
                     help="Filename template for plateau fit files for "
                     + "value in the numerator",
                     type=str )

parser.add_argument( "numer_tsf_template", action='store',
                     help="Filename template for two-state fit files for "
                     + "value in the numerator",
                     type=str )

parser.add_argument( "denom_plat_template", action='store',
                     help="Filename template for plateau fit files for "
                     + "value in the denominator",
                     type=str )

parser.add_argument( "denom_tsf_template", action='store',
                     help="Filename template for two-state fit files for "
                     + "value in the denominator",
                     type=str )

parser.add_argument( 't_sink', action='store',
                     help="Comma seperated list of Tsink's",
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( 'plus_minus', action='store',
                     help="Comma seperated list of how many t_i data "
                     + "points above and below tsink/2 to include "
                     + "in plateau fit for each Tsink",
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "bin_num", action='store',
                     help="Number of bins in input files",
                     type=int )

args = parser.parse_args()


#########
# Setup #
#########


# Input directories and filename templates

data_dir = args.data_dir

numer_plat_template = args.numer_plat_template
numer_tsf_template = args.numer_tsf_template

denom_plat_template = args.denom_plat_template
denom_tsf_template = args.denom_tsf_template

# Info on what to analyze

tsink = args.t_sink

plusMinus = args.plus_minus

binNum = args.bin_num

result_template = "{0}: {1:.3f}({2:.0f})"

for ts, pm in zip( tsink, plusMinus ):

    numer_plat_filename = numer_plat_template.format( data_dir,
                                                      ts // 2 - pm,
                                                      ts // 2 + pm,
                                                      ts )

    numer_plat = rw.readNthDataCol( numer_plat_filename, 1 )

    denom_plat_filename = denom_plat_template.format( data_dir,
                                                      ts // 2 - pm,
                                                      ts // 2 + pm,
                                                      ts )

    denom_plat = rw.readNthDataCol( denom_plat_filename, 1 )

    ratio_plat = numer_plat / denom_plat

    ratio_plat_avg = np.average( ratio_plat )
    ratio_plat_err = fncs.calcError( ratio_plat, binNum )

    result_msg = result_template.format( "plateau, tsink={}".format( ts ),
                                         ratio_plat_avg,
                                         10**3
                                         * ratio_plat_err )

    print( result_msg )

numer_tsf_filename = numer_tsf_template.format( data_dir )
denom_tsf_filename = denom_tsf_template.format( data_dir )

numer_tsf = rw.readNthDataCol( numer_tsf_filename, 1 )

denom_tsf = rw.readNthDataCol( denom_tsf_filename, 1 )

ratio_tsf = numer_tsf / denom_tsf

ratio_tsf_avg = np.average( ratio_tsf )
ratio_tsf_err = fncs.calcError( ratio_tsf, binNum )

result_msg = result_template.format( "2-state fit",
                                     ratio_tsf_avg,
                                     10**3
                                     * ratio_tsf_err )

print( result_msg )

print( "Excited state systamatics: {}".format( abs( ratio_plat_avg
                                                    - ratio_tsf_avg ) ) )
