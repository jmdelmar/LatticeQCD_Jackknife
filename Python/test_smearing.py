import math
import h5py
import numpy as np
import argparse as argp
from os import listdir as ls
import functions as fncs
from glob import glob

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="test smearing" )

parser.add_argument( "confs_dir", action='store', type=str )

parser.add_argument( "filename_template", action='store', type=str )

parser.add_argument( "binSize", action='store', type=int )

parser.add_argument( "-o", "--output", action='store', type=str, default="./" )

args = parser.parse_args()

confs_dir = args.confs_dir

fn_template = args.filename_template

binSize = args.binSize

output_dir = args.output

confs_list = ls( confs_dir )

confs_list = sorted( confs_list )

confsNum = len( confs_list )

if confsNum % binSize != 0:

    print "Number of configurations " + str( confsNum ) \
        + " not evenly divided by bin size " \
        + str( binSize ) + ".\n"

    binSize = input( "Input new bin size: " )

binNum = confsNum / binSize

confs_filename = output_dir + "/confs.txt"

with open( confs_filename, "w" ) as conf_file:

    for conf in confs_list:

        conf_file.write( str( conf ) + "\n" )

print "Conf list written"

#############################################
# Fill matrices with values from data files #
#############################################

twop = fncs.getTwopDatasets( confs_dir, confs_list, fn_template )

print "Read two-point functions from HDF5 files"
            
#############
# Jackknife #
#############

twop_jk = fncs.jackknife( twop[ :, :, :, 0, 0 ], binSize )

##############################
# Calculate effective masses #
##############################

mEff = fncs.calculateEffMass( twop_jk )

mEff_avg = np.average( mEff, axis=0 )

mEff_err = np.std( mEff, axis=0 ) * float( binNum - 1 ) / math.sqrt( float( binNum ) )

######################
# Write output files #
######################

# Loop though number of smears

for ns in range( len( mEff_avg ) ):

    output_filename = output_dir + "/effMass_nsmear" + str( ns + 1 ).zfill(2) + ".dat"

    output_avg_filename = output_dir + "/effMass_avg_nsmear" + str( ns + 1 ).zfill(2) + ".dat"

    fncs.writeDataFile( mEff[ :, ns, : ], output_filename )

    fncs.writeAvgDataFile( mEff_avg[ ns, : ], mEff_err[ ns, : ], output_avg_filename )
