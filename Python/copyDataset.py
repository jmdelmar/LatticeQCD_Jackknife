import math
import h5py
import numpy as np
import argparse as argp
from os import listdir as ls
import readWrite as rw
import functions as fncs
from glob import glob

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Average Sources" )

parser.add_argument( "input_dir", action='store', type=str )

parser.add_argument( "input_template", action='store', type=str )

parser.add_argument( "output_dir", action='store', type=str )

parser.add_argument( "output_template", action='store', type=str )

parser.add_argument( "config_list", action='store', type=str )

parser.add_argument( "keywords", action='store', nargs=argp.REMAINDER )

args = parser.parse_args()

inputDir = args.input_dir

input_template = args.input_template

outputDir = args.output_dir

output_template = args.output_template

configList = fncs.getConfigList( args.config_list, inputDir )

keywords = args.keywords

configNum = len( configList )

###################################################
# Fill matrices with values from input data files #
###################################################

dataset, datasetName = rw.getHDF5File_wNames( inputDir, configList, input_template, *keywords )

print "Read data from HDF5 files"

##################################
# Write datasets to output files #
##################################

# Loop over insertion currents

for dset_curr, dsetName_curr in zip( dataset, datasetName ):

    # Loop through configurations

    #for c in range( configNum ):
    for conf, dset_c, dsetName_c in zip( configList, dset_curr, dsetName_curr ):

        # Loop through sources

        #for src in range( dsetDatasetName[ c ].shape[ 0 ] ):
        for dset_src, dsetName_src in zip( dset_c, dsetName_c ):

            #srcPos = rw.getSourcePostitions( dsetDatasetName[ c ][ src ][ 0 ][ 1 ] )
            srcPos = rw.getSourcePositions( dsetName_src[ 0 ][ 1 ] )

            # Set output filename

            outputFilename = outputDir + "/" + conf + "/" \
                             + output_template.replace( "*", conf ) \

            for s in srcPos:

                outputFilename = outputFilename + s + "."

            outputFilename = outputFilename + "h5"
                
            with h5py.File( outputFilename, "a" ) as outputFile:
            
                # Loop over datasets

                for dset, dsetName in zip( dset_src, dsetName_src ):
                
                    dset = np.array( dset )

                    outputDataset = ""

                    for grp in dsetName:

                        outputDataset += "/" + grp
                    
                    # Write three-point functions to output file if it
                    # does not already exist

                    if outputDataset not in outputFile:

                        outputFile.create_dataset( outputDataset, data=dset )
                    
                # End loop over datasets

            # Close file

        # End loop over sources

    # End loop over configs

# End loop over insertion currents

print "Copied dataset to output file"
