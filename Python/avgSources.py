import math
import h5py
import numpy as np
import argparse as argp
from os import listdir as ls
import readWrite as rw
import functions as fncs
from glob import glob

nptFunction_list = [ "2", "3", "both" ]

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Average Sources" )

parser.add_argument( "threep_dir", action='store', type=str )

parser.add_argument( "threep_template", action='store', type=str )

parser.add_argument( "twop_dir", action='store', type=str )

parser.add_argument( "twop_template", action='store', type=str )

parser.add_argument( 't_sink', action='store', \
                     help="Comma seperated list of t sink's", \
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "-npt", "--n_point_function", action='store', type=str, default="both", \
                     help="n-point functions to average. Should be '2', '3', or 'both'." )

parser.add_argument( "-3o", "--threep_out", action='store', type=str, default="./threep.*_avgSources.h5" )

parser.add_argument( "-2o", "--twop_out", action='store', type=str, default="./twop.*_avgSources.h5" )

parser.add_argument( "-c", "--config_list", action='store', type=str, default="" )

args = parser.parse_args()

threepDir = args.threep_dir

twopDir = args.twop_dir

threep_template = args.threep_template

twop_template = args.twop_template

tsink = args.t_sink

nptFunction = args.n_point_function

threepOut_template = args.threep_out

twopOut_template = args.twop_out

configList = fncs.getConfigList( args.config_list, threepDir )

configNum = len( configList )

# Check inputs

assert nptFunction in nptFunction_list, "Error: n-point function not supported. " \
    + "Supported n-point functions: " + str( particle_list )

#############################################
# Fill matrices with values from data files #
#############################################

threep = []

threepDatasetName = []

if nptFunction == "3" or nptFunction == "both":

    for ts in tsink:

        # Read three-point functions
        # threep[ ts ][ curr_type ][ c, src, ds, ti, q, curr, part ]

        threep_ts, threepDatasetName_ts = rw.getHDF5File_wNames( threepDir, configList, threep_template, "threep", "tsink_" + str( ts ) )

        threep.append( threep_ts )

        threepDatasetName.append( threepDatasetName_ts )

        momentaList_threep = rw.getDatasets( threepDir, configList, threep_template, "Momenta_list_xyz" )

    # Check that all of the momenta are the same

    for c in range( configNum ):

        momList_0 = momentaList_threep[ c, 0 ].flat

        for momList in momentaList_threep[ c ]:

            for i in range( momList.size ):

                assert momList.flat[ i ] == momList_0[ i ], \
                    "Momenta lists in configuration " + configList[ c ] \
                    + " do not match"

    # End loop over tsink

if nptFunction == "2" or nptFunction == "both":

    # Read two-point functions

    twop, twopDatasetName = rw.getDatasets_wNames( twopDir, configList, twop_template, "twop" )

    # Read momenta lists

    momentaList_twop = rw.getDatasets( twopDir, configList, twop_template, "Momenta_list_xyz" )

    # Check that all of the momenta are the same

    for c in range( configNum ):

        momList_0 = momentaList_twop[ c, 0 ].flat

        for momList in momentaList_twop[ c ]:

            for i in range( momList.size ):

                assert momList.flat[ i ] == momList_0[ i ], \
                    "Momenta lists in configuration " + configList[ c ] \
                    + " do not match"

print "Read data from HDF5 files"


if nptFunction == "3" or nptFunction == "both":

    for c in range( configNum ):

        momList_0 = momentaList_threep[ c, 0 ].flat

        for momList in momentaList_threep[ c ]:

            for i in range( momList.size ):

                assert momList.flat[ i ] == momList_0[ i ], \
                    "Momenta lists in configuration " + configList[ c ] \
                    + " do not match"


if nptFunction == "2" or nptFunction == "both":

    for c in range( configNum ):

        momList_0 = momentaList_twop[ c, 0 ].flat

        for momList in momentaList_twop[ c ]:

            for i in range( momList.size ):

                assert momList.flat[ i ] == momList_0[ i ], \
                    "Momenta lists in configuration " + configList[ c ] \
                    + " do not match"


########################
# Average over sources #
########################
    
threep_avg = []
    
twop_avg = []
    
if nptFunction == "3" or nptFunction == "both":

    for threep_ts in threep:

        threep_avg.append( [] )

        for threep_curr in threep_ts:
    
            threep_avg[ -1 ].append( np.average( threep_curr, axis=1 ) )

if nptFunction == "2" or nptFunction == "both":

    twop_avg = np.average( twop, axis=1 )

######################
# Write output files #
######################

# Loop through configurations

for c in range( configNum ):
        
    if nptFunction == "3" or nptFunction == "both":

        # Set three-point output filename

        threepOutFilename = threepOut_template.replace( "*", configList[ c ] )

        with h5py.File( threepOutFilename, "w" ) as outputFile:
        
            # Write momenta list from first source file, which has already been 
            # checked to be the same as the others
        
            outputFile.create_dataset( "/Momenta_list_xyz", data=momentaList_threep[ c ][ 0 ][ 0 ] )

            # Loop over tsink

            for threep_ts, threepDsetName_ts in zip( threep_avg, threepDatasetName ):

                # Loop over insertion currents

                for threep_curr, threepDsetName_curr in zip( threep_ts, threepDsetName_ts ):

                    # Loop over datasets

                    for ds in range( len( threep_curr[ c ] ) ):

                        # Set name output dataset which is the same as the input
                        # but without the source (2nd) group
            
                        outputDataset = threepDsetName_curr[ c ][ 0 ][ ds ][ 0 ]

                        for grp in range( 2, len( threepDsetName_curr[ c ][ 0 ][ ds ] ) ):
                            
                            outputDataset += "/" + threepDsetName_curr[ c ][ 0 ][ ds ][ grp ]
                    
                        # Write averaged three-point functions to output file

                        outputFile.create_dataset( outputDataset, data=threep_curr[ c ][ ds ] )

                    # End loop over datasets

                # End loop over insertion currents

            # End loop over tsink

        # Close threep file

    # End threep

    if nptFunction == "2" or nptFunction == "both":

        # Set two-point output filename

        twopOutFilename = twopOut_template.replace( "*", configList[ c ] )

        with h5py.File( twopOutFilename, "w" ) as outputFile:
        
            # Write momenta list from first source file, which has already been 
            # checked to be the same as the others

            outputFile.create_dataset( "/Momenta_list_xyz", data=momentaList_twop[ c ][ 0 ][ 0 ] )

            # Loop over datasets

            for ds in range( len( twop_avg[ c ] ) ):

                # Set name output dataset which is the same as the input
                # but without the source (2nd) group

                outputDataset = twopDatasetName[ c ][ 0 ][ ds ][ 0 ]

                for grp in range( 2, len( twopDatasetName[ c ][ 0 ][ ds ] ) ):

                    outputDataset += "/" + twopDatasetName[ c ][ 0 ][ ds ][ grp ]

                # Write averaged two-point functions to output file

                outputFile.create_dataset( outputDataset, data=twop_avg[ c ][ ds ] )

            # End loop over datasets

        # Close twop file

    # End twop

# End loop over configs
