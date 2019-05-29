import time
import numpy as np
import argparse as argp
from scipy.optimize import curve_fit
import functions as fncs
import readWrite as rw
import physQuants as pq

Z = 1.0

particle_list = [ "pion", "kaon" ]

format_list = [ "gpu", ]

#########################
# Parse input arguments #
#########################

parser = argp.ArgumentParser( description="Calculate quark momentum fraction <x>" )

parser.add_argument( "threep_dir", action='store', type=str )

parser.add_argument( "threep_template", action='store', type=str )

parser.add_argument( "twop_dir", action='store', type=str )

parser.add_argument( "twop_template", action='store', type=str )

parser.add_argument( "particle", action='store', help="Particle to calculate gA for. Should be 'pion' or 'kaon'.", type=str )

parser.add_argument( 't_sink', action='store', \
                     help="Comma seperated list of t sink's", \
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "-f", "--data_format", action='store', help="Data format. Should be 'gpu' or 'cpu'.", type=str, default="gpu" )

parser.add_argument( "-c", "--config_list", action='store', type=str, default="" )

args = parser.parse_args()

#########
# Setup #
#########

threepDir = args.threep_dir

twopDir = args.twop_dir

threep_template = args.threep_template

twop_template = args.twop_template

particle = args.particle

tsink = args.t_sink

tsinkNum = len( tsink )

dataFormat = args.data_format

# Check inputs

assert particle in particle_list, "Error: Particle not supported. " \
    + "Supported particles: " + str( particle_list )

assert dataFormat in format_list, "Error: Data format not supported. " \
    + "Supported particles: " + str( format_list )

# Get configurations from given list or from given 
# threep directory if list not given

configList = fncs.getConfigList( args.config_list, threepDir )

configNum = len( configList )

#######################
# Two-point functions #
#######################

# Get the real part of two-point functions
# twop[ c, t ]

twop = []

start = time.time()

twop = rw.getDatasets( twopDir, configList, twop_template, "twop" )

print "Got two-point function datasets from HDF5 files in " \
    + str(time.time() - start) + " sec."

print twop.shape

#########################
# Three-point functions #
#########################

for ts in tsink:
 
    start = time.time()

    threep_gxDx = rw.getDatasets( threepDir, configList, threep_template, \
                              "tsink_" + str( ts ), "oneD", "dir_00", \
                              "up", "threep" )

    threep_gyDy = rw.getDatasets( threepDir, configList, threep_template, \
                                  "tsink_" + str( ts ), "oneD", "dir_01", \
                                  "up", "threep" )
    
    threep_gzDz = rw.getDatasets( threepDir, configList, threep_template, \
                                  "tsink_" + str( ts ), "oneD", "dir_02", \
                                  "up", "threep" )

    threep_gtDt = rw.getDatasets( threepDir, configList, threep_template, \
                                  "tsink_" + str( ts ), "oneD", "dir_03", \
                                  "up", "threep" )

    threep_s_gxDx = np.array( [] )
    
    threep_s_gyDy = np.array( [] )
        
    threep_s_gzDz = np.array( [] )
    
    threep_s_gtDt = np.array( [] )

    if particle == "kaon":
            
        threep_s_gxDx = rw.getDatasets( threepDir, configList, threep_template, \
                                        "tsink_" + str( ts ), "oneD", "dir_00", \
                                        "strange", "threep" )

        threep_s_gyDy = rw.getDatasets( threepDir, configList, threep_template, \
                                        "tsink_" + str( ts ), "oneD", "dir_01", \
                                        "strange", "threep" )
    
        threep_s_gzDz = rw.getDatasets( threepDir, configList, threep_template, \
                                        "tsink_" + str( ts ), "oneD", "dir_02", \
                                        "strange", "threep" )

        threep_s_gtDt = rw.getDatasets( threepDir, configList, threep_template, \
                                        "tsink_" + str( ts ), "oneD", "dir_03", \
                                        "strange", "threep" )

    print "Got three-point function datasets from HDF5 files for tsink " \
        + str( ts ) +" in " + str(time.time() - start) +" sec."

    print threep_gtDt.shape
    
    print threep_gxDx.shape
        
    print threep_gyDy.shape

    print threep_gzDz.shape

exit()
