import re
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

parser = argp.ArgumentParser( description="Fix sign of three-point functions near the boundary." )

parser.add_argument( "threep_dir", action='store', type=str )

parser.add_argument( "source_list_filename", action='store', type=str )

parser.add_argument( 't_sink', action='store', \
                     help="Comma seperated list of t sink's", \
                     type=lambda s: [int(item) for item in s.split(',')] )

parser.add_argument( "pSq", action='store', type=int )

parser.add_argument( "-s", "--source_number", action='store', type=int, default=16 )

parser.add_argument( "-T", "--lattice_dim", action='store', type=int, default=64 )

parser.add_argument( "-o", "--parent_out_dir", action='store', type=str, default="./" )

parser.add_argument( "-c", "--config_list", action='store', type=str, default="" )

args = parser.parse_args()

threepDir = args.threep_dir

srcListFilename = args.source_list_filename

tsink = args.t_sink

parentOutDir = args.parent_out_dir

configList = fncs.getConfigList( args.config_list, threepDir )

configNum = len( configList )

if args.pSq == 0:

    pList = [ [ 0, 0, 0 ] ]

elif args.pSq == 3:

    pList = [ [ +1, +1, +1 ], \
              [ -1, +1, +1 ], \
              [ +1, -1, +1 ], \
              [ +1, +1, -1 ], \
              [ -1, -1, +1 ], \
              [ -1, +1, -1 ], \
              [ +1, -1, -1 ], \
              [ -1, -1, -1 ] ]

else:
    
    print( "ERROR: p^2 values not supported." )

    exit()

filenameTemplate = "{}/ft_thrp_{}_sx{:0>2}sy{:0>2}sz{:0>2}st{:0>2}_{}_aN50a0p5_dt{}_mom_{:+d}_{:+d}_{:+d}.{}.h5"

partList = [ "pion", "kaon", "kaon" ]
flavList = [ "up", "up", "strange" ]
smrStrList = [ "gN50a0p2", "gN40a0p2", "gN40a0p2" ]

for conf in configList:

    inDir = "{}/{}".format( threepDir, conf )
    outDir = "{}/{}".format( parentOutDir, conf )

    repl = conf.split("-")[ 0 ]
    traj = conf.split("-")[ 1 ]

    srcList = []

    for ts in tsink:

        with open( srcListFilename, "r" ) as srcListFile:

            for line in srcListFile:

                if re.search( traj + ":", line ):

                    srcList = np.array( line.split() )[ 1: ]

        for src in srcList[ :args.source_number ]:

            srcPos = rw.getSourcePositions( src )

            sx = int( srcPos[ 0 ] )
            sy = int( srcPos[ 1 ] )
            sz = int( srcPos[ 2 ] )
            st = int( srcPos[ 3 ] )

            for part, flav, smrStr in zip( partList, flavList, smrStrList ):

                for p in pList:
        
                    px = p[ 0 ]
                    py = p[ 1 ]
                    pz = p[ 2 ]

                    inFilename = filenameTemplate.format( inDir, part, \
                                                          sx, sy, sz, st, \
                                                          smrStr, ts, \
                                                          px, py, pz, flav )
                        
                    dsetnameList = []
                    dset = {}

                    with h5py.File( inFilename, "r" ) as dataFile:
                
                        # Put all datasets into list

                        dataFile.visititems( lambda name,obj: \
                                             dsetnameList.append(name) \
                                             if type( obj ) is h5py.Dataset \
                                             else None )
                            
                        for dsetname in dsetnameList:
                            
                            dset[ dsetname ] = np.array( dataFile[ dsetname ] )

                    # Close file

                    outFilename = filenameTemplate.format( outDir, part, \
                                                           sx, sy, sz, st, \
                                                           smrStr, ts, \
                                                           px, py, pz, flav )
                        
                    with h5py.File( outFilename, "w" ) as outFile:

                        for dsetname in dsetnameList:

                            if st >= args.lattice_dim - ts and "arr" in dsetname:

                                dset[ dsetname ][ :ts+1, : ] \
                                    = -1.0 * dset[ dsetname ][ :ts+1, : ]
                                
                            outFile.create_dataset( dsetname, \
                                                    data=dset[ dsetname ] )

                    print( "Wrote " + outFilename )

                # End loop over p
            # End loop over part
        # End loop over source
    # End loop over tsink
# End loop over configs

