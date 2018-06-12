import math
import h5py
import numpy as np
import re
from os import listdir as ls
from glob import glob

def getTwopDatasets( confs_dir, confs_list, fn_template ):
    
    twop = []

    # Loop through conf indexes

    for c in range( len( confs_list ) ):
        
        twop.append( [] )

        traj = confs_list[c].rpartition( "-" )[-1]

        # Get filenames in specific confs directory which follow template

        filename = glob( confs_dir + "/" + confs_list[c] + "/" + fn_template )

        filename = sorted( filename )

        # Loop through indexes of file names in specific conf directory

        for fn in range( len( filename ) ): 

            with h5py.File( filename[fn], "r" ) as data_file:

                # Ensure the twop group of the file has the 
                # correct trajectory name
                
                dsetname = data_file[ "/conf_" + traj ].visit( lambda x: x if "twop" in x else None )
            
                # Append top group name to the rest of dataset path

                dsetname = "/conf_" + traj + "/" + dsetname

                # Copy dataset to twop

                twop[c].append( np.array( data_file[ dsetname ] ) )

    return np.array( twop )


def get_datasets( filename, datasets ):

    with h5py.File( filename, "r" ) as data_file:

        names = []
    
        data_file.visit( lambda x: names.append(x) if "threep" in x or "twop" in x else None )
    
        for name in names:

            datasets[name] = np.array( data_file[name] )

def check_sources( filenames, sourceNum ):

    check = True

    # Check that the number of files is the same as the given number of sources

    if len( filenames ) is not sourceNum:
        print "check_sources: number of sources is " + str( len( filenames ) ) + ", should be " + str( sourceNum )
        check = False
        
    # Check that all of the source positions are unique

    sourcePos = []

    for fn in filenames:

        with h5py.File( fn, "r" ) as data_file:

            sourcePos.append( str( data_file.visit( lambda x: data_file[x].keys() if "conf" in x else None ) ) )

    for src1 in sourcePos:
        
        matches = 0

        for src2 in sourcePos:

            if src1 == src2:
                matches += 1

        if matches > 1:
            print "check_sources: source positions are not all unique"
            check = False
                
    # Check that all of the secondary groups are actually source dirctories

    # source position format for st < 100
    srcPosFormat2 = "\[u'sx[0-9]{2}sy[0-9]{2}sz[0-9]{2}st[0-9]{2}'\]"
    # source position format for st => 100
    srcPosFormat3 = "\[u'sx[0-9]{2}sy[0-9]{2}sz[0-9]{2}st[0-9]{3}'\]"

    for src in sourcePos:
        # if the source position does not match either format
        if re.match( srcPosFormat2, src ) is None and re.match( srcPosFormat3, src ) is None:
            print "check_sources: second-level group %s not of the proper form" % src
            check = False

    return check

def jackknife( vals, binSize ):

    configNum = len( vals )
    binNum = configNum / binSize

    vals_jk =[]

    for bins in range( binNum ):

        vals_jk.append( [] )

        temp = np.vstack( ( vals[ : bins * binSize, : ], vals[ ( bins + 1 ) * binSize :, : ] ) )

        vals_jk[ bins ] = np.average( temp, axis=0 )

    return np.array( vals_jk )

def calculateEffMass( twop ):

    mEff = np.zeros( twop.shape )

    # Recursively loop though each axis of twop, until the
    # last dimension, which should be time, is reached

    if( twop.ndim > 1 ):

        for dim in range( len( twop ) ):

            mEff[dim] = calculateEffMass( twop[dim] )

    else:

        timeLength = len( twop )

        # Loop through timestep, excluding the last timestep

        for t in range( timeLength - 1 ):

            mEff[ t ] = math.log( twop[ t ] / twop[ t + 1 ] )

            # Calculate effective mass at last timestep, 
            # applying boundary conditions

        mEff[ timeLength - 1 ] = math.log( twop[ timeLength - 1 ] / twop[ 0 ] )

    return np.array( mEff )


def writeDataFile( data, filename ):

    if data.ndim != 2:

        print "Error (writeDataFile): Data array does not have two dimensions"

        return -1

    with open( filename, "w" ) as output:

        for d0 in range( len( data ) ):

            for d1 in range( len( data[ d0 ] ) ):
                
                output.write( str( d1 ).ljust(5) + str( data[ d0, d1 ] ) + "\n" )


def writeAvgDataFile( data, error, filename ):

    if data.ndim != 1:

        print "Error (writeAvgDataFile): Data array has more than one dimension"

        return -1

    if data.shape != error.shape or len( data ) != len( error ):

        print "Error (writeAvgDataFile): Error array's length and shape does not match data array's"
        
        return -1

    with open( filename, "w" ) as output:

        for d0 in range( len( data ) ):

            output.write( str( d0 ).ljust(5) + str( data[ d0 ] ).ljust(20) + str( error[ d0 ] ) + "\n" )
