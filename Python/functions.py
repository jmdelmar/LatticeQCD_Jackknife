import math
import h5py
import numpy as np
import re
from os import listdir as ls
from glob import glob

def calcError( vals, binNum, **kwargs ):

    if kwargs:

        return np.std( vals, **kwargs ) \
            * float( binNum - 1 ) \
            * float( binNum ) ** -0.5

    else:

        return np.std( vals, axis=0 ) \
            * float( binNum - 1 ) \
            * float( binNum ) ** -0.5


def signToString( val ):

    if val >= 0:

        return "+"

    else:

        return ""

def combineMomBoosts(threep, momSq):

    if momSq == 0:

        # Nothing to be done for zero-momentum boost

        return threep

    elif momSq == 2:

        # Momentum boosts should be ordered as:

        # [ +1, +1,  0],
        # [ +1,  0, +1],
        # [  0, +1, +1],
        # [ +1, -1,  0],
        # [ +1,  0, -1],
        # [  0, +1, -1],
        # [ -1, +1,  0],
        # [ -1,  0, +1],
        # [  0, -1, +1],
        # [ -1, -1,  0],
        # [ -1,  0, -1],
        # [  0, -1, -1]

        return threep[0, ...] + threep[1, ...] + threep[2, ...] \
            + threep[3, ...] + threep[4, ...] + threep[5, ...] \
            + threep[6, ...] + threep[7, ...] + threep[8, ...] \
            + threep[9, ...] + threep[10, ...] + threep[11, ...] / 12

    elif momSq == 3:

        # Momentum boosts should be ordered as:

        # [ +1, +1, +1],
        # [ -1, +1, +1],
        # [ +1, -1, +1],
        # [ +1, +1, -1],
        # [ +1, -1, -1],
        # [ -1, +1, -1],
        # [ -1, -1, +1],
        # [ -1, -1, -1]

        return threep[0, ...] + threep[1, ...] \
            + threep[2, ...] + threep[3, ...] \
            + threep[4, ...] + threep[5, ...] \
            + threep[6, ...] + threep[7, ...] / 8

    else:

        print( "Error: momentum boost squared value " \
            + momSq + " is not supported.\n" )


def initEmptyList( list_in, list_out_order ):

    list_out = []

    if list_out_order == 1:

        list_out = [ [] for i in range( list_in ) ]

    elif list_out_order == 2:

        list_out = [ [ [] for j in range( len( list_in[i] ) ) ] \
                     for i in range( len( list_in ) ) ]

    elif list_out_order == 3:

        list_out = [ [ [ [] for k in range( len( list_in[i][j] ) ) ] \
                       for j in range( len( list_in[i] ) ) ] \
                     for i in range( len( list_in ) ) ]

    else:

        print( "Error(initEmptyList): list order not supported.\n" )

    return list_out


def fold( data ):

    timestepNum = data.shape[ -1 ]

    out = np.zeros( data.shape[ :-1 ] + ( timestepNum // 2 + 1, ) )

    out[ ..., 0 ] = data[ ..., 0 ]

    for t in range( 1, timestepNum // 2 ):
        
        out[ ..., t ] = ( data[ ..., t ] + data[ ..., -t ] ) / 2

    out[ ..., timestepNum // 2 ] = data[ ..., timestepNum // 2 ]

    return out


def getConfigList( configListFilename, configDir ):

    if configListFilename:

        if glob( configListFilename ):

            with open( configListFilename, "r" ) as configFile:
                
                configList = configFile.read().splitlines()

                #print( "Configuration list read" )

        else:

            print( "WARNING: Given configuration does not exist. " \
                + "Will use all configurations in configuration directory." )
            
            configList = ls( configDir )

            configList = sorted( configList )

            with open( configListFilename, "w" ) as configFile:

                for config in configList:

                    configFile.write( str( config ) + "\n" )

            print( "Configuration list written" )

    else:

        configList = ls( configDir )

        configList = sorted( configList )

        configFilename = "./confs.txt"

        if glob( configFilename ):

            print( "WARNING: Configuration list already exists in this directory. " \
                + "Will not overwrite." )

        else:

            with open( configFilename, "w" ) as configFile:

                for config in configList:

                    configFile.write( str( config ) + "\n" )

            print( "Configuration list written" )

    #print( "Number of configurations: " + str( len( configList ) ) )

    return configList


def processMomList( momLists ):

    if momLists.ndim > 2:

        if len( momLists ) > 1:

            # Check that momenta lists are the same across configurations

            momList_0 = momLists[ 0 ].flat

            for ml in momLists[ 1: ]:

                for i in range( ml.size ):

                    assert ml.flat[ i ] == momList_0[ i ], \
                        "Momenta lists do not match."

        momList = momLists[ 0 ]

    else:

        momList = momLists

    # Get indexes where each Q^2 begins and ends

    Qsq_start = []

    Qsq_end = []

    Qsq = np.round( ( np.apply_along_axis( np.linalg.norm, \
                                           1, np.array( momList ) ) ) ** 2 )
    
    q_last = Qsq[ 0 ]

    Qsq_start.append( 0 )

    Qsq_end.append( 0 )

    for q in Qsq[ 1: ]:

        if q != q_last:

            assert q > q_last, "Momentum list not in assending order."

            Qsq_start.append( np.where( Qsq == q )[0][0] )

            Qsq_end.append ( np.where( Qsq == q )[0][-1] )

            q_last = q

    # Remove duplicate Q^2's

    Qsq = sorted( list( set( Qsq ) ) )

    return np.array( Qsq, dtype=int ), np.array( Qsq_start ), np.array( Qsq_end )


# Averages over equal Q^2 for numpy array whose last dimension is Q and 
# returns averaged data as a numpy array whose first dimension is Q^2

def averageOverQsq( data, Qsq_start, Qsq_end ):

    avg = []

    for start, end in zip( Qsq_start, Qsq_end ):

        avg.append( np.average( data[ ..., start : end + 1 ], axis=-1 ) )
            
    return np.array( avg )
    

def check_sources( filenames, sourceNum ):

    check = True

    # Check that the number of files is the same as the given number of sources

    if len( filenames ) is not sourceNum:
        print( "check_sources: number of sources is " \
               + str( len( filenames ) ) + ", should be " \
               + str( sourceNum ) )
        check = False
        
    # Check that all of the source positions are unique

    sourcePos = []

    for fn in filenames:

        with h5py.File( fn, "r" ) as dataFile:

            sourcePos.append( str( dataFile.visit( lambda x: dataFile[x].keys() if "conf" in x else None ) ) )

    for src1 in sourcePos:
        
        matches = 0

        for src2 in sourcePos:

            if src1 == src2:
                matches += 1

        if matches > 1:
            print( "check_sources: source positions are not all unique" )
            check = False
                
    # Check that all of the secondary groups are actually source dirctories

    # source position format for st < 100
    srcPosFormat2 = "\[u'sx[0-9]{2}sy[0-9]{2}sz[0-9]{2}st[0-9]{2}'\]"
    # source position format for st => 100
    srcPosFormat3 = "\[u'sx[0-9]{2}sy[0-9]{2}sz[0-9]{2}st[0-9]{3}'\]"

    for src in sourcePos:
        # if the source position does not match either format
        if re.match( srcPosFormat2, src ) is None and re.match( srcPosFormat3, src ) is None:
            print( "check_sources: second-level group " + src + " not of the proper form" )
            check = False

    return check


def jackknifeBin( vals, binSize, ibin ):

    return np.average( np.vstack( ( vals[ : ibin * binSize, \
                                          ... ], \
                                    vals[ ( ibin + 1 ) * binSize :, \
                                          ... ] ) ), \
                       axis=0 )


def jackknifeBinSubset( vals, binSize, bin_glob ):

    assert len( vals ) % binSize == 0, "Number of configurations " \
        + str( len( vals ) ) + " not evenly divided by bin size " \
        + str( binSize ) + " (functions.jackknifeBinSubset).\n"

    binNum_loc = len( bin_glob )

    vals_jk = initEmptyList( binNum_loc, 1 )

    for b in range( binNum_loc ):        

        vals_jk[ b ] = jackknifeBin( vals, binSize, bin_glob[ b ] )

    return np.array( vals_jk )
        

def jackknife( vals, binSize ):

    configNum = len( vals )

    assert configNum % binSize == 0, "Number of configurations " \
        + str( configNum ) + " not evenly divided by bin size " \
        + str( binSize ) + " (functions.jackknife).\n"

    binNum = configNum // binSize

    vals_jk = initEmptyList( binNum, 1 )

    for b in range( binNum ):
        
        vals_jk[ b ] = jackknifeBin( vals, binSize, b )

    return np.array( vals_jk )
