import math
import h5py
import numpy as np
import re
from os import listdir as ls
from glob import glob

# Calculates the jackknife error from the standard deviation.
# Can be given any keyword arguments accepted by numpy.std(),
# otherwise, calculates the error along the first axis.

# vals: Values for which to calculate jackknife error
# binNum: Number of bins
# kwargs (Optional): Keyword arguments to be passed to numpy.std()

def calcError( vals, binNum, **kwargs ):

    # Err = ( N - 1 ) / sqrt( N ) * stdev

    if kwargs:

        return np.std( vals, **kwargs ) \
            * float( binNum - 1 ) \
            * float( binNum ) ** -0.5

    else:

        return np.std( vals, axis=0 ) \
            * float( binNum - 1 ) \
            * float( binNum ) ** -0.5


# Return either "+" or "-" as a string, depending on the sign
# of the argument given.

# val: Values whose sign is checked

def signToString( val ):

    if val >= 0:

        return "+"

    else:

        return "-"


# Combines the different final momentum three-point functions for <x>. 
# It is obsolete, however, because the combination is just an average.
# Will keep for now  in case a different combination is needed for a
# future quantity.

# threep: Three-point functions to be combined 
# momSq: Value of mometum squared which determines which momenta we have

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


# Initializes a list of empty lists of the given order with the same
# dimensions as the given list.

# list_in: List which determines dimensions of list_out
# list_out_order: Order of list_out

def initEmptyList( list_in, list_out_order ):

    list_out = []

    if list_out_order == 1:

        # list_out[ i ]

        list_out = [ [] for i in range( list_in ) ]

    elif list_out_order == 2:

        # list_out[ i ][ j ]

        list_out = [ [ [] for j in range( len( list_in[i] ) ) ] \
                     for i in range( len( list_in ) ) ]

    elif list_out_order == 3:

        # list_out[ i ][ j ][ k ]

        list_out = [ [ [ [] for k in range( len( list_in[i][j] ) ) ] \
                       for j in range( len( list_in[i] ) ) ] \
                     for i in range( len( list_in ) ) ]

    else:

        print( "Error(initEmptyList): list order not supported.\n" )

    return list_out


# Averages data with the points opposite them across the center 
# of the data array.

# data: Data to be folded

def fold( data ):

    timestepNum = data.shape[ -1 ]

    out = np.zeros( data.shape[ :-1 ] + ( timestepNum // 2 + 1, ) )

    # First data point is unchanged

    out[ ..., 0 ] = data[ ..., 0 ]

    for t in range( 1, timestepNum // 2 ):
        
        # Average opposite points

        out[ ..., t ] = ( data[ ..., t ] + data[ ..., -t ] ) / 2

    # Data point at center is unchanged

    out[ ..., timestepNum // 2 ] = data[ ..., timestepNum // 2 ]

    return out


# Reads a configuration list from file if given, else writes a list from
# directory names in given directory.

# configListFilename: Name of configuration list file, if None, will 
#                     write list from directory names in directory
#                     given as 2nd argument
# configDir: Direcoty containing directories for each configuration

def getConfigList( configListFilename, configDir ):

    if configListFilename:

        # Check that configuration list exists
        
        if glob( configListFilename ):

            with open( configListFilename, "r" ) as configFile:
                
                # Read configuration list
                
                configList = configFile.read().splitlines()

                #print( "Configuration list read" )

        else:

            print( "WARNING: Given configuration list does not exist. " \
                + "Will use all configurations in configuration directory." )
            
            configList = ls( configDir )

            configList = sorted( configList )

            with open( configListFilename, "w" ) as configFile:

                for config in configList:

                    configFile.write( str( config ) + "\n" )

            #print( "Configuration list written" )

    else:

        configList = ls( configDir )

        configList = sorted( configList )

        configFilename = "./confs.txt"

        # Check that ./confs.txt does not already exist

        if glob( configFilename ):

            print( "WARNING: Configuration list already exists in this directory. " \
                + "Will not overwrite." )

        else:

            # Write configuration list

            with open( configFilename, "w" ) as configFile:

                for config in configList:

                    configFile.write( str( config ) + "\n" )

            #print( "Configuration list written" )

    #print( "Number of configurations: " + str( len( configList ) ) )

    return configList


# Calculates Q^2 from a given list of Q's and determines the 
# position of the first and last Q for each Q^2.

# momLists: Momentum lists. Should either identicle.

def processMomList( momLists ):

    # Check that momenta lists are the same across configurations

    momList_0 = momLists[ 0 ].flat

    for ml in momLists[ 1: ]:

        for i in range( ml.size ):

                assert ml.flat[ i ] == momList_0[ i ], \
                    "Momenta lists in configuration " + configList[ c ] \
                    + " do not match"

    momList = momLists[ 0 ]

    # Get indices where each Q^2 begins and ends

    Qsq_start = []

    Qsq_end = []

    Qsq = np.round( ( np.apply_along_axis( np.linalg.norm, 1, \
                                           np.array( momList ) ) ) ** 2 )

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

    return np.array( Qsq, dtype=int ), \
        np.array( Qsq_start ), \
        np.array( Qsq_end )


# Averages over equal Q^2 for numpy array whose last dimension is Q and 
# returns averaged data as a numpy array whose first dimension is Q^2

# data: Data to be averaged
# Qsq_start: List of the starting index for each Q^2 to be averaged over
# Qsq_end: List of the ending index for each Q^2 to be averaged over

def averageOverQsq( data, Qsq_start, Qsq_end ):

    QsqNum = len( Qsq_start )

    assert len( Qsq_end ) == QsqNum, "Error( averageOverQsq ): " \
        + "Qsq_start and Qsq_end have different lengths " \
        + QsqNum + " and " + len( Qsq_end ) + "."

    avg = initEmptyList( QsqNum, 1 )

    for q in range( QsqNum ):

        avg[ q ] = np.average( data[ ..., start[ q ] : end[ q ] + 1 ], \
                               axis=-1 )
            
    return np.array( avg )


# Checks that there are the correct number of source files, that the 
# source positions are unique, and that the secondary groups are all
# source groups. Returns True if all checks pass and False if a 
# check does not pass.

# filenames: Names of files to be checked
# sourceNum: Expected number of sources

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
    # source position format for st >= 100
    srcPosFormat3 = "\[u'sx[0-9]{2}sy[0-9]{2}sz[0-9]{2}st[0-9]{3}'\]"

    for src in sourcePos:
        # if the source position does not match either format
        if re.match( srcPosFormat2, src ) is None and re.match( srcPosFormat3, src ) is None:
            print( "check_sources: second-level group " + src + " not of the proper form" )
            check = False

    return check


# Average over configurations, excluding one bin.

# vals: Values to be averaged
# binSize: Size of bin to be exclude
# ibin: index of bin to be exclude

def jackknifeBin( vals, binSize, ibin ):

    return np.average( np.vstack( ( vals[ : ibin * binSize, \
                                          ... ], \
                                    vals[ ( ibin + 1 ) * binSize :, \
                                          ... ] ) ), \
                       axis=0 )


# Perform jackknife averaging over a subset of bins. 
# Used so that multiple subsets can be averaged in parallel.

# vals: Values to be averaged
# binSize: Size of bin to be excluded
# bin_glob: Global indices for subset of bins

def jackknifeBinSubset( vals, binSize, bin_glob ):

    assert len( vals ) % binSize == 0, "Number of configurations " \
        + str( len( vals ) ) + " not evenly divided by bin size " \
        + str( binSize ) + " (functions.jackknifeBinSubset).\n"

    # Local number of bins in subset

    binNum_loc = len( bin_glob )

    vals_jk = initEmptyList( binNum_loc, 1 )

    for b in range( binNum_loc ):

        vals_jk[ b ] = jackknifeBin( vals, binSize, bin_glob[ b ] )

    return np.array( vals_jk )


# Perform jackknife averaging over all bins.        

# vals: Values to be averaged
# binSize: Size of bin to be excluded

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
