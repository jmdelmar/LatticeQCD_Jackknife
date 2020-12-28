import math
import h5py
import numpy as np
import re
from os import listdir as ls
from glob import glob
import mpi_functions as mpi_fncs


def particleList():

    return [ "nucleon", "pion", "kaon" ]

def dataFormatList():

    return [ "cpu", "gpu", "ASCII" ]

def mellinMomentList():

    return [ "avgX", "avgX2", "avgX3" ]

def GEList():

    return [ "GE0_local", "GE0_noether" ]

def formFactorList():

    return [ "GE_GM", "A20_B20" ]


def zipXandIndex( x ):

    return zip( x, range( len( x ) ) )


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


# Set data format for twop and threep files 
# based on particle and p^2

def setDataFormat( particle, pSq ):

    dataFormat_threep = "gpu" if pSq == 0 else "cpu"

    if particle == "pion":

        if pSq == 0:
            
            dataFormat_twop = [ "gpu" ]

        else:

            dataFormat_twop = [ "cpu" ]

    elif particle == "kaon":

        if pSq == 0:

            dataFormat_twop = [ "gpu", "cpu", "cpu" ]
            
        else:

            dataFormat_twop = [ "cpu", "cpu" ]

    else:

        dataFormat_twop = [ "gpu" ]

    return dataFormat_twop, dataFormat_threep


# Set template for booste twop files
# based on particle and p^2

def setTwopBoostTemplate( particle, pSq, twop_template ):

    if particle == "pion":

        if pSq == 0:
            
            twop_boost_template = None

        else:

            twop_boost_template = twop_template

    elif particle == "kaon":

        if pSq == 0:

            twop_boost_template = None

        else:

            twop_boost_template = [ twop_template[ 1 ], twop_template[ 0 ] ]

    else:

        twop_boost_template = None

    return twop_boost_template


# Set smear strings based based on particle and p^2

def setSmearString( particle, pSq ):

    smear_str_template = "_gN{}a0p2"

    if particle == "pion":

        if pSq == 0:
            
            # [ "" ]
            smear_str_list = [ "" ]
            # None
            smear_str_list_boost = None

        else:

            # [ gN50a0p2 ]
            smear_str_list = [ smear_str_template.format( 50 ) ]
            # [ gN50a0p2 ]
            smear_str_list_boost = smear_str_list

    elif particle == "kaon":

        if pSq == 0:

            # [ "", gN40a0p2_gN50a0p2, gN50a0p2_gN40a0p2 ]
            smear_str_list = [ "",
                               smear_str_template.format( 40 ) 
                               + smear_str_template.format( 50 ),
                               smear_str_template.format( 50 ) 
                               + smear_str_template.format( 40 ) ]        
            smear_str_list_boost = None

        else:

            # [ gN40a0p2 ]
            smear_str_list = [ smear_str_template.format( 40 ) ]
            # [ gN40a0p2_gN50a0p2, gN40a0p2 ]
            smear_str_list_boost = [ smear_str_template.format( 40 ) 
                                     + smear_str_template.format( 50 ),
                                     smear_str_template.format( 40 ) ]        
            
    else:

        # ""
        smear_str_list = ""

    smearNum = len( smear_str_list )

    if pSq > 0:

        smearNum_boost = len( smear_str_list_boost )

    else:

        smearNum_boost = None

    return smear_str_list, smear_str_list_boost, smearNum, smearNum_boost


# Return either "+" or "-" as a string, depending on the sign
# of the argument given.

# val: Values whose sign is checked

def signToString( val ):

    if val >= 0:

        return "+"

    else:

        return ""


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

    return np.array( configList )


# Calculates Q^2 from a given list of Q's and determines the 
# position of the first and last Q for each Q^2.

# momLists: Momentum lists. Should either identicle.

def processMomList( QLists ):

    if QLists.ndim > 2:

        if len( QLists ) > 1:

            # Check that momenta lists are the same across configurations

            Q_0 = QLists[ 0 ].flat

            for ml in QLists[ 1: ]:

                for i in range( ml.size ):

                    assert ml.flat[ i ] == Q_0[ i ], \
                        "Momenta lists do not match."

        Q = QLists[ 0 ]

    else:

        Q = QLists

    # List of indices where each Q^2 begins and ends

    Qsq_start = []

    Qsq_end = []

    # List of indices where Q^2 value is in Q

    Qsq_where = []

    Qsq = np.round( ( np.apply_along_axis( np.linalg.norm, 1, \
                                           np.array( Q ) ) ) ** 2 )

    q_last = -1
    iq_where = -1

    for q in Qsq:

        if q != q_last:

            assert q > q_last, "Momentum list not in assending order."

            Qsq_start.append( np.where( Qsq == q )[0][0] )

            Qsq_end.append( np.where( Qsq == q )[0][-1] )

            q_last = q

            iq_where += 1
            
        # End if q > q_last

        Qsq_where.append( iq_where )

    # End loop over Q^2

    # Remove duplicate Q^2's

    Qsq = sorted( list( set( Qsq ) ) )

    return np.array( Q, dtype=int ), \
        np.array( Qsq, dtype=int ), \
        np.array( Qsq_start ), \
        np.array( Qsq_end ), \
        np.array( Qsq_where )


# Averages over equal Q^2 for numpy array whose last dimension is Q and 
# returns averaged data as a numpy array whose first dimension is Q^2

# data: Data to be averaged with Q in the second to last dimension
# Qsq_start: List of the starting index for each Q^2 to be averaged over
# Qsq_end: List of the ending index for each Q^2 to be averaged over

def averageOverQsq( data, Qsq_start, Qsq_end ):

    QsqNum = len( Qsq_start )

    assert len( Qsq_end ) == QsqNum, "Error( averageOverQsq ): " \
        + "Qsq_start and Qsq_end have different lengths " \
        + QsqNum + " and " + len( Qsq_end ) + "."

    avg = np.zeros( data.shape[ :-2 ] + ( QsqNum, data.shape[ -1 ] ) )

    for q in range( QsqNum ):

        avg[ ..., q, : ] \
            = np.average( data[ ...,
                                Qsq_start[ q ] : Qsq_end[ q ] + 1, : ],
                          axis=-2 )

    return avg


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


def setFlavorStrings( particle, dataFormat ):

    # Check inputs

    assert particle in particleList(), \
        "Particle " + particle + " is not supported."

    assert dataFormat in dataFormatList(), \
        "Data format " + dataFormat + " is not supported."

    # Set flavors

    if particle == "nucleon":

        if dataFormat == "cpu" or dataFormat == "ASCII":

            flavor = [ "up", "dn" ]

        elif dataFormat == "gpu":

            flavor = [ "up", "down" ]

    elif particle == "pion":

        flavor = [ "up" ]

    elif particle == "kaon":

        flavor = [ "up", "strange" ]

    flavorNum = len( flavor )

    return flavor, flavorNum


def setRatioNumber( formFactor, particle, mpi_info ):

    if formFactor == "GE_GM":

        if particle == "nucleon":

            return 10

        else: # meson

            return 4

    elif formFactor == "A20_B20":

        return 7
        
    else:

        mpi_fncs.mpiPrintError( "Error (functions.setRatioNumber): " \
                           + "form factor " \
                           + "{} not supported.".format( formFactor ),
                           mpi_info )


def setCurrentNumber( formFactor, mpi_info ):

    if formFactor == "GE_GM":

        return 4

    elif formFactor == "A20_B20":

        return 16
        
    else:

        mpi_fncs.mpiPrintError( "Error (functions.setCurrentNumber): " \
                           + "form factor " \
                           + "{} not supported.".format( formFactor ),
                           mpi_info )


# Average over configurations, excluding one bin.

# vals: Values to be averaged
# binSize: Size of bin to be exclude
# ibin: index of bin to be exclude

def jackknifeBin( vals, binSize, ibin, **kwargs ):

    if "comm" in kwargs:

        mpi_fncs.mpiPrint(binSize,kwargs["comm"])
        mpi_fncs.mpiPrint(ibin,kwargs["comm"])
        mpi_fncs.mpiPrint(vals.shape,kwargs["comm"])
        mpi_fncs.mpiPrint(vals,kwargs["comm"])
        mpi_fncs.mpiPrint(vals[ : ibin * binSize, \
                                ... ], \
                          kwargs["comm"])
        mpi_fncs.mpiPrint(vals[ ( ibin + 1 ) * binSize :, \
                                ... ], \
                          kwargs["comm"])
        mpi_fncs.mpiPrint(np.average( np.vstack( ( vals[ : ibin * binSize, \
                                          ... ], \
                                    vals[ ( ibin + 1 ) * binSize :, \
                                          ... ] ) ), \
                                      axis=0 ), \
                          kwargs["comm"] )

    return np.average( np.vstack( ( vals[ : ibin * binSize, \
                                          ... ], \
                                    vals[ ( ibin + 1 ) * binSize :, \
                                          ... ] ) ), \
                       axis=0 )


# Perform jackknife averaging over a subset of bins. 
# Used so that multiple subsets can be averaged in parallel.

# vals: Values to be averaged with configurations in first dimension
# binSize: Size of bin to be excluded
# bin_glob: Global indices for subset of bins

def jackknifeBinSubset( vals, binSize, bin_glob, **kwargs ):

    assert len( vals ) % binSize == 0, "Number of configurations " \
        + str( len( vals ) ) + " not evenly divided by bin size " \
        + str( binSize ) + " (functions.jackknifeBinSubset).\n"

    if bin_glob.size != 0:

        # Local number of bins in subset

        binNum_loc = len( bin_glob )
        
        vals_jk = np.zeros( ( binNum_loc, ) + vals.shape[ 1: ] )

        for b in range( binNum_loc ):

            vals_jk[ b ] = jackknifeBin( vals, binSize, \
                                         bin_glob[ b ], \
                                         **kwargs)

    else:

        vals_jk = np.array( [] )

    return vals_jk


# Perform jackknife averaging over all bins.        

# vals: Values to be averaged with configurations in first dimension
# binSize: Size of bin to be excluded

def jackknife( vals, binSize ):

    configNum = vals.shape[ 0 ]

    assert configNum % binSize == 0, "Number of configurations " \
        + str( configNum ) + " not evenly divided by bin size " \
        + str( binSize ) + " (functions.jackknife).\n"

    binNum = configNum // binSize

    vals_jk = np.zeros( ( binNum, ) + vals.shape[ 1: ] )

    for b in range( binNum ):
        
        vals_jk[ b ] = jackknifeBin( vals, binSize, b )

    return vals_jk
