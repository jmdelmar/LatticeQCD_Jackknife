import math
import h5py
import numpy as np
import re
from scipy.optimize import leastsq
from os import listdir as ls
from glob import glob

def twoStateFit( twop, threep ):

    # twop[ ts ][ b ]

    # threep[ ts ][ b, t ]
    
    fit = []

    # Check that number of bins is the same for all values of tsink

    binNum = []

    for ts in range( len( threep ) ):
 
        binNum.append( threep[ ts ].shape[ 0 ] )

        assert binNum[ ts ] == binNum [ 0 ], "Number of bins not the same for every value of tsink."

    for b in range( binNum[ 0 ] ):

        # ti[ ts ][ t ], ...

        a00 = 1.0 
        a01 = 1.0 
        a11 = 1.0 
        c0 = 1.0 
        c1 = 1.0 
        E0 = 1.0 
        E1 = 1.0 

        fitParams = np.array( [ a00, a01, a11, c0, c1, E0, E1 ] )

        ti = []
        tsink = [] 
        twop_cp = [] 
        threep_cp = []

        for ts in range( len( threep ) ):
 
            tNum = threep[ ts ].shape[ -1 ]

            ti.append( np.array( range( tNum ) ) )

            tsink.append( ( tNum - 1 ) * np.ones( tNum ) )

            twop_cp.append( twop[ ts ][ b ] * np.ones( tNum ) )

            threep_cp.append( threep[ ts ][ b ] )

        fit.append( leastsq( twoStateErrorFunction, fitParams, \
                             args = ( ti[0], ti[1], tsink[0], tsink[1], twop_cp[0], twop_cp[1], threep_cp[0], threep_cp[1] ) )[0] )

    return fit


def twoStateThreep( ti, ts, a00, a01, a11, E0, E1 ):

    return a00 * np.exp( -E0 * ts ) \
        + a01 * np.exp( -E0 * ( ts - ti ) - E1 * ti ) \
        + a01 * np.exp( -E1 * ( ts - ti ) - E0 * ti ) \
        + a11 * np.exp( -E1 * ts )


def twoStateTwop( ts, c0, c1, E0, E1 ):

    return c0 * np.exp( -E0 * ts ) + c1 * np.exp( -E1 * ts )


#def twoStateErrorFunction( fitParams, ti, tsink, twop, threep ):

def twoStateErrorFunction( fitParams, ti0, ti1, tsink0, tsink1, twop0, twop1, threep0, threep1 ):

    a00 = fitParams[ 0 ]
          
    a01 = fitParams[ 1 ]
          
    a11 = fitParams[ 2 ]
          
    c0 = fitParams[ 3 ]
          
    c1 = fitParams[ 4 ]
        
    E0 = fitParams[ 5 ]
                
    E1 = fitParams[ 6 ]

    """
    for ts in range( len( threep ) ):

        err.append( twoStateTwop( tsink[ ts ], c0, c1, E0, E1 ) \
                    - twop[ ts ] )
        
        err.append( twoStateThreep( ti[ ts ], tsink[ ts ], a00, a01, a11, E0, E1 ) \
                    - threep[ ts ] )
    """
    err0 = twoStateTwop( tsink0, c0, c1, E0, E1 ) \
                - twop0
        
    err1 = twoStateThreep( ti0, tsink0, a00, a01, a11, E0, E1 ) \
                - threep0

    err2 = twoStateTwop( tsink1, c0, c1, E0, E1 ) \
                - twop1

    err3 = twoStateThreep( ti1, tsink1, a00, a01, a11, E0, E1 ) \
                - threep1

    return np.concatenate( ( err0, err1, err2, err3 ) )


def fold( data ):

    timestepNum = data.shape[ -1 ]

    out = np.zeros( data.shape[ :-1 ] + ( timestepNum / 2 + 1, ) )

    out[ ..., 0 ] = data[ ..., 0 ]

    for t in range( 1, timestepNum / 2 ):
        
        out[ ..., t ] = ( data[ ..., t ] + data[ ..., -t ] ) / 2

    out[ ..., timestepNum / 2 ] = data[ ..., timestepNum / 2 ]

    return out


def getConfigList( configListFilename, configDir ):

    if configListFilename:

        if glob( configListFilename ):

            with open( configListFilename, "r" ) as configFile:

                configList = configFile.read().splitlines()

            print "Configuration list read"

        else:

            print "WARNING: Given configuration does not exist. " \
                + "Will use all configurations in configuration directory."
            
            configList = ls( configDir )

            configList = sorted( configList )

            with open( configListFilename, "w" ) as configFile:

                for config in configList:

                    configFile.write( str( config ) + "\n" )

            print "Configuration list written"

    else:

        configList = ls( configDir )

        configList = sorted( configList )

        configFilename = "./confs.txt"

        if glob( configFilename ):

            print "WARNING: Configuration list already exists in this directory. " \
                + "Will not overwrite."

        else:

            with open( configFilename, "w" ) as configFile:

                for config in configList:

                    configFile.write( str( config ) + "\n" )

            print "Configuration list written"

    print "Number of configurations: " + str( len( configList ) )

    return configList


def processMomList( momLists ):

    # Check that momenta lists are the same across configurations

    momList_0 = momLists[ 0 ].flat

    for ml in momLists[ 1: ]:

        for i in range( ml.size ):

                assert ml.flat[ i ] == momList_0[ i ], \
                    "Momenta lists in configuration " + configList[ c ] \
                    + " do not match"

    momList = momLists[ 0 ]

    # Get indexes where each Q^2 begins and ends

    Qsq_start = []

    Qsq_end = []

    Qsq = np.round( ( np.apply_along_axis( np.linalg.norm, 1, np.array( momList ) ) ) ** 2 )
    
    #Qsq = ( np.apply_along_axis( np.linalg.norm, 1, np.array( momList ) ) ) ** 2

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

    #return np.array( Qsq ), np.array( Qsq_start ), np.array( Qsq_end )

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
        print "check_sources: number of sources is " + str( len( filenames ) ) + ", should be " + str( sourceNum )
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
