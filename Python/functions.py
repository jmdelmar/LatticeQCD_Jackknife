import math
import h5py
import numpy as np
import re
from scipy.optimize import leastsq, fmin, minimize
#from scipy.optimize import differential_evolution
from os import listdir as ls
from glob import glob

#def twoStateFitCurve_constTs( a00, a01, a11, E0, E1 ):


#def twoStateFitCurve_constTi( ):


def twoStateFit( twop, twop_err, twop_rangeStart, twop_rangeEnd, threep, threep_err, threep_neglect ):

    # twop[ b, t ]
    # twop_err[ t ]

    # threep[ ts ][ b, t ]
    # threep_err[ ts ][ t ]

    fit = []

    chi_sq = []

    # Check that number of bins is the same for all values of tsink

    binNum = []

    tsinkNum = len( threep )

    for ts in range( tsinkNum ):
 
        binNum.append( threep[ ts ].shape[ 0 ] )

        assert binNum[ ts ] == binNum [ 0 ], "Number of bins not the same for every value of tsink."

    for b in range( binNum[ 0 ] ):

        a00 = 1.0 
        a01 = 1.0 
        a11 = 1.0 
        c0 = 1.0
        c1 = 1.0 
        E0 = 0.1 
        E1 = 0.01

        fitParams = np.array( [ a00, a01, a11, c0, c1, E0, E1 ] )

        # ti[ts][t]

        ti = []

        # tsink[ts]

        tsink = [] 

        # twop_cp

        twop_cp = twop[ b, twop_rangeStart : twop_rangeEnd + 1 ]

        twop_tsink = np.array( range( twop_rangeStart, twop_rangeEnd + 1 ) )

        twop_err_cp = twop_err[ twop_rangeStart : twop_rangeEnd + 1 ]

        #print "twop: " + str( twop[ b, : ] )
        
        #print "twop_cp: " + str( twop_cp )

        #print "twop_tsink: " + str( twop_tsink )

        twop_err_cp = []

        threep_cp = []

        for ts in range( tsinkNum ):

            tNum = threep[ ts ].shape[ -1 ] - 2 * threep_neglect

            #print "tNum: " + str( tNum )
            """
            ti.append( np.array( range( tNum ) ) )

            tsink.append( tNum - 1 )
            """
            ti.append( np.array( range( threep_neglect, threep_neglect + tNum ) ) )

            tsink.append( tNum - 1 + 2 * threep_neglect )

            #threep_cp[ ts ][ ti ]

            threep_cp.append( threep[ ts ][ b, threep_neglect : threep_neglect + tNum ] )

            #print "threep: " + str( threep )

            #print "threep_cp: " + str( threep_cp[ -1 ] )

        ti = np.array( ti )

        #print "ti: " + str( ti )

        tsink = np.array( tsink )

        #print "tsink: " + str( tsink )
        
        twop_cp = np.array( twop_cp )

        #fit.append( leastsq( twoStateErrorFunction, fitParams, \
        #                     args = ( ti, tsink, twop_cp, threep_cp ) )[0] )

        min = minimize( twoStateErrorFunction, fitParams, \
                              args = ( twop_tsink, ti, tsink, twop_cp, twop_err_cp, threep_cp, threep_err ), \
                              method='Nelder-Mead', jac = False, tol=1e-9, \
                              options = {'maxiter':100000} )

        fit.append( min.x )

        chi_sq.append( min.fun )

    # End loop over bins

    return np.array( fit ), np.array( chi_sq )


def twoStateErrorFunction( fitParams, twop_tsink, ti, tsink, twop, twop_err, threep, threep_err ):

    a00 = fitParams[ 0 ]
          
    a01 = fitParams[ 1 ]
          
    a11 = fitParams[ 2 ]
          
    c0 = fitParams[ 3 ]
          
    c1 = fitParams[ 4 ]
        
    E0 = fitParams[ 5 ]
                
    E1 = fitParams[ 6 ]

    # twopErr[ ts ]

    #twopErr = np.array( ( ( twoStateTwop( twop_tsink, c0, c1, E0, E1 ) \
    #                        - twop ) / twop ) ** 2 )

    #twopErr = np.array( ( ( twoStateTwop( tsink, c0, c1, E0, E1 ) \
    #                        - twop ) / twop_err ) ** 2 )

    #print tsink

    #print twoStateTwop( tsink, c0, c1, E0, E1 )

    #print twop_tsink

    #print twoStateTwop( twop_tsink, c0, c1, E0, E1 )

    #print twop

    twopErr = np.array( ( twoStateTwop( twop_tsink, c0, c1, E0, E1 ) \
                          - twop ) ** 2 )

    # threepErr[ ts ][ ti ]

    threepErr = []

    for ti_ts, ts, threep_ts, threep_err_ts in zip( ti, tsink, threep, threep_err ):

        for t, threep_ti, threep_err_ti in zip( ti_ts, threep_ts, threep_err_ts ):

            #print "ti: " + str(t) + ", ts: " + str(ts) + ", a00: " + str(a00) + ", a01: " + str(a01) + ", a11: " + str(a11) + ", c0: " + str(c0) + ", c1: " + str(c1) + ", E0: " + str(E0) + ", E1: " + str(E1)

            #print "data: " + str(threep_ti)

            #print "function: " + str(twoStateThreep( t, ts, a00, a01, a11, E0, E1 ) )

            #threepErr.append( ( ( twoStateThreep( t, ts, a00, a01, a11, E0, E1 ) \
            #                      - threep_ti ) / threep_ti ) ** 2 )

            #threepErr.append( ( ( twoStateThreep( t, ts, a00, a01, a11, E0, E1 ) \
            #                      - threep_ti ) / threep_err_ti ) ** 2 )

            threepErr.append( ( twoStateThreep( t, ts, a00, a01, a11, E0, E1 ) \
                                - threep_ti ) ** 2 )

    #print np.concatenate( ( twopErr, threepErr ) )

    return np.sum( np.concatenate( ( twopErr, threepErr ) ) )


def twoStateThreep( ti, tsink, a00, a01, a11, E0, E1 ):

    return a00 * np.exp( -E0 * tsink ) \
        + a01 * np.exp( -E0 * ( tsink - ti ) - E1 * ti ) \
        + a01 * np.exp( -E1 * ( tsink - ti ) - E0 * ti ) \
        + a11 * np.exp( -E1 * tsink )


def twoStateTwop( tsink, c0, c1, E0, E1 ):

    return c0 * np.exp( -E0 * tsink ) \
        + c1 * np.exp( -E1 * tsink )


def twopFit( twop, fitStart, fitEnd ):

    # twop[ b, t ]

    fit = []

    # Check that number of bins is the same for all values of tsink

    binNum = twop.shape[ 0 ]

    print fitStart

    print fitEnd

    print range(fitStart,fitEnd+1)

    t = np.array( range( fitStart, fitEnd + 1 ) )

    for b in range( binNum ):

        G = 0.1 
        E = 0.1 
        
        fitParams = np.array( [ G, E ] )

        print t

        print twop[ b, : ]

        print twop[ b, fitStart : fitEnd + 1 ]

        fit.append( minimize( twopExpErrFunction, fitParams, \
                              args = ( t, twop[ b, fitStart : fitEnd + 1 ] ), \
                              method='Nelder-Mead', jac = False, \
                              options = {'maxiter':100000} ).x )

    # End loop over bins

    return np.array( fit )


def twopExpErrFunction( fitParams, t, twop ):

    G = fitParams[ 0 ]
          
    E = fitParams[ 1 ]
        
    # twopErr[ t ]

    twopErr = np.array( ( ( twopExp( t, G, E, ) \
                            - twop ) ) ** 2 )

    return np.sum( twopErr )


def twopExp( t, G, E ):
    
    return G**2 / 2 / E * ( np.exp( -E * t ) + np.exp( -E * ( 64 - t ) ) )


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
