import numpy as np
import functions as fncs
from scipy.optimize import least_squares, fmin, minimize, differential_evolution

def twoStateFit( twop, twop_err, twop_rangeStart, twop_rangeEnd, \
                 threep, threep_err, threep_neglect, tsink ):

    # twop[ b, t ]
    # twop_err[ t ]

    # threep[ ts ][ b, t ]
    # threep_err[ ts ][ t ]

    fit = []

    chi_sq = []

    # Check that number of bins is the same for all values of tsink

    tsinkNum = len( tsink )

    assert tsinkNum == len( threep ), \
        "Number of tsink's does not match " \
        + "number of three-point function datasets."

    # twop_avg[t]

    twop_avg = np.average( twop, axis=0 )[ twop_rangeStart : \
                                           twop_rangeEnd + 1 ]

    # threep_avg[ts][t]

    threep_avg = fncs.initEmptyList( tsinkNum, 1 )

    # ti[ts][t]

    ti = fncs.initEmptyList( tsinkNum, 1 )

    binNum = threep[ 0 ].shape[ 0 ]

    for ts in range( tsinkNum ):
 
        assert threep[ ts ].shape[ 0 ] == binNum, \
            "Number of bins not the same for " \
            + "every value of tsink."

        ti[ ts ] = np.array( range( threep_neglect, \
                                    tsink[ ts ] + 1 - threep_neglect ) )
    
        threep_avg[ ts ] = np.average( threep[ ts ], axis=0 )[ ti[ ts ][ 0 ] \
                                : ti[ ts ][ -1 ] + 1 ]

    tsink_twop = np.array( range( twop_rangeStart, twop_rangeEnd + 1 ) )

    # Find fit parameters of mean values to use as initial guess
    """
    a00 = 1.0
    a01 = 1.0
    a11 = 1.0
    c0 = 1.0
    c1 = 1.0
    E0 = 0.5
    E1 = 0.1
    """
    a00 = (-1, 1)
    a01 = (-1, 1)
    a11 = (-1, 1)
    c0 = (-0.1, 0.1)
    c1 = (-0.1, 0.1)
    E0 = (0, 1)
    E1 = (0, 0.1)

    fitParams = np.array( [ a00, a01, a11, c0, c1, E0, E1 ] )
    """
    min_avg = minimize( twoStateErrorFunction, fitParams, \
                        args = ( tsink_twop, ti, tsink, \
                                 twop_avg, twop_err, \
                                 threep_avg, threep_err ), \
                        method='Nelder-Mead' )
    #method='Powell' )
    """
    min_avg = differential_evolution( twoStateErrorFunction, fitParams, \
                                      args = ( tsink_twop, ti, tsink, \
                                               twop_avg, twop_err, \
                                               threep_avg, threep_err ) )
    """
    min_avg = least_squares( twoStateErrorFunction, fitParams, \
                             args = ( tsink_twop, ti, tsink, \
                                      twop_avg, twop_err, \
                                      threep_avg, threep_err ), \
                             method="lm" )
    """
    a00 = [ min_avg.x[ 0 ] - 0.1, min_avg.x[ 0 ] + 0.1 ]
          
    a01 = [ min_avg.x[ 1 ] - 0.1, min_avg.x[ 1 ] + 0.1 ]
          
    a11 = [ min_avg.x[ 2 ] - 0.1, min_avg.x[ 2 ] + 0.1 ]
          
    c0 = [ min_avg.x[ 3 ] - 0.01, min_avg.x[ 3 ] + 0.01 ]
          
    c1 = [ min_avg.x[ 4 ] - 0.01, min_avg.x[ 4 ] + 0.01 ]
        
    E0 = [ min_avg.x[ 5 ] - 0.1, min_avg.x[ 5 ] + 0.1 ]
                
    E1 = [ min_avg.x[ 6 ] - 0.01, min_avg.x[ 6 ] + 0.01 ]

    fitParams = np.array( [ a00, a01, a11, c0, c1, E0, E1 ] )

    #fitParams = min_avg.x

    for b in range( binNum ):

        # twop_cp

        twop_cp = twop[ b, twop_rangeStart : twop_rangeEnd + 1 ]

        #print( "twop: " + str( twop[ b, : ] ) )
        
        #print( "twop_cp: " + str( twop_cp ) )

        #print "tsink_twop: " + str( tsink_twop )

        threep_cp = fncs.initEmptyList( tsinkNum, 1 )

        for ts in range( tsinkNum ):

            #threep_cp[ ts ][ ti ]

            threep_cp[ ts ] = threep[ ts ][ b, ti[ ts ][ 0 ] \
                                : ti[ ts ][ -1 ] + 1 ]
            
            #print( "threep: " + str( threep[ts][b,:] ) )

            #print( "threep_cp: " + str( threep_cp[ -1 ] ) )

        #print "ti: " + str( ti )

        #print "tsink: " + str( tsink )
        
        #fit.append( leastsq( twoStateErrorFunction, fitParams, \
        #                     args = ( ti, tsink, twop_cp, threep_cp ) )[0] )
        """
        min = least_squares( twoStateErrorFunction, fitParams, \
                             args = ( tsink_twop, ti, tsink, \
                                      twop_cp, twop_err, \
                                      threep_cp, threep_err ), \
                             method="lm" )
        """
        min = differential_evolution( twoStateErrorFunction, fitParams, \
                                      args = ( tsink_twop, ti, tsink, \
                                               twop_cp, twop_err, \
                                               threep_cp, threep_err ) )
        """
        min = minimize( twoStateErrorFunction, fitParams, \
                        args = ( tsink_twop, ti, tsink, \
                        twop_cp, twop_err, \
                        threep_cp, threep_err ), \
                        method='Nelder-Mead' )
        #method='Powell' )
        """
        fit.append( min.x )

        #chi_sq.append( min.cost )
        chi_sq.append( min.fun )

    # End loop over bins

    return np.array( fit ), np.array( chi_sq )


def twoStateErrorFunction( fitParams, tsink_twop, ti, tsink, twop, twop_err, threep, threep_err ):

    a00 = fitParams[ 0 ]
          
    a01 = fitParams[ 1 ]
          
    a11 = fitParams[ 2 ]
          
    c0 = fitParams[ 3 ]
          
    c1 = fitParams[ 4 ]
        
    E0 = fitParams[ 5 ]
                
    E1 = fitParams[ 6 ]

    #print( "a00: " + str(a00) + ", a01: " + str(a01) + ", a11: " + str(a11) + ", c0: " + str(c0) + ", c1: " + str(c1) + ", E0: " + str(E0) + ", E1: " + str(E1) )

    # twopErr[ ts ]

    #print( "tsink_twop: " + str(tsink_twop) )

    #print( "data: " + str(twop) )

    #print( "function: " + str( twoStateTwop( tsink_twop, c0, c1, E0, E1 ) ) )
    """
    twopErr = np.array( twoStateTwop( tsink_twop, c0, c1, E0, E1 ) \
                        - twop )
    """
    twopErr = np.array( ( twoStateTwop( tsink_twop, c0, c1, E0, E1 ) \
                          - twop ) ** 2 )
    """
    twopErr = np.array( ( ( twoStateTwop( tsink_twop, c0, c1, E0, E1 ) \
                            - twop ) / twop ) ** 2 )

    twopErr = np.array( ( ( twoStateTwop( tsink, c0, c1, E0, E1 ) \
                            - twop ) / twop_err ) ** 2 )
    """
    # threepErr[ ts ][ ti ]

    threepErr = []

    for ti_ts, ts, threep_ts, threep_err_ts in zip( ti, tsink, threep, threep_err ):

        for t, threep_ti, threep_err_ti in zip( ti_ts, threep_ts, threep_err_ts ):

            #print( "ti: " + str(t) + ", ts: " + str(ts)  )

            #print( "data: " + str(threep_ti) )

            #print( "function: " + str(twoStateThreep( t, ts, a00, a01, a11, E0, E1 ) ) )
            """
            threepErr.append( twoStateThreep( t, ts, a00, a01, a11, E0, E1 ) \
                              - threep_ti )
            """
            threepErr.append( ( twoStateThreep( t, ts, a00, a01, a11, E0, E1 ) \
                                - threep_ti ) ** 2 )
            """
            threepErr.append( ( ( twoStateThreep( t, ts, a00, a01, a11, E0, E1 ) \
                                  - threep_ti ) / threep_ti ) ** 2 )

            threepErr.append( ( ( twoStateThreep( t, ts, a00, a01, a11, E0, E1 ) \
                                  - threep_ti ) / threep_err_ti ) ** 2 )
            """

    #print( np.concatenate( ( twopErr, threepErr ) ) )

    #print( np.sum(np.concatenate( ( twopErr, threepErr ) ) ) )

    return np.sum( np.concatenate( ( twopErr, threepErr ) ) )
    
    #return np.concatenate( ( twopErr, threepErr ) )


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

    #print fitStart

    #print fitEnd

    #print range(fitStart,fitEnd+1)

    t = np.array( range( fitStart, fitEnd + 1 ) )

    for b in range( binNum ):

        G = 0.1 
        E = 0.1 
        
        fitParams = np.array( [ G, E ] )

        #print t

        #print twop[ b, : ]

        #print twop[ b, fitStart : fitEnd + 1 ]

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


"""
def threepFit( threep, fitStart, fitEnd ):

    # threep[ b, t ]

    fit = []

    # Check that number of bins is the same for all values of tsink

    binNum = threep.shape[ 0 ]

    #print fitStart

    #print fitEnd

    #print range(fitStart,fitEnd+1)

    t = np.array( range( fitStart, fitEnd + 1 ) )

    for b in range( binNum ):

        G = 0.1 
        E = 0.1 
        
        fitParams = np.array( [ G, E ] )

        #print t

        #print threep[ b, : ]

        #print threep[ b, fitStart : fitEnd + 1 ]

        fit.append( minimize( threepExpErrFunction, fitParams, \
                              args = ( t, threep[ b, fitStart : fitEnd + 1 ] ), \
                              method='Nelder-Mead', jac = False, \
                              options = {'maxiter':100000} ).x )

    # End loop over bins

    return np.array( fit )


def threepExpErrFunction( fitParams, t, threep ):

    G = fitParams[ 0 ]
          
    E = fitParams[ 1 ]
        
    # threepErr[ t ]

    threepErr = np.array( ( ( threepExp( t, G, E, ) \
                            - threep ) ) ** 2 )

    return np.sum( threepErr )


def threepExp( t, G, E ):
    
    return G**2 / 2 / E * ( np.exp( -E * t ) + np.exp( -E * ( 64 - t ) ) )
"""

