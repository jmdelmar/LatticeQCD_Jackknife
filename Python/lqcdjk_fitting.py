import numpy as np
import functions as fncs
from scipy.optimize import least_squares, fmin, minimize, differential_evolution

class lqcdjk_BadFitError(Exception):
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)


def mEffTwopFit( mEff, twop, rangeEnd, tsf ):

    binNum = mEff.shape[ 0 ]
    T = twop.shape[ -1 ]

    mEff_err = fncs.calcError( mEff, binNum )

    for mEff_rangeStart in range( 1, rangeEnd - 2 ):

        mEff_fit = np.zeros( binNum )

        for b in range( binNum ):

            mEff_fit[ b ] = np.polyfit( range( mEff_rangeStart, \
                                               rangeEnd + 1 ), \
                                        mEff[ b, \
                                              mEff_rangeStart \
                                              : rangeEnd + 1 ], 0, \
                                        w=mEff_err[ mEff_rangeStart \
                                                    : rangeEnd + 1 ] )

        # Average over bins

        mEff_fit_avg = np.average( mEff_fit )

        mEff_fit_err = fncs.calcError( mEff_fit, binNum )

        if tsf:

            for tsf_rangeStart in range( 1, rangeEnd - 2 ):

                # fitParams[ b, param ]

                fitParams, chiSq = twoStateFit_twop( twop, \
                                                     tsf_rangeStart, \
                                                     rangeEnd, T )

                E0_avg = np.average( fitParams[ :, 2 ] )
                E0_err = fncs.calcError( fitParams[ :, 2 ], binNum )

                # Check if the fit is good by comparing mass values
                # obtained from two-state and mEff fits

                relDiff = np.abs( mEff_fit_avg - E0_avg ) \
                          / ( 0.5 * ( mEff_fit_avg + E0_avg ) )

                if 0.5 * E0_err > relDiff \
                   and 0.5 * mEff_fit_err > relDiff:

                    results = ( fitParams, chiSq, mEff_fit, \
                                tsf_rangeStart, mEff_rangeStart )

                    return results

                # End if relDiff < dm/2
            # End loop over two-state fit start

        else: # One-state fit

            for twop_rangeStart in range( 1, rangeEnd - 2 ):

                fitParams, chiSq = oneStateFit_twop( twop, \
                                                     twop_rangeStart, \
                                                     rangeEnd, T )
                
                E_avg = np.average( fitParams[ :, 1 ], axis=0 )
                E_err = fncs.calcError( fitParams[ :, 1 ], binNum )

                relDiff = np.abs( mEff_fit_avg - E_avg ) \
                          / ( 0.5 * ( mEff_fit_avg + E_avg ) )
        
                if 0.5 * E_err > relDiff \
                   and 0.5 * mEff_fit_err > relDiff:

                    results = ( fitParams, chiSq, mEff_fit, \
                                twop_rangeStart, mEff_rangeStart )

                    return results

                # End if relDiff < dm/2
            # End loop over twop fit start
        # End if no two-state fit
    # End loop over effective mass fit start

    raise lqcdjk_BadFitError( "fitTwop() could not find a good fit with " \
                              + "given effective masses, two-point functions, " \
                              + "and range end." )

    return -1

def twoStateFit_twop( twop, twop_rangeStart, twop_rangeEnd, T ):

    # twop[ b, t ]
    # twop_err[ t ]

    twop_to_fit = np.concatenate( ( twop[ :, twop_rangeStart : \
                                          twop_rangeEnd + 1 ], \
                                    twop[ :, T - twop_rangeEnd : \
                                          T - twop_rangeStart + 1 ] ), \
                                  axis=1 )

    # fit[b]

    binNum = twop.shape[ 0 ]

    fit = fncs.initEmptyList( binNum, 1 )

    chi_sq = fncs.initEmptyList( binNum, 1 )

    # twop_avg[ts]

    twop_avg = np.average( twop_to_fit, axis=0 )

    twop_err = fncs.calcError( twop_to_fit, binNum )
    
    tsink = np.concatenate( ( range( twop_rangeStart, \
                                     twop_rangeEnd + 1 ), \
                              range( T - twop_rangeEnd, \
                                     T - twop_rangeStart + 1 ) ) )

    # Find fit parameters of mean values to use as initial guess

    c0 = 1.0
    c1 = 1.0
    E0 = 0.1
    E1 = 1.0

    fitParams = np.array( [ c0, c1, E0, E1 ] )

    leastSq_avg = least_squares( twoStateErrorFunction_twop, fitParams, \
                             args = ( tsink, T, twop_avg, twop_err ), \
                             method="lm" )

    fitParams = leastSq_avg.x

    for b in range( binNum ):

        leastSq = least_squares( twoStateErrorFunction_twop, fitParams, \
                             args = ( tsink, T, twop_to_fit[ b, : ], \
                                      twop_err ), \
                             method="lm" )

        fit[ b ] = leastSq.x

        chi_sq[ b ] = leastSq.cost

    # End loop over bins

    return np.array( fit ), np.array( chi_sq )


def twoStateFit_threep( threep, neglect, tsink, E0, E1, T ):

    # threep[ ts ][ b, t ]
    # threep_err[ ts ][ t ]

    tsinkNum = len( tsink )

    assert tsinkNum == len( threep ), \
        "Number of tsink's does not match " \
        + "number of three-point function datasets."

    threep_to_fit = fncs.initEmptyList( tsinkNum, 1 )

    # fit[b]

    binNum = threep[ 0 ].shape[ 0 ]

    fit = fncs.initEmptyList( binNum, 1 )

    chi_sq = fncs.initEmptyList( binNum, 1 )

    # threep_avg[ts][t]

    threep_avg = fncs.initEmptyList( tsinkNum, 1 )

    threep_err = fncs.initEmptyList( tsinkNum, 1 )

    E0_avg = np.average( E0 )
    E1_avg = np.average( E1 )

    # ti[ts][t]

    ti = fncs.initEmptyList( tsinkNum, 1 )

    for ts in range( tsinkNum ):
 
        assert threep[ ts ].shape[ 0 ] == binNum, \
            "Number of bins not the same for " \
            + "every value of tsink."

        threep_to_fit[ ts ] = np.concatenate( ( threep[ ts ][ :, neglect \
                                                              : tsink[ ts ] - neglect + 1 ], \
                                                threep[ ts ][ :, tsink[ ts ] + neglect + 5 \
                                                              : T - neglect - 5 + 1 ] ), \
                                              axis=1 )

        ti[ ts ] = np.concatenate( ( range( neglect, \
                                      tsink[ ts ] - neglect + 1 ), \
                                     range( tsink[ ts ] + neglect + 5, \
                                            T - neglect - 5 + 1 ) ) )
    
        threep_avg[ ts ] = np.average( threep_to_fit[ ts ], \
                                       axis=0 )
                           
        threep_err[ ts ] = np.std( threep_to_fit[ ts ], \
                                   axis=0 ) \
            * float( binNum - 1 ) / np.sqrt( float( binNum ) )

    # Find fit parameters of mean values to use as initial guess

    a00 = 1.0
    a01 = 1.0
    a11 = 1.0

    fitParams = np.array( [ a00, a01, a11 ] )

    leastSq_avg = least_squares( twoStateErrorFunction_threep, fitParams, \
                             args = ( ti, tsink, T, \
                                      threep_avg, threep_err, \
                                      E0_avg, E1_avg ), \
                             method="lm" )

    fitParams = leastSq_avg.x

    for b in range( binNum ):

        threep_cp = fncs.initEmptyList( tsinkNum, 1 )

        for ts in range( tsinkNum ):

            #threep_cp[ ts ][ ti ]

            threep_cp[ ts ] = threep_to_fit[ ts ][ b, : ]
            
        leastSq = least_squares( twoStateErrorFunction_threep, fitParams, \
                             args = ( ti, tsink, T, \
                                      threep_cp, threep_err, \
                                      E0[ b ], E1[ b ] ), \
                             method="lm" )

        fit[ b ] = leastSq.x

        chi_sq[ b ] = leastSq.cost

    # End loop over bins

    return np.array( fit ), np.array( chi_sq )

def twoStateErrorFunction_twop( fitParams, tsink, T, twop, twop_err ):

    c0 = fitParams[ 0 ]
    c1 = fitParams[ 1 ]
    E0 = fitParams[ 2 ]
    E1 = fitParams[ 3 ]

    twopErr = np.array( ( twoStateTwop( tsink, T, c0, c1, E0, E1 ) \
                          - twop ) / twop_err )
    
    return twopErr
    

def twoStateErrorFunction_threep( fitParams, ti, tsink, T, \
                                  threep, threep_err, E0, E1):

    a00 = fitParams[ 0 ]
    a01 = fitParams[ 1 ]
    a11 = fitParams[ 2 ]

    # threepErr[ ts * ti ]

    threepErr = []

    for ti_ts, ts, threep_ts, threep_err_ts \
        in zip( ti, tsink, threep, threep_err ):

        for t, threep_ti, threep_err_ti \
            in zip( ti_ts, threep_ts, threep_err_ts ):

            threepErr.append( ( twoStateThreep( t, ts, T, \
                                                a00, a01, a11, \
                                                E0, E1 ) \
                                - threep_ti ) / threep_err_ti )

    return np.array( threepErr )
    

def twoStateThreep( ti, tsink, T, a00, a01, a11, E0, E1 ):

    if ti < tsink:

        return a00 * np.exp( -E0 * tsink ) \
            + a01 * np.exp( -E0 * ( tsink - ti ) - E1 * ti ) \
            + a01 * np.exp( -E1 * ( tsink - ti ) - E0 * ti ) \
            + a11 * np.exp( -E1 * tsink )

    else:
        
        return a00 * np.exp( -E0 * ( T - tsink ) ) \
            a01 * np.exp( -E0 * ( ti - tsink ) \
                            - E1 * ( T - ti ) ) \
            a01 * np.exp( -E1 * ( ti - tsink ) \
                            - E0 * ( T - ti ) ) \
            a11 * np.exp( -E1 * ( T - tsink ) )
        

def twoStateTwop( tsink, T, c0, c1, E0, E1 ):

    return c0 * ( np.exp( -E0 * tsink ) \
                  + np.exp( -E0 * ( T - tsink ) ) ) \
        + c1 * ( np.exp( -E1 * tsink ) \
                 + np.exp( -E1 * ( T - tsink ) ) )


"""
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
    
    a00 = 1.0
    a01 = 1.0
    a11 = 1.0
    c0 = 1.0
    c1 = 1.0
    E0 = 0.5
    E1 = 0.1
    
    a00 = (-1, 1)
    a01 = (-1, 1)
    a11 = (-1, 1)
    c0 = (-0.1, 0.1)
    c1 = (-0.1, 0.1)
    E0 = (0, 1)
    E1 = (0, 0.1)

    fitParams = np.array( [ a00, a01, a11, c0, c1, E0, E1 ] )
    
    leastSq_avg = minimize( twoStateErrorFunction, fitParams, \
                        args = ( tsink_twop, ti, tsink, \
                                 twop_avg, twop_err, \
                                 threep_avg, threep_err ), \
                        method='Nelder-Mead' )
    #method='Powell' )
    
    min_avg = differential_evolution( twoStateErrorFunction, fitParams, \
                                      args = ( tsink_twop, ti, tsink, \
                                               twop_avg, twop_err, \
                                               threep_avg, threep_err ) )
    
    min_avg = least_squares( twoStateErrorFunction, fitParams, \
                             args = ( tsink_twop, ti, tsink, \
                                      twop_avg, twop_err, \
                                      threep_avg, threep_err ), \
                             method="lm" )
    
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
        
        min = least_squares( twoStateErrorFunction, fitParams, \
                             args = ( tsink_twop, ti, tsink, \
                                      twop_cp, twop_err, \
                                      threep_cp, threep_err ), \
                             method="lm" )
        
        min = differential_evolution( twoStateErrorFunction, fitParams, \
                                      args = ( tsink_twop, ti, tsink, \
                                               twop_cp, twop_err, \
                                               threep_cp, threep_err ) )
        
        min = minimize( twoStateErrorFunction, fitParams, \
                        args = ( tsink_twop, ti, tsink, \
                        twop_cp, twop_err, \
                        threep_cp, threep_err ), \
                        method='Nelder-Mead' )
        #method='Powell' )
        
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
    
    twopErr = np.array( twoStateTwop( tsink_twop, c0, c1, E0, E1 ) \
                        - twop )
    
    twopErr = np.array( ( twoStateTwop( tsink_twop, c0, c1, E0, E1 ) \
                          - twop ) ** 2 )
    
    twopErr = np.array( ( ( twoStateTwop( tsink_twop, c0, c1, E0, E1 ) \
                            - twop ) / twop ) ** 2 )

    twopErr = np.array( ( ( twoStateTwop( tsink, c0, c1, E0, E1 ) \
                            - twop ) / twop_err ) ** 2 )
    
    # threepErr[ ts ][ ti ]

    threepErr = []

    for ti_ts, ts, threep_ts, threep_err_ts in zip( ti, tsink, threep, threep_err ):

        for t, threep_ti, threep_err_ti in zip( ti_ts, threep_ts, threep_err_ts ):

            #print( "ti: " + str(t) + ", ts: " + str(ts)  )

            #print( "data: " + str(threep_ti) )

            #print( "function: " + str(twoStateThreep( t, ts, a00, a01, a11, E0, E1 ) ) )
            
            threepErr.append( twoStateThreep( t, ts, a00, a01, a11, E0, E1 ) \
                              - threep_ti )
            
            threepErr.append( ( twoStateThreep( t, ts, a00, a01, a11, E0, E1 ) \
                                - threep_ti ) ** 2 )
            
            threepErr.append( ( ( twoStateThreep( t, ts, a00, a01, a11, E0, E1 ) \
                                  - threep_ti ) / threep_ti ) ** 2 )

            threepErr.append( ( ( twoStateThreep( t, ts, a00, a01, a11, E0, E1 ) \
                                  - threep_ti ) / threep_err_ti ) ** 2 )
            

    #print( np.concatenate( ( twopErr, threepErr ) ) )

    #print( np.sum(np.concatenate( ( twopErr, threepErr ) ) ) )

    return np.sum( np.concatenate( ( twopErr, threepErr ) ) )
    
    #return np.concatenate( ( twopErr, threepErr ) )
"""

def oneStateFit_twop( twop, fitStart, fitEnd, T ):

    # twop[ b, t ]

    twop_to_fit = np.concatenate( ( twop[ :, fitStart : \
                                          fitEnd + 1 ], \
                                    twop[ :, T - fitEnd : \
                                          T - fitStart + 1 ] ), \
                                  axis=1 )

    binNum = twop.shape[ 0 ]

    fit = fncs.initEmptyList( binNum, 1 )

    chi_sq = fncs.initEmptyList( binNum, 1 )

    twop_avg = np.average( twop_to_fit, axis=0 )

    twop_err = fncs.calcError( twop_to_fit, binNum )
    
    t = np.concatenate( ( range( fitStart, fitEnd + 1 ), \
                          range( T - fitEnd, T-fitStart + 1 ) ) )

    # Find fit parameters of mean values to use as initial guess

    G = 0.1 
    E = 0.1 
        
    fitParams = np.array( [ G, E ] )

    leastSq_avg = least_squares( oneStateErrorFunction_twop, fitParams, \
                             args = ( t, T, twop_avg, twop_err ), \
                             method="lm" )
    

    fitParams = leastSq_avg.x

    for b in range( binNum ):

        leastSq = least_squares( oneStateErrorFunction_twop, fitParams, \
                                 args = ( t, T, twop_to_fit[ b, : ], \
                                          twop_err ), \
                                 method="lm" )
    
        fit[ b ] = leastSq.x

        chi_sq[ b ] = leastSq.cost

    # End loop over bins

    return np.array( fit ), np.array( chi_sq )


def oneStateErrorFunction_twop( fitParams, t, T, twop, twop_err ):

    G = fitParams[ 0 ]
          
    E = fitParams[ 1 ]
        
    # twopErr[ t ]

    twopErr = np.array( ( oneStateTwop( t, T, G, E, ) \
                          - twop ) / twop_err )

    return twopErr


def oneStateTwop( t, T, G, E ):
    
    return G**2 / 2 / E * ( np.exp( -E * t ) + np.exp( -E * ( T - t ) ) )

