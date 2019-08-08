import numpy as np
import functions as fncs
from scipy.optimize import least_squares, minimize

# Exception thrown if good fit cannot be found.
# The definition of a good fit can vary on fitting routine.

class lqcdjk_BadFitError(Exception):
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)


# Fit the effective mass using two different methods and vary the fit range
# starting point until the relative difference between the masses calculated
# by both methods is less than half of both their jackkinfe errors. 
# The different methods are fitting the effective mass plateau to a constant
# value and either a one- or two-state fit on the two-point functions.

# mEff: effective masses to be fit to a constant value
# twop: two-point functions to be fit using a one- or two-state fit
# rangeEnd: The last t value to be include in the fit range
# tsf: Perform two-state fit if True, else perform one-state fit

def mEffTwopFit( mEff, twop, rangeEnd, pSq, L, tsf ):

    binNum = mEff.shape[ 0 ]
    T = twop.shape[ -1 ]

    mEff_err = fncs.calcError( mEff, binNum )

    # Loop over plateau fit range starts
    for mEff_rangeStart in range( 10, rangeEnd - 2 ):

        mEff_fit = np.zeros( binNum )

        # Loop over bins
        for b in range( binNum ):

            # Perform the plateau fit

            mEff_fit[ b ] = np.polyfit( range( mEff_rangeStart, \
                                               rangeEnd + 1 ), \
                                        mEff[ b, \
                                              mEff_rangeStart \
                                              : rangeEnd + 1 ], 0, \
                                        w=mEff_err[ mEff_rangeStart \
                                                    : rangeEnd + 1 ] )

        # End loop over bins

        # Average over bins

        mEff_fit_avg = np.average( mEff_fit )
        mEff_fit_err = fncs.calcError( mEff_fit, binNum )

        for twop_rangeStart in range( 1, rangeEnd - 2 ):

            # Two-state fit

            if tsf:

                # fitParams[ b, param ]

                fitParams, chiSq = twoStateFit_twop( twop, \
                                                     twop_rangeStart, \
                                                     rangeEnd, T )

                E_avg = np.average( fitParams[ :, 2 ] )
                E_err = fncs.calcError( fitParams[ :, 2 ], binNum )

            # One-state fit

            else:

                # Perform one-state fit
                
                fitParams, chiSq = oneStateFit_twop( twop, \
                                                     twop_rangeStart, \
                                                     rangeEnd, T )
                
                E_avg = np.average( fitParams[ :, 1 ], axis=0 )
                E_err = fncs.calcError( fitParams[ :, 1 ], binNum )

            # End if no two-state fit

            mass = np.sqrt( E_avg ** 2 \
                            - ( 2.0 * np.pi / L ) ** 2 * pSq )

            # Check if the fits are good

            relDiff = np.abs( mEff_fit_avg - mass ) \
                      / ( 0.5 * ( mEff_fit_avg + mass ) )
        
            if 0.5 * E_err > relDiff \
               and 0.5 * mEff_fit_err > relDiff:
                
                results = ( fitParams, chiSq, mEff_fit, \
                            twop_rangeStart, mEff_rangeStart )

                return results

            # End if relDiff < dm/2
        # End loop over twop fit start
    # End loop over effective mass fit start

    raise lqcdjk_BadFitError( "fitTwop() could not find a good fit with " \
                              + "given effective masses, " \
                              + "two-point functions, " \
                              + "and range end." )

    return -1


# Fit two-point functions to a two-state fit.

# twop: Two-point functions to be fit
# twop_rangeStart: Starting t value to include in fit range
# twop_rangeEnd: Ending t value to include in fit range
# T: Time dimension length for ensemble

def twoStateFit_twop( twop, twop_rangeStart, twop_rangeEnd, T ):

    # twop[ b, t ]
    # twop_err[ t ]

    # Set two-point functions to fit based on fit range start and end

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

    # Find fit parameters for each bins

    # Loop over bins
    for b in range( binNum ):

        leastSq = least_squares( twoStateErrorFunction_twop, fitParams, \
                             args = ( tsink, T, twop_to_fit[ b, : ], \
                                      twop_err ), \
                             method="lm" )

        fit[ b ] = leastSq.x

        chi_sq[ b ] = leastSq.cost

    # End loop over bins

    return np.array( fit ), np.array( chi_sq )


# Fit three-point functions to a two-state fit.

# threep: three-point functions to be fit
# ti_to_fit: Values of insertion time to be fit over
# tsink: list of tsink values to fit over
# E0: ground state energy value calculated from two-state function fit
# E1: first excited state energy value calculated from two-state function fit
# T: Time dimension length for ensemble

def twoStateFit_threep( threep, ti_to_fit, tsink, E0, E1, T ):

    # threep[ ts, b, t ]

    tsinkNum = len( tsink )

    assert tsinkNum == len( threep ), \
        "Number of tsink's does not match " \
        + "number of three-point function datasets."

    threep_to_fit = fncs.initEmptyList( tsinkNum, 1 )

    # Set three-point functions to fit based on ti_to_fit

    for ts in range( tsinkNum ):
 
        threep_to_fit[ ts ] = threep[ ts ].take( ti_to_fit[ ts ], axis=-1 )

    # fit[b]

    binNum = threep.shape[ 1 ]

    fit = fncs.initEmptyList( binNum, 1 )

    chi_sq = fncs.initEmptyList( binNum, 1 )

    # threep_avg[ts, t]

    threep_to_fit_avg = fncs.initEmptyList( tsinkNum, 1 )
    threep_to_fit_err = fncs.initEmptyList( tsinkNum, 1 )
    
    for ts in range( tsinkNum ):

        threep_to_fit_avg[ ts ] = np.average( threep_to_fit[ ts ], \
                                              axis=0 )
        threep_to_fit_err[ ts ] = fncs.calcError( threep_to_fit[ ts ], \
                                                  binNum )

    E0_avg = np.average( E0 )
    E1_avg = np.average( E1 )

    # Find fit parameters of mean values to use as initial guess

    a00 = 1.0
    a01 = 1.0
    a11 = 1.0

    fitParams = np.array( [ a00, a01, a11 ] )

    leastSq_avg = least_squares( twoStateErrorFunction_threep, fitParams, \
                                 args = ( ti_to_fit, tsink, T, \
                                          threep_to_fit_avg, \
                                          threep_to_fit_err, \
                                          E0_avg, E1_avg ), \
                                 method="lm" )

    fitParams = leastSq_avg.x

    # Find fit parameters for each bin

    # Loop over bins
    for b in range( binNum ):

        threep_cp = fncs.initEmptyList( tsinkNum, 1 )


        # Loop over tsink
        for ts in range( tsinkNum ):

            threep_cp[ ts ] = threep_to_fit[ ts ][ b, : ]

        # End loop over tsink

        leastSq = least_squares( twoStateErrorFunction_threep, fitParams, \
                             args = ( ti_to_fit, tsink, T, \
                                      threep_cp, \
                                      threep_to_fit_err, \
                                      E0[ b ], E1[ b ] ), \
                             method="lm" )

        fit[ b ] = leastSq.x

        chi_sq[ b ] = leastSq.cost

    # End loop over bins

    return np.array( fit ), np.array( chi_sq )


# Calculate the difference between two-point function values of the data 
# and calculated from the two-state fit divided by the jackknife errors
# of the data

# fitParams: Parameters of fit (c0, c1, E0, E1)
# tsink: tsink values to fit over
# T: time dimension length of ensemble
# twop: two-point functions to fit
# twop_err: jacckife errors associated with two-point functions

def twoStateErrorFunction_twop( fitParams, tsink, T, twop, twop_err ):

    c0 = fitParams[ 0 ]
    c1 = fitParams[ 1 ]
    E0 = fitParams[ 2 ]
    E1 = fitParams[ 3 ]

    twopErr = np.array( ( twoStateTwop( tsink, T, c0, c1, E0, E1 ) \
                          - twop ) / twop_err )
    
    return twopErr
    

# Calculate the difference between three-point function values of the data 
# and calculated from the two-state fit divided by the jackknife errors
# of the data

# fitParams: Parameters of fit (a00, a01, a11 )
# ti: insertion time values to fit over
# tsink: tsink values to fit over
# T: time dimension length of ensemble
# threep: three-point functions to fit
# threep_err: jacckife errors associated with three-point functions
# E0: ground state energy value calculated from two-state function fit
# E1: first excited state energy value calculated from two-state function fit

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
    

# Calculate three-point function from given two-state fit parameters and time values

# ti: insertion time value
# tsink: tsink value
# T: time dimension length of ensemble
# a00: amplitude of ground state term (fit parameter)
# a01: amplitude of mixed state terms (fit parameter)
# a11: amplitude of first excited state term (fit parameter)
# E0: ground state energy value calculated from two-state function fit
# E1: first excited state energy value calculated from two-state function fit

def twoStateThreep( ti, tsink, T, a00, a01, a11, E0, E1 ):

    if ti < tsink:

        return a00 * np.exp( -E0 * tsink ) \
            + a01 * np.exp( -E0 * ( tsink - ti ) - E1 * ti ) \
            + a01 * np.exp( -E1 * ( tsink - ti ) - E0 * ti ) \
            + a11 * np.exp( -E1 * tsink )

    else:
        
        return a00 * np.exp( -E0 * ( T - tsink ) ) \
            + a01 * np.exp( -E0 * ( T - ti ) \
                            - E1 * ( ti - tsink ) ) \
            + a01 * np.exp( -E1 * ( T - ti ) \
                            - E0 * ( ti - tsink ) ) \
            + a11 * np.exp( -E1 * ( T - tsink ) )
        

# Calculate two-point functions from given two-state fit parameters and 
# time values

# tsink: tsink value
# T: time dimension length of ensemble
# c0: amplitude of ground state term (fit parameter)
# c1: amplitude of first excited state term (fit parameter)
# E0: ground state energy (fit parameter)
# E1: first excited state energy (fit parameter)

def twoStateTwop( tsink, T, c0, c1, E0, E1 ):

    return c0 * ( np.exp( -E0 * tsink ) \
                  + np.exp( -E0 * ( T - tsink ) ) ) \
        + c1 * ( np.exp( -E1 * tsink ) \
                 + np.exp( -E1 * ( T - tsink ) ) )


# Fit two-point functions to a one-state fit.

# twop: Two-point functions to be fit
# twop_rangeStart: Starting t value to include in fit range
# twop_rangeEnd: Ending t value to include in fit range
# T: Time dimension length for ensemble

def oneStateFit_twop( twop, twop_rangeStart, twop_rangeEnd, T ):

    # twop[ b, t ]

    twop_to_fit = np.concatenate( ( twop[ :, twop_rangeStart : \
                                          twop_rangeEnd + 1 ], \
                                    twop[ :, T - twop_rangeEnd : \
                                          T - twop_rangeStart + 1 ] ), \
                                  axis=1 )

    binNum = twop.shape[ 0 ]

    fit = fncs.initEmptyList( binNum, 1 )

    chi_sq = fncs.initEmptyList( binNum, 1 )

    twop_avg = np.average( twop_to_fit, axis=0 )

    twop_err = fncs.calcError( twop_to_fit, binNum )
    
    t = np.concatenate((range( twop_rangeStart, twop_rangeEnd + 1 ), \
                        range( T - twop_rangeEnd, T-twop_rangeStart + 1 ) ))

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


# Calculate the difference between two-point function values of the data 
# and calculated from the one-state fit divided by the jackknife errors
# of the data

# fitParams: Parameters of fit (G, E)
# tsink: tsink values to fit over
# T: time dimension length of ensemble
# twop: two-point functions to fit
# twop_err: jacckife errors associated with two-point functions

def oneStateErrorFunction_twop( fitParams, tsink, T, twop, twop_err ):

    G = fitParams[ 0 ]
    E = fitParams[ 1 ]
        
    # twopErr[ tsink ]

    twopErr = np.array( ( oneStateTwop( tsink, T, G, E, ) \
                          - twop ) / twop_err )

    return twopErr


# Calculate two-point functions from given one-state fit parameters and 
# time values

# tsink: tsink value
# T: time dimension length of ensemble
# G: amplitude (fit parameter)
# E: ground state energy (fit parameter)

def oneStateTwop( tsink, T, G, E ):
    
    return G**2 / 2 / E * ( np.exp( -E * tsink ) \
                            + np.exp( -E * ( T - tsink ) ) )


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

def fitGenFormFactor( vals, vals_err, fitStart, fitEnd ):

    # vals[ b, Q, ratio, t ]
    # vals_err[ Q, ratio, t ]

    fit = np.empty( vals.shape[ :-1 ] )

    # Loop over bins
    for b in range( vals.shape[ 0 ] ):
        # Loop over Q
        for iq in range( vals.shape[ 1 ] ):
            # Loop over ratio
            for ir in range( vals.shape[ 2 ] ):

                fit[ b, iq, ir ] = np.polyfit(range( fitStart, \
                                                      fitEnd + 1 ), \
                                               vals[ b, iq, ir, \
                                                     fitStart \
                                                     : fitEnd + 1 ], \
                                               0, w=vals_err[ iq, ir, \
                                                              fitStart \
                                                              : fitEnd + 1 ])

    return fit

