import numpy as np
import functions as fncs
import physQuants as pq
from scipy.optimize import least_squares, minimize

# Exception thrown if good fit cannot be found.
# The definition of a good fit can vary on fitting routine.

class lqcdjk_BadFitError(Exception):
    def __init__(self, mismatch):
        Exception.__init__(self, mismatch)


# Wrapper for numpy.polyfit to fit a plateau line to data

def fitPlateau( data, err, start, end ):

    # data[ b, x ]
    # err[ b ]
    # start
    # end

    binNum = data.shape[ 0 ]

    # x values to fit

    x = range( start, \
               end + 1 )

    fit = np.zeros( binNum )
    chiSq = np.zeros( binNum )

    # Loop over bins
    for b in range( binNum ):

        fit[ b ], chiSq[ b ], \
            dum, dum, dum = np.polyfit( x, \
                                        data[ b, \
                                              start \
                                              : end + 1 ], 0, \
                                        w=err[ start \
                                               : end + 1 ] ** -1, \
                                        full=True )
        
    # End loop over bin

    return fit, chiSq


def testmEffTwopFit( mEff, twop, rangeEnd, pSq, L, tsf ):

    binNum = mEff.shape[ 0 ]
    T = 2 * ( twop.shape[ -1 ] - 1 )

    mEff_err = fncs.calcError( mEff, binNum )

    mEff_results = []
    twop_tsf_results = []
    mEff_tsf_results = []

    # Loop over plateau fit range starts
    for mEff_rangeStart in range( 5, rangeEnd - 5 ):

        mEff_fit = np.zeros( binNum )
        mEff_chiSq = np.zeros( binNum )

        # Perform the plateau fit

        mEff_fit, mEff_chiSq, \
            = fitPlateau( mEff, mEff_err, \
                          mEff_rangeStart, mEff_rangeEnd)
            
        mEff_chiSq_pdf = mEff_chiSq / ( len( t_to_fit ) - 1 )

        mEff_results.append( ( mEff_fit, mEff_chiSq_pdf, \
                               mEff_rangeStart ) )

        mEff_fit_avg = np.average( mEff_fit, axis=0 )
        mEff_fit_err = fncs.calcError( mEff_fit, binNum )

    # End loop over effective mass fit start

    for twop_rangeStart in range( 1, 8 ):

        # Two-state fit

        if tsf:

            # fitParams[ b, param ]

            fitParams, chiSq = twoStateFit_twop( twop, \
                                                 twop_rangeStart, \
                                                 rangeEnd, T )

            E_avg = np.average( fitParams[ :, 2 ] )
            E_err = fncs.calcError( fitParams[ :, 2 ], binNum )
                
            # Degress of freedom

            twop_dof = rangeEnd - twop_rangeStart + 1 - 4
            mEff_tsf_dof = rangeEnd - twop_rangeStart + 1 - 3

            mEff_tsf_fitParams, mEff_tsf_chiSq = twoStateFit_mEff( mEff, \
                                                                   twop_rangeStart, \
                                                                   rangeEnd, T )

            #E_avg = np.average( mEff_tsf_fitParams[:,1], axis=0 )
            
            #mass = np.sqrt( E_avg ** 2 \
            #                - ( 2.0 * np.pi / L ) ** 2 * pSq )

            #relDiff = np.abs( mEff_fit_avg - mass ) \
            #          / ( 0.5 * ( mEff_fit_avg + mass ) )
        
            #goodFit = 0.5 * mEff_fit_err > relDiff

            #print("t_low_plat={}< t_low_tsf={}:".format(mEff_rangeStart,twop_rangeStart))
            #print("m_plat={}, m_tsf={}".format(mEff_fit_avg,mass))
            #print("relDif={}, err={}, goodFit={}".format(relDiff,mEff_fit_err,goodFit))

        
        else: # One-state fit
        
            # fitParams[ b, param ]

            fitParams, chiSq = oneStateFit_twop( twop, \
                                                 twop_rangeStart, \
                                                 rangeEnd, T )
                
            E_avg = np.average( fitParams[ :, 1 ], axis=0 )
            E_err = fncs.calcError( fitParams[ :, 1 ], binNum )
            
            dof = twop_rangeStart - rangeEnd + 1 - 2

        # End if no two-state fit

        chiSq_pdf = chiSq / twop_dof
        mEff_tsf_chiSq_pdf = mEff_tsf_chiSq / mEff_tsf_dof

        twop_tsf_results.append( ( fitParams, chiSq_pdf, \
                                   twop_rangeStart ) )
        
        mEff_tsf_results.append( ( mEff_tsf_fitParams, mEff_tsf_chiSq_pdf, \
                                   twop_rangeStart ) )

    # End loop over twop fit start

    return mEff_results, twop_tsf_results, mEff_tsf_results


# Fit the effective mass using two different methods and vary the fit range
# starting point until the relative difference between the masses calculated
# by both methods is less than half of both their jackkinfe errors. 
# The different methods are fitting the effective mass plateau to a constant
# value and either a one- or two-state fit on the two-point functions.

# mEff: effective masses to be fit to a constant value
# twop: two-point functions to be fit using a one- or two-state fit
# rangeEnd: The last t value to be include in the fit range
# tsf: Perform two-state fit if True, else perform one-state fit

def mEffTwopFit( mEff, twop, rangeEnd, pSq, L, tsf, **kwargs ):

    binNum = mEff.shape[ 0 ]
    T = 2 * ( twop.shape[ -1 ] - 1 )

    mEff_err = fncs.calcError( mEff, binNum )

    if "mEff_t_low_range" in kwargs \
       and kwargs[ "mEff_t_low_range" ]:

        mEff_t_low_range = kwargs[ "mEff_t_low_range" ]

    else:

        mEff_t_low_range = range( 10, rangeEnd - 10 )

    if "twop_t_low_range" in kwargs \
       and kwargs[ "twop_t_low_range" ]:

        twop_t_low_range = kwargs[ "twop_t_low_range" ]

    else:

        twop_t_low_range = range( 1, 8 )

    if "checkFit" in kwargs:

        checkFit = kwargs[ "checkFit" ]

    else:

        checkFit = True

    # Loop over plateau fit range starts
    for mEff_t_low in mEff_t_low_range:

        mEff_fit = np.zeros( binNum )

        # Perform the plateau fit

        mEff_fit, chiSq = fitPlateau( mEff, mEff_err, \
                                      mEff_t_low, \
                                      rangeEnd )

        # End loop over bins

        # Average over bins

        mEff_fit_avg = np.average( mEff_fit )
        mEff_fit_err = fncs.calcError( mEff_fit, binNum )

        for twop_t_low in twop_t_low_range:

            # Two-state fit

            if tsf:

                # fitParams[ b, param ]

                dof = rangeEnd - twop_t_low + 1 - 3
            
                fitParams, chiSq = twoStateFit_mEff( mEff, \
                                                     twop_t_low, \
                                                     rangeEnd, T )
                
                chiSq /= dof

                E_avg = np.average( fitParams[ :, 1 ] )
                E_err = fncs.calcError( fitParams[ :, 1 ], binNum )
                    
                #fitParams, chiSq = twoStateFit_twop( twop, \
                #                                     twop_t_low, \
                #                                     rangeEnd, T )
                    
                #E_avg = np.average( fitParams[ :, 2 ] )
                #E_err = fncs.calcError( fitParams[ :, 2 ], binNum )

                # One-state fit

            else:

                # Perform one-state fit
                
                fitParams, chiSq = oneStateFit_twop( twop, \
                                                     twop_t_low, \
                                                     rangeEnd, T )
                
                E_avg = np.average( fitParams[ :, 1 ], axis=0 )
                E_err = fncs.calcError( fitParams[ :, 1 ], binNum )

            # End if no two-state fit

            mass = np.sqrt( E_avg ** 2 \
                            - ( 2.0 * np.pi / L ) ** 2 * pSq )

            if checkFit:

                # Check if the fits are good
            
                relDiff = np.abs( mEff_fit_avg - mass ) \
                          / ( 0.5 * ( mEff_fit_avg + mass ) )
        
                if 0.5 * mEff_fit_err > relDiff:

                    return ( fitParams, chiSq, mEff_fit, \
                             twop_t_low, mEff_t_low )

            else:

                return ( fitParams, chiSq, mEff_fit, \
                         twop_t_low, mEff_t_low )

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

    twop_to_fit = twop[ :, twop_rangeStart : \
                        twop_rangeEnd + 1 ]

    # fit[b]

    binNum = twop.shape[ 0 ]

    fit = fncs.initEmptyList( binNum, 1 )
    chi_sq = fncs.initEmptyList( binNum, 1 )

    # twop_avg[ts]

    twop_avg = np.average( twop_to_fit, axis=0 )
    twop_err = fncs.calcError( twop_to_fit, binNum )
    
    tsink = np.array( range( twop_rangeStart, \
                             twop_rangeEnd + 1 ) )

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


def twoStateFit_mEff( mEff, rangeStart, rangeEnd, T ):

    # mEff[ b, t ]
    # twop_err[ t ]

    # Set two-point functions to fit based on fit range start and end

    mEff_to_fit = mEff[ :, rangeStart : \
                        rangeEnd + 1 ]

    # fit[b]

    binNum = mEff.shape[ 0 ]

    fit = fncs.initEmptyList( binNum, 1 )
    chi_sq = fncs.initEmptyList( binNum, 1 )

    # mEff_avg[t]

    mEff_avg = np.average( mEff_to_fit, axis=0 )
    mEff_err = fncs.calcError( mEff_to_fit, binNum )
    
    t_to_fit = np.array( range( rangeStart, \
                                rangeEnd + 1 ) )

    # Find fit parameters of mean values to use as initial guess

    c = 1.0
    E0 = 0.1
    E1 = 1.0

    fitParams = np.array( [ c, E0, E1 ] )

    leastSq_avg = least_squares( twoStateErrorFunction_mEff, fitParams, \
                                 args = ( t_to_fit, T, mEff_avg, mEff_err ), \
                                 method="lm" )

    fitParams = leastSq_avg.x

    # Find fit parameters for each bins

    # Loop over bins
    for b in range( binNum ):

        leastSq = least_squares( twoStateErrorFunction_mEff, fitParams, \
                                 args = ( t_to_fit, T, mEff_to_fit[ b, : ], \
                                      mEff_err ), \
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
    

def twoStateErrorFunction_mEff( fitParams, tsink, T, mEff, mEff_err ):

    c = fitParams[ 0 ]
    E0 = fitParams[ 1 ]
    E1 = fitParams[ 2 ]

    twopErr = np.array( ( twoStatemEff( tsink, T, c, E0, E1 ) \
                          - mEff ) / mEff_err )
    
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


def twoStatemEff( tsink, T, c, E0, E1 ):

    twop_halfT = twoStateTwop( T // 2, T, \
                               1, c, \
                               E0, E1 )

    twop_tp1 = twoStateTwop( tsink + 1, T, \
                             1, c, \
                             E0, E1 )

    twop_tm1 = twoStateTwop( tsink - 1, T, \
                             1, c, \
                             E0, E1 )
    
    return 0.5 * np.log(( twop_tm1 \
                          + np.sqrt( twop_tm1 ** 2 \
                                     - twop_halfT ** 2 )) \
                        / ( twop_tp1 \
                            + np.sqrt( twop_tp1 ** 2 \
                                       - twop_halfT ** 2 )))


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


def fitGenFormFactor( vals, vals_err, fitStart, fitEnd ):

    # vals[ b, Q, ratio, t ]
    # vals_err[ Q, ratio, t ]

    fit = np.empty( vals.shape[ :-1 ] )
    chiSq = np.empty( vals.shape[ :-1 ] )

    # Loop over Q
    for iq in range( vals.shape[ 1 ] ):
        # Loop over ratio
        for ir in range( vals.shape[ 2 ] ):

            fit[ :, iq, ir ], chiSq[ :, iq, ir ] \
                = fitPlateau( vals[ :, iq, ir ], \
                              vals_err[ iq, ir ], \
                              fitStart, \
                              fitEnd )

    return fit


def calcmEffTwoStateCurve( c0, c1, E0, E1, T, rangeStart, rangeEnd ):

    binNum = c0.shape[ 0 ]

    curve = np.zeros( ( binNum, 100 ) )

    ts = np.linspace( rangeStart, \
                      rangeEnd, 100 )

    for b in range( binNum ):

        twop_halfT = twoStateTwop( T // 2, T, \
                                   c0[ b ], c1[ b ], \
                                   E0[ b ], E1[ b ] )

        for t in range( ts.shape[ -1 ] ):
                
            twop_tp1 = twoStateTwop( ts[ t ] + 1, T, \
                                     c0[ b ], c1[ b ], \
                                     E0[ b ], E1[ b ] )

            twop_tm1 = twoStateTwop( ts[ t ] - 1, T, \
                                     c0[ b ], c1[ b ], \
                                     E0[ b ], E1[ b ] )

            curve[ b, t ] = 0.5 * np.log(( twop_tm1 \
                                           + np.sqrt( twop_tm1 ** 2 \
                                                      - twop_halfT ** 2 )) \
                                         / ( twop_tp1 \
                                             + np.sqrt( twop_tp1 ** 2 \
                                                        - twop_halfT ** 2 )))

    
    return curve, ts


def calcTwopOneStateCurve( G, E, T, rangeStart, rangeEnd ):

    binNum = c0.shape[ 0 ]

    curve = np.zeros( ( binNum, 100 ) )

    ts = np.linspace( rangeStart, \
                      rangeEnd, 100 )

    for b in range( binNum ):
        for t in range( ts.shape[ -1 ] ):
                
            curve[ b, t ] = oneStateTwop( t_s[ t ], T, \
                                              G[ b ], E[ b ] )

    
    return curve, ts


def calcTwopTwoStateCurve( c0, c1, E0, E1, T, rangeStart, rangeEnd ):

    binNum = c0.shape[ 0 ]

    curve = np.zeros( ( binNum, 100 ) )

    ts = np.linspace( rangeStart, \
                      rangeEnd, 100 )

    for b in range( binNum ):
        for t in range( ts.shape[ -1 ] ):
                
            curve[ b, t ] = twoStateTwop( ts[ t ], T, \
                                          c0[ b ], c1[ b ], \
                                          E0[ b ], E1[ b ] )

    
    return curve, ts


def calcThreepTwoStateCurve( a00, a01, a11, E0, E1, T, tsink, \
                             ti_to_fit, neglect ):

    # a00[ b ] 
    # a01[ b ] 
    # a11[ b ] 
    # tsink[ ts ] 
    # ti_to_fit[ ts ][ t ]
    # neglect

    tsinkNum = len( tsink )

    binNum = a00.shape[ 0 ]

    ti = np.zeros( ( tsinkNum, 100 ) )
    curve = np.zeros( ( binNum, tsinkNum, 100 ) )
            
    for b in range( binNum ):
        for ts in range( tsinkNum ):

            ti[ ts ] = np.linspace( ti_to_fit[ ts ][ 0 ], \
            ti_to_fit[ ts ][ -1 ], \
            num = 100 )
            """
            ti[ts]=np.concatenate((np.linspace(ti_to_fit[ts][0],\
                                               ti_to_fit[ts][tsink[ts] - 2\
                                                         * neglect], \
                                               num = 50), \
                                   np.linspace(ti_to_fit[ts][tsink[ts] - 2\
                                                         * neglect + 1],\
                                               ti_to_fit[ts][-1], \
                                               num = 50)))
            """
            for t in range( len( ti[ ts ] ) ):
                
                curve[b,ts,t] = twoStateThreep( ti[ ts, t ], \
                                                tsink[ ts ], \
                                                T, \
                                                a00[ b ], \
                                                a01[ b ], \
                                                a11[ b ], \
                                                E0[ b ], \
                                                E1[ b ] )

            # End loop over insertion time
        # End loop over tsink
    # End loop over bin

    return curve, ti


def calcAvgXTwoStateCurve_const_ts( a00, a01, a11, c0, c1, \
                                    E0, E1, mEff, momSq, L, T, \
                                    ZvD1, tsink, ti_to_fit, \
                                    neglect ):

    # a00[ b ] 
    # a01[ b ] 
    # a11[ b ] 
    # c0[ b ] 
    # c11[ b ] 
    # E0[ b ] 
    # E1[ b ] 
    # mEff[ b ] 
    # momSq
    # ZvD1
    # L
    # T
    # tsink[ ts ] 
    # ti_to_fit[ ts ][ t ]
    # neglect

    tsinkNum = len( tsink )

    binNum = a00.shape[ 0 ]

    curve = np.zeros( ( binNum, tsinkNum, 100 ) )
    ti = np.zeros( ( tsinkNum, 100 ) )
            
    for b in range( binNum ):
        for ts in range( tsinkNum ):

            ti[ ts ] = np.linspace( ti_to_fit[ ts ][ 0 ], \
            ti_to_fit[ ts ][ -1 ], \
            num = 100 )
            """
            ti[ts]=np.concatenate((np.linspace(ti_to_fit[ts][0],\
                                               ti_to_fit[ts][tsink[ts] - 2\
                                                         * neglect], \
                                               num = 50), \
                                   np.linspace(ti_to_fit[ts][tsink[ts] - 2\
                                                         * neglect + 1],\
                                               ti_to_fit[ts][-1], \
                                               num = 50)))
            """
            for t in range( len( ti[ ts ] ) ):
                
                curve[ b, ts, t ] = ZvD1 \
                                    * pq.avgXKineFactor( mEff[ b ], \
                                                         momSq, \
                                                         L ) \
                                    * twoStateThreep( ti[ ts, t ], \
                                                      tsink[ ts ], \
                                                      T, \
                                                      a00[ b ], \
                                                      a01[ b ], \
                                                      a11[ b ], \
                                                      E0[ b ], \
                                                      E1[ b ] ) \
                                    / c0[ b ] / np.exp( -E0[ b ] \
                                                        * tsink[ ts ] )
                """
                                    / twoStateTwop( tsink[ ts ], \
                                                    T, \
                                                    c0[ b ], \
                                                    c1[ b ], \
                                                    E0[ b ], \
                                                    E1[ b ] )
                """
            # End loop over insertion time
        # End loop over tsink
    # End loop over bin

    return curve, ti


def calcAvgXTwoStateCurve_const_ti( a00, a01, a11, c0, c1, \
                                    E0, E1, mEff, momSq, L, T, \
                                    ZvD1, firstTs, lastTs ):

    # a00[ b ] 
    # a01[ b ] 
    # a11[ b ] 
    # c0[ b ] 
    # c11[ b ] 
    # E0[ b ] 
    # E1[ b ] 
    # mEff[ b ] 
    # momSq
    # L
    # T
    # ZvD1
    #firstTs
    #lastTs

    binNum = a00.shape[ 0 ]

    curve = np.zeros( ( binNum, 100 ) )
    tsink = np.linspace( firstTs, lastTs, num=100 )
            
    for b in range( binNum ):
        for ts in range( len( tsink ) ):
                
            curve[ b, ts ] = ZvD1 \
                             * pq.avgXKineFactor( mEff[ b ], \
                                                  momSq, \
                                                  L ) \
                             * twoStateThreep( tsink[ ts ] / 2, \
                                               tsink[ ts ], \
                                               T, \
                                               a00[ b ], \
                                               a01[ b ], \
                                               a11[ b ], \
                                               E0[ b ], \
                                               E1[ b ] ) \
                             / c0[ b ] / np.exp( -E0[ b ] \
                                                 * tsink[ ts ] )
            """
            / twoStateTwop( tsink[ ts ], \
            T, \
            c0[ b ], \
            c1[ b ], \
            E0[ b ], \
            E1[ b ] )
            """
        # End loop over tsink
    # End loop over bin

    return curve, tsink

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
