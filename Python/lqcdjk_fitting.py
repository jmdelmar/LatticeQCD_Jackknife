import numpy as np
import functions as fncs
import physQuants as pq
import mpi_functions as mpi_fncs
from mpi4py import MPI
from scipy.optimize import least_squares, minimize, \
    differential_evolution, curve_fit
from scipy.special import beta

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

    dof = end - start + 1 - 1

    fit = np.zeros( binNum )
    chiSq = np.zeros( binNum )

    # Loop over bins
    for b in range( binNum ):

        if np.any( np.isnan( err[ start : end + 1 ] ) ):

            rangeEnd \
                = min( np.where( np.isnan( err ) )[-1] ) - 1

        else:

            rangeEnd = end

        # x values to fit

        x = range( start,
                   rangeEnd + 1 )

        fit[ b ], chiSq[ b ], \
            dum, dum, dum = np.polyfit( x,
                                        data[ b,
                                              start \
                                              : rangeEnd + 1 ], 0,
                                        w=err[ start \
                                               : rangeEnd + 1 ] ** -1,
                                        full=True )

    # End loop over bin

    chiSq = chiSq / dof

    return fit, chiSq


# Wrapper for numpy.polyfit to fit a plateau line to data in parallel

def fitPlateau_parallel( data, start, end, mpi_confs_info ):

    # data[ b, x ]
    # err[ b ]
    # start
    # end

    comm = mpi_confs_info[ 'comm' ]
    binNum = mpi_confs_info[ 'binNum_glob' ]
    recvCount = mpi_confs_info[ 'recvCount' ]
    recvOffset = mpi_confs_info[ 'recvOffset' ]
    binList_loc = mpi_confs_info[ 'binList_loc' ]
    binNum_loc = mpi_confs_info[ 'binNum_loc' ]

    assert data.shape[ 0 ] == binNum, \
        "First dimension size of data " + str( data.shape[0] ) \
        + " does not match number of bins " + str( binNum ) + "."

    dof = end - start + 1 - 1

    # x values to fit

    fit_loc = np.zeros( binNum_loc )
    chiSq_loc = np.zeros( binNum_loc )

    err = fncs.calcError( data, binNum )

    # Loop over bins
    for b, ib in zip( binList_loc, range( binNum_loc ) ):

        if np.any( np.isnan( err[ start : end + 1 ] ) ):

            rangeEnd \
                = min( np.where( np.isnan( err ) )[-1] ) - 1

        else:

            rangeEnd = end

        # x values to fit

        x = range( start, \
                   rangeEnd + 1 )

        fit_loc[ ib ], chiSq_loc[ ib ], \
            dum, dum, dum = np.polyfit( x,
                                        data[ b,
                                              start \
                                              : rangeEnd + 1 ], 0,
                                        w=err[ start \
                                               : rangeEnd + 1 ] ** -1,
                                        full=True )
        
    # End loop over bin

    fit = np.zeros( binNum )
    chiSq = np.zeros( binNum )

    comm.Allgatherv( fit_loc, 
                     [ fit, recvCount, recvOffset, MPI.DOUBLE ] )
    comm.Allgatherv( chiSq_loc, 
                     [ chiSq, recvCount, recvOffset, MPI.DOUBLE ] )

    chiSq = chiSq / dof

    return fit, chiSq


def testEffEnergyTwopFit( effEnergy, twop, rangeEnd, pSq, L, particle, 
                          tsf, mpi_confs_info ):

    comm = mpi_confs_info[ 'comm' ]
    rank = mpi_confs_info[ 'rank' ]
    binNum = mpi_confs_info[ 'binNum_glob' ]
    binNum_loc = mpi_confs_info[ 'binNum_loc' ]

    assert effEnergy.shape[ 0 ] == binNum, \
        "First dimension size of effEnergy " + str( effEnergy.shape[0] ) \
        + " does not match number of bins " + str( binNum ) + "."

    rangeEnd_twop = rangeEnd
    rangeEnd_effEnergy = rangeEnd

    T = 2 * ( twop.shape[ -1 ] - 1 )

    plat_results = []
    twop_tsf_results = []
    effEnergy_tsf_results = []

    # Loop over plateau fit range starts
    for plat_rangeStart in range( 5, rangeEnd - 5 ):

        plat_fit = np.zeros( binNum )
        plat_chiSq = np.zeros( binNum )

        # Perform the plateau fit

        plat_fit, plat_chiSq, \
            = fitPlateau_parallel( effEnergy,
                                   plat_rangeStart, rangeEnd, 
                                   mpi_confs_info )
            
        if rank == 0:

            plat_results.append( ( plat_fit, plat_chiSq,
                                   plat_rangeStart ) )

        # Average over bins

    # End loop over effective mass fit start

    for twop_rangeStart in range( 1, 8 ):

        # Two-state fit

        if particle == "pion":

            E_guess = 0.3

        elif particle == "kaon":

            E_guess = 0.4

        if tsf:

            # fitParams[ b, param ]

            fitParams, chiSq = twoStateFit_twop( twop,
                                                 twop_rangeStart,
                                                 rangeEnd_twop, 
                                                 E_guess, T,
                                                 mpi_confs_info )

            effEnergy_tsf_fitParams, effEnergy_tsf_chiSq \
                    = twoStateFit_effEnergy( effEnergy, twop_rangeStart,
                                        rangeEnd_effEnergy, 
                                        E_guess, T,
                                        mpi_confs_info )
                
        else: # One-state fit
        
            # fitParams[ b, param ]

            fitParams, chiSq = oneStateFit_twop( twop, \
                                                 twop_rangeStart, \
                                                 rangeEnd, T )
                
            E_avg = np.average( fitParams[ :, 1 ], axis=0 )
            E_err = fncs.calcError( fitParams[ :, 1 ], binNum )
            
        # End if no two-state fit

        if rank == 0:

            twop_tsf_results.append( ( fitParams, chiSq,
                                       twop_rangeStart ) )
        
            effEnergy_tsf_results.append( ( effEnergy_tsf_fitParams, effEnergy_tsf_chiSq,
                                       twop_rangeStart ) )

    # End loop over twop fit start

    if rank == 0:

        # Loop over plateau fit range starts
        for plat_tlow, iplat_tlow in zip( range( 5, rangeEnd - 5 ), 
                                          range( 0, rangeEnd - 5 - 5 ) ):
        
            plat_fit = plat_results[ iplat_tlow ][ 0 ]
        
            plat_fit_avg = np.average( plat_fit, axis=0 )
            plat_fit_err = fncs.calcError( plat_fit, binNum )

            for twop_tlow, itwop_tlow in zip( range( 1, 8 ), range( 0, 7 ) ):

                E = twop_tsf_results[ itwop_tlow ][ 0 ][ :, 2 ]

                E_avg = np.average( E, axis=0 )
                E_err = fncs.calcError( E, binNum )

                # Check if the fits are good
            
                diff = np.abs( plat_fit_avg - E_avg )
        
                if 0.5 * plat_fit_err > diff and E_err > diff:

                    template = "plat tlow={}, 2sf twop tlow={}, " \
                               + "plat={}, 2sf E={}"

                    print(template.format(plat_tlow, twop_tlow,
                                          E_avg, plat_fit_avg))

                E = effEnergy_tsf_results[ itwop_tlow ][ 0 ][ :, 1 ]
                
                E_avg = np.average( E, axis=0 )
                E_err = fncs.calcError( E, binNum )

                # Check if the fits are good
            
                diff = np.abs( plat_fit_avg - E_avg )
        
                if 0.5 * plat_fit_err > diff and E_err > diff:

                    template = "plat tlow={}, 2sf effE tlow={}, " \
                               + "plat={}, 2sf effE={}"

                    print(template.format(plat_tlow, twop_tlow,
                                          E_avg, plat_fit_avg))

    return plat_results, twop_tsf_results, effEnergy_tsf_results


# Fit the effective mass using two different methods and vary the fit range
# starting point until the relative difference between the masses calculated
# by both methods is less than half of both their jackknife errors. 
# The different methods are fitting the effective mass plateau to a constant
# value and either a one- or two-state fit on the two-point functions.

# mEff: effective masses to be fit to a constant value
# twop: two-point functions to be fit using a one- or two-state fit
# rangeEnd: The last t value to be include in the fit range
# tsf: Perform two-state fit if True, else perform one-state fit

def effEnergyTwopFit( effEnergy, twop, rangeEnd, pSq, L, tsf, 
                      mpi_confs_info, **kwargs ):

    comm = mpi_confs_info[ 'comm' ]
    rank = mpi_confs_info[ 'rank' ]
    binNum = mpi_confs_info[ 'binNum_glob' ]
    binNum_loc = mpi_confs_info[ 'binNum_loc' ]

    assert effEnergy.shape[ 0 ] == binNum, \
        "First dimension size of effEnergy " + str( effEnergy.shape[0] ) \
        + " does not match number of bins " + str( binNum ) + "."

    T = 2 * ( twop.shape[ -1 ] - 1 )    

    if "plat_t_low_range" in kwargs \
       and None not in kwargs[ "plat_t_low_range" ]:

        plat_t_low_range = kwargs[ "plat_t_low_range" ]

    else:

        if pSq == 0:

            plat_t_low_range = range( 11, 21 )

        elif pSq <= 3 :
            
            plat_t_low_range = range( 11, 21 )

        else:

            plat_t_low_range = range( 5, 21 )

    if "tsf_t_low_range" in kwargs \
       and None not in kwargs[ "tsf_t_low_range" ]:

        twop_t_low_range = kwargs[ "tsf_t_low_range" ]

    else:

        twop_t_low_range = range( 1, 5 )

    

    if "fitType" in kwargs \
       and kwargs[ "fitType" ] != None:

        fitType = kwargs[ "fitType" ]

    else:

        fitType = "effEnergy"

    if "checkFit" in kwargs:

        checkFit = kwargs[ "checkFit" ]

    else:

        checkFit = True

    if checkFit:

        goodFit = False

    # Loop over plateau fit range starts
    for plat_t_low in plat_t_low_range:

        # Perform the plateau fit

        plat_fit, chiSq = fitPlateau_parallel( effEnergy,
                                               plat_t_low, rangeEnd, 
                                               mpi_confs_info )

        if rank == 0:

            plat_fit_avg = np.average( plat_fit, axis=0 )

        else:

            plat_fit_avg = None

        plat_fit_avg = comm.bcast( plat_fit_avg, root=0 )

        for twop_t_low in twop_t_low_range:

            if tsf: # Two-state fit

                E_guess = plat_fit_avg
                
                # fitParams[ b, param ]

                if fitType == "effEnergy":

                    fitParams, chiSq \
                        = twoStateFit_effEnergy( effEnergy,
                                                 twop_t_low,
                                                 rangeEnd, 
                                                 E_guess, T, 
                                                 mpi_confs_info )
                    
                    if rank == 0:

                        E = fitParams[ :, 1 ]
                    
                elif fitType == "twop":

                    fitParams, chiSq \
                        = twoStateFit_twop( twop,
                                            twop_t_low,
                                            rangeEnd, 
                                            E_guess, T,
                                            mpi_confs_info )

                    if rank == 0:

                        E = fitParams[ :, 2 ]

                else:

                    print( "ERROR (lqcdjk_fitting.effEnergyTwopFit): " \
                           + "fit type " + str( fitType ) \
                           + " is not supported." )

                    return -1

            else: # One-state fit
                
                fitParams, chiSq = oneStateFit_twop( twop, \
                                                     twop_t_low, \
                                                     rangeEnd, T )
                
                if rank == 0:

                    E = fitParams[ :, 1 ]

            # End if no two-state fit

            if checkFit:

                if rank == 0:
                
                    # Average over bins

                    plat_fit_err = fncs.calcError( plat_fit, binNum )

                    E_avg = np.average( E )
                    E_err = fncs.calcError( E, binNum )

                    # Check if the fits are good
            
                    diff = np.abs( plat_fit_avg - E_avg )
        
                    #print(plat_t_low,twop_t_low,plat_fit_avg,E_avg,diff,plat_fit_err)

                    if 0.5 * plat_fit_err > diff and E_err > diff:
                        
                        goodFit = True

                    # End if diff < dm/2
                # End if first process

                goodFit = comm.bcast( goodFit, root=0 )

                if goodFit:
                        
                    return ( fitParams, chiSq, plat_fit,
                             twop_t_low, plat_t_low,
                             fitType )

                # End if fit is good
            # End check fit

            else: # Return fit without checking

                return ( fitParams, chiSq, plat_fit,
                         twop_t_low, plat_t_low,
                         fitType )

            # End not check fit

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

def twoStateFit_twop( twop, rangeStart, rangeEnd, E_guess, T, 
                      mpi_confs_info, **kwargs ):

    comm = mpi_confs_info[ 'comm' ]
    rank = mpi_confs_info[ 'rank' ]
    binNum = mpi_confs_info[ 'binNum_glob' ]
    binNum_loc = mpi_confs_info[ 'binNum_loc' ]
    binList_loc = mpi_confs_info[ 'binList_loc' ]
    recvCount = mpi_confs_info[ 'recvCount' ]
    recvOffset = mpi_confs_info[ 'recvOffset' ]

    assert twop.shape[ 0 ] == binNum, \
        "First dimension size of two-point functions " \
        + str( twop.shape[0] ) + " does not match number of bins " \
        + str( binNum ) + "."

    if "method" in kwargs:
        
        method = kwargs[ 'method' ]

    else:

        method = None

    paramNum = 4
    dof = rangeEnd - rangeStart + 1 - paramNum
    
    # Set two-point functions to fit based on fit range start and end

    twop_to_fit = twop[ :, rangeStart : \
                        rangeEnd + 1 ]

    # fit[b]

    fit_loc = np.zeros( ( binNum_loc, paramNum ) )
    chiSq_loc = np.zeros( binNum_loc )

    # Find fit parameters of mean values to use as initial guess

    if method == "BFGS":

        c0 = 10 ** -3
        c1 = 10 ** -3
        E0 = E_guess
        E1 = E_guess + 0.2
    
    else:

        c0 = [ 0.0, 10**-2 ]
        c1 = [ 0.0, 10**-2 ]
        E0 = [ 0.0, E_guess + 0.2 ]
        E1 = [ E_guess + 0.2, 10.0 ]
    
    fitParams = np.array( [ c0, c1, E0, E1 ] )

    tsink = np.arange( rangeStart, rangeEnd + 1 )
    #twop_err = fncs.calcError( twop_to_fit, binNum )

    # Calculate inverse of the covariant matrix

    twop_err = np.linalg.inv( np.cov( twop_to_fit, rowvar=False ) 
                              * ( binNum - 1 ) )

    if rank == 0:

        # twop_avg[ts]

        twop_avg = np.average( twop_to_fit, axis=0 )
    
        #leastSq_avg = least_squares( twoStateErrorFunction_twop, fitParams,
        #                             args = ( tsink, T,
        #                                      twop_avg, twop_err ),
        #                             method="lm" )

        if method == "BFGS":

            leastSq_avg = minimize( twoStateCostFunction_twop, fitParams, \
                                    args = ( tsink, T,
                                             twop_avg, twop_err ), \
                                    method="BFGS" )
        
            fitParams = leastSq_avg.x

        else:

            leastSq_avg = differential_evolution( twoStateCostFunction_twop, 
                                                  fitParams, ( tsink, T, 
                                                               twop_avg, 
                                                               twop_err ),
                                                  tol=0.01 )
        
            fitParams = np.array( [ [ max( leastSq_avg.x[ 0 ] - 10**-4,
                                           0.0 ), 
                                      leastSq_avg.x[ 0 ] + 10**-4 ],
                                    [ max( leastSq_avg.x[ 1 ] - 10**-4,
                                           0.0 ),
                                      leastSq_avg.x[ 1 ] + 10**-4 ],
                                    [ max( leastSq_avg.x[ 2 ] - 0.1, 0.0 ),
                                      leastSq_avg.x[ 2 ] + 0.1 ],
                                    [ max( leastSq_avg.x[ 3 ] - 0.1, 0.0 ),
                                      leastSq_avg.x[ 3 ] + 0.1 ] ] )

    comm.Bcast( fitParams, root=0 )

    # Find fit parameters for each bins

    # Loop over bins
    for b, ib in zip( binList_loc, range( binNum_loc ) ):
        
        #leastSq = least_squares( twoStateErrorFunction_twop, fitParams, \
        #                         args = ( tsink, T, twop_to_fit[ b, : ], \
        #                                  twop_err ), \
        #                         method="lm" )

        if method == "BFGS":

            leastSq = minimize( twoStateCostFunction_twop, fitParams, \
                                args = ( tsink, T, twop_to_fit[ b, : ], 
                                         twop_err ), \
                                method="BFGS" )

            fit_loc[ ib ] = np.abs( leastSq.x )

        else:

            leastSq = differential_evolution( twoStateCostFunction_twop, 
                                              fitParams, ( tsink, T, 
                                                           twop_to_fit[ b,
                                                                        : ], 
                                                           twop_err ),
                                              tol=0.0001 )

            fit_loc[ ib ] = leastSq.x

        chiSq_loc[ ib ] = leastSq.fun
        #chiSq_loc[ ib ] = leastSq.cost

    # End loop over bins

    fit = np.zeros( ( binNum, ) + fit_loc.shape[ 1: ] )
    chiSq = np.zeros( ( binNum, ) + chiSq_loc.shape[ 1: ] )

    comm.Allgatherv( fit_loc, [ fit, 
                                recvCount * np.prod( fit.shape[ 1: ] ),
                                recvOffset * np.prod( fit.shape[ 1: ] ),
                                MPI.DOUBLE ] )
    comm.Allgatherv( chiSq_loc, [ chiSq, 
                                  recvCount * np.prod( chiSq.shape[ 1: ] ),
                                  recvOffset * np.prod( chiSq.shape[ 1: ] ),
                                  MPI.DOUBLE ] )

    chiSq = np.array( chiSq ) / dof

    return fit, chiSq


def twoStateFit_twop_dispersionRelation( twop,
                                         rangeStart, rangeEnd,
                                         E_ground,
                                         pSq, L,
                                         mpi_confs_info ):

    comm = mpi_confs_info[ 'comm' ]
    rank = mpi_confs_info[ 'rank' ]
    binNum = mpi_confs_info[ 'binNum_glob' ]
    binNum_loc = mpi_confs_info[ 'binNum_loc' ]
    binList_loc = mpi_confs_info[ 'binList_loc' ]
    recvCount = mpi_confs_info[ 'recvCount' ]
    recvOffset = mpi_confs_info[ 'recvOffset' ]

    assert twop.shape[ 0 ] == binNum, \
        "First dimension size of two-point functions " \
        + str( twop.shape[0] ) + " does not match number of bins " \
        + str( binNum ) + "."

    T = 2 * ( twop.shape[ -1 ] - 1 )    

    paramNum = 3
    dof = rangeEnd - rangeStart + 1 - paramNum
    
    # Set two-point functions to fit based on fit range start and end

    twop_to_fit = twop[ :, rangeStart : \
                        rangeEnd + 1 ]

    # fit[b]

    fit_loc = np.zeros( ( binNum_loc, paramNum ) )
    chiSq_loc = np.zeros( binNum_loc )

    # Find fit parameters of mean values to use as initial guess

    E0 = pq.energy( E_ground, pSq, L )

    tsink = np.arange( rangeStart, rangeEnd + 1 )
    #twop_err = fncs.calcError( twop_to_fit, binNum )

    # Calculate inverse of the covariant matrix

    twop_err = np.linalg.inv( np.cov( twop_to_fit, rowvar=False ) 
                              * ( binNum - 1 ) )
    """
    if rank == 0:

        # twop_avg[ts]

        twop_avg = np.average( twop_to_fit, axis=0 )

        E0_avg = np.average( E0, axis=0 )
    
        c0 = 10 ** -3
        c1 = 10 ** -3
        E1 = E0_avg + 0.2

        #c0 = [ 0.0, 10**-2 ]
        #c1 = [ 0.0, 10**-2 ]
        #E1 = [ E0_avg + 0.2, 10.0 ]
    
        fitParams = np.array( [ c0, c1, E1 ] )

        #leastSq_avg = least_squares( twoStateErrorFunction_twop, fitParams,
        #                             args = ( tsink, T, twop_avg, twop_err ),
        #                             method="lm" )
        #leastSq_avg = differential_evolution( twoStateCostFunction_twop_dispRel, 
        #                                      fitParams,
        #                                      ( E0_avg, tsink, T, 
        #                                        twop_avg, 
        #                                        twop_err ),
        #                                      tol=0.01 )
        
        #fitParams = np.array( [ [ max( leastSq_avg.x[ 0 ] - 10**-4, 0.0 ), 
        #                          leastSq_avg.x[ 0 ] + 10**-4 ],
        #                        [ max( leastSq_avg.x[ 1 ] - 10**-4, 0.0 ),
        #                          leastSq_avg.x[ 1 ] + 10**-4 ],
        #                        [ max( leastSq_avg.x[ 2 ] - 0.1, 0.0 ),
        #                          leastSq_avg.x[ 2 ] + 0.1 ] ] )
        
        leastSq_avg = minimize( twoStateCostFunction_twop_dispRel, 
                                fitParams,
                                args = ( E0_avg, tsink, T, 
                                         twop_avg, 
                                         twop_err ),
                                tol=0.01,
                                method="BFGS" )
        
        fitParams = np.abs( leastSq_avg.x )
        
    else:

        #fitParams = np.zeros( ( 3, 2 ) )
        fitParams = np.zeros( 3 )

    comm.Bcast( fitParams, root=0 )
    """
    # Find fit parameters for each bins

    E0_avg = np.average( E0, axis=0 )
    
    c0 = 10 ** -3
    c1 = 10 ** -3
    E1 = E0_avg + 0.2
    
    fitParams = np.array( [ c0, c1, E1 ] )
    
    # Loop over bins
    for b, ib in zip( binList_loc, range( binNum_loc ) ):
        
        #leastSq = least_squares( twoStateErrorFunction_twop, fitParams, \
        #                         args = ( tsink, T, twop_to_fit[ b, : ], \
        #                                  twop_err ), \
        #                         method="lm" )
        #leastSq = differential_evolution( twoStateCostFunction_twop_dispRel, 
        #                                  fitParams,
        #                                  ( E0[ b ], tsink, T, 
        #                                    twop_to_fit[ b, : ], 
        #                                    twop_err ),
        #                                  tol=0.0001 )
        leastSq = minimize( twoStateCostFunction_twop_dispRel, 
                            fitParams,
                            args = ( E0[ b ], tsink, T, 
                                     twop_to_fit[ b, : ], 
                                     twop_err ),
                            tol=0.0001,
                            method="BFGS" )

        fit_loc[ ib ] = np.abs( leastSq.x )
        chiSq_loc[ ib ] = leastSq.fun
        #chiSq_loc[ ib ] = leastSq.cost

    # End loop over bins

    fit = np.zeros( ( binNum, ) + fit_loc.shape[ 1: ] )
    chiSq = np.zeros( ( binNum, ) + chiSq_loc.shape[ 1: ] )

    comm.Allgatherv( fit_loc, [ fit, 
                                recvCount * np.prod( fit.shape[ 1: ] ),
                                recvOffset * np.prod( fit.shape[ 1: ] ),
                                MPI.DOUBLE ] )
    comm.Allgatherv( chiSq_loc, [ chiSq, 
                                  recvCount * np.prod( chiSq.shape[ 1: ] ),
                                  recvOffset * np.prod( chiSq.shape[ 1: ] ),
                                  MPI.DOUBLE ] )

    chiSq = np.array( chiSq ) / dof

    return fit, chiSq


def twoStateFit_effEnergy( effEnergy, rangeStart, rangeEnd, E_guess, T, 
                           mpi_confs_info ):

    comm = mpi_confs_info[ 'comm' ]
    rank = mpi_confs_info[ 'rank' ]
    binNum = mpi_confs_info[ 'binNum_glob' ]
    binNum_loc = mpi_confs_info[ 'binNum_loc' ]
    binList_loc = mpi_confs_info[ 'binList_loc' ]
    recvCount = mpi_confs_info[ 'recvCount' ]
    recvOffset = mpi_confs_info[ 'recvOffset' ]

    assert effEnergy.shape[ 0 ] == binNum, \
        "First dimension size of effective mass " \
        + str( effEnergy.shape[0] ) + " does not match number of bins " \
        + str( binNum ) + "."

    paramNum = 3
    dof = rangeEnd - rangeStart + 1 - paramNum

    # Set two-point functions to fit based on fit range start and end

    effEnergy_to_fit = effEnergy[ :, rangeStart : \
                        rangeEnd + 1 ]

    # fit[b]

    fit_loc = np.zeros( ( binNum_loc, paramNum ) )
    chiSq_loc = np.zeros( binNum_loc )

    # Find fit parameters of mean values to use as initial guess
    
    #c = 0.5
    #E0 = 0.1
    #E1 = 1.0
    c = [ 0.0, 2.0 ]
    E0 = [ 0.0, E_guess + 0.2 ]
    E1 = [ E_guess + 0.2, 2.0 ]

    fitParams = np.array( [ c, E0, E1 ] )

    t_to_fit = np.arange( rangeStart, rangeEnd + 1 )
    #effEnergy_err = fncs.calcError( effEnergy_to_fit, binNum )

    # Calculate inverse of the covariant matrix

    effEnergy_err = np.linalg.inv( np.cov( effEnergy_to_fit, rowvar=False ) 
                                   * ( binNum - 1 ) )

    if rank == 0:

        # effEnergy_avg[t]

        effEnergy_avg = np.average( effEnergy_to_fit, axis=0 )
    
        #leastSq_avg = least_squares( twoStateErrorFunction_effEnergy, fitParams,
        #                             args = ( t_to_fit, T, effEnergy_avg, effEnergy_err ),
        #                             method="lm" )
        #leastSq_avg = minimize( twoStateCostFunction_effEnergy, fitParams,
        #                        args = ( t_to_fit, T, effEnergy_avg, effEnergy_err ),
        #                        method="BFGS" )
        leastSq_avg = differential_evolution( twoStateCostFunction_effEnergy, 
                                              fitParams, ( t_to_fit, T, 
                                                           effEnergy_avg, 
                                                           effEnergy_err ),
                                              tol=0.01 )

        #fitParams = leastSq_avg.x
        fitParams = np.array( [ [ max( leastSq_avg.x[ 0 ] - 0.1, 0.0 ),
                                  min( leastSq_avg.x[ 0 ] + 0.1, 1.0 ) ],
                                [ max( leastSq_avg.x[ 1 ] - 0.1, 0.0 ),
                                  leastSq_avg.x[ 1 ] + 0.1 ],
                                [ max( leastSq_avg.x[ 2 ] - 0.1, 0.0 ),
                                  leastSq_avg.x[ 2 ] + 0.1 ] ] )

    comm.Bcast( fitParams, root=0 )

    # Find fit parameters for each bins

    # Loop over bins
    for b, ib in zip( binList_loc, range( binNum_loc ) ):

        #leastSq = least_squares( twoStateErrorFunction_effEnergy, fitParams,
        #                         args = ( t_to_fit, T, effEnergy_to_fit[ b, : ],
        #                                  effEnergy_err ),
        #                         method="lm" )
        #leastSq = minimize( twoStateCostFunction_effEnergy, fitParams,
        #                        args = ( t_to_fit, T, effEnergy_to_fit[ b, : ], 
        #                                 effEnergy_err ),
        #                        method="BFGS" )
        leastSq = differential_evolution( twoStateCostFunction_effEnergy, 
                                          fitParams, ( t_to_fit, T, 
                                                       effEnergy_to_fit[ b, : ], 
                                                       effEnergy_err ),
                                          tol=0.0001 )

        fit_loc[ ib ] = np.array( leastSq.x )
        
        #chiSq_loc[ ib ] = leastSq.cost
        chiSq_loc[ ib ] = leastSq.fun

    # End loop over bins

    fit = np.zeros( ( binNum, ) + fit_loc.shape[ 1: ] )
    chiSq = np.zeros( ( binNum, ) + chiSq_loc.shape[ 1: ] )

    comm.Allgatherv( fit_loc, 
                     [ fit, 
                       recvCount * np.prod( fit.shape[ 1: ] ),
                       recvOffset * np.prod( fit.shape[ 1: ] ),
                       MPI.DOUBLE ] )
    comm.Allgatherv( chiSq_loc, 
                     [ chiSq, 
                       recvCount * np.prod( chiSq.shape[ 1: ] ),
                       recvOffset * np.prod( chiSq.shape[ 1: ] ),
                       MPI.DOUBLE ] )

    chiSq = chiSq / dof

    return fit, chiSq


# Fit three-point functions to a two-state fit.

# threep: three-point functions to be fit
# ti_to_fit: Values of insertion time to be fit over
# tsink: list of tsink values to fit over
# E0: ground state energy value calculated from two-state function fit
# E1: first excited state energy value calculated from two-state function fit
# T: Time dimension length for ensemble

def twoStateFit_threep( threep, ti_to_fit, tsink, E0, E1,
                        mpi_confs_info ):

    comm = mpi_confs_info[ 'comm' ]
    rank = mpi_confs_info[ 'rank' ]
    binNum = mpi_confs_info[ 'binNum_glob' ]
    binNum_loc = mpi_confs_info[ 'binNum_loc' ]
    binList_loc = mpi_confs_info[ 'binList_loc' ]
    recvCount = mpi_confs_info[ 'recvCount' ]
    recvOffset = mpi_confs_info[ 'recvOffset' ]

    assert threep.shape[ 1 ] == binNum, \
        "First dimension size of two-point functions " \
        + str( threep.shape[ 1 ] ) \
        + " does not match number of bins " \
        + str( binNum ) + "."

    tsinkNum = len( tsink )

    assert tsinkNum == len( threep ), \
        "Number of tsink's does not match " \
        + "number of three-point function datasets."

    ti_flat = np.array( [] )
    tsink_flat = np.array( [] )

    # Loop over tsinks
    for its, ts in zip( range( tsinkNum ), tsink ):

        ti = ti_to_fit[ its ]

        ti_flat = np.append( ti_flat, ti )

        tsink_flat = np.append( tsink_flat, 
                                np.repeat( ts, len( ti ) ) )

    # End loop over tsink

    paramNum = 3
    dof = len( ti_flat ) - paramNum

    # threep[ ts, b, t ]

    threep_to_fit = fncs.initEmptyList( tsinkNum, 1 )

    # threep_avg[ts, t]

    threep_to_fit_avg = fncs.initEmptyList( tsinkNum, 1 )
    threep_to_fit_err = fncs.initEmptyList( tsinkNum, 1 )
    threep_err_flat = np.array( [] )

    # Set three-point functions to fit based on ti_to_fit

    for its, ts in zip( range( tsinkNum ), tsink ):
 
        threep_to_fit[ its ] = threep[ its ].take( ti_to_fit[ its ], 
                                                   axis=-1 )
        threep_to_fit_avg[ its ] = np.average( threep_to_fit[ its ],
                                               axis=0 )

        threep_to_fit_err[ its ] = fncs.calcError( threep_to_fit[ its ],
                                                   binNum )

        threep_err_flat = np.append( threep_err_flat,
                                     threep_to_fit_err[ its ] )

    # End loop over tsink

    E0_avg = np.average( E0 )
    E1_avg = np.average( E1 )

    # Find fit parameters of mean values to use as initial guess

    #a00 = -10.0 ** -5
    #a01 = -10.0 ** -4
    #a11 = -10.0 ** -4
    a00 = [ -10 ** -2, 10 ** 2 ]
    a01 = [ -10 ** -2, 10 ** 2 ]
    a11 = [ -10 ** -2, 10 ** 2 ]
    #a00 = [ -10 ** -2, 0.0 ]
    #a01 = [ -10 ** -2, 0.0 ]
    #a11 = [ -10 ** -2, 0.0 ]

    fitParams = np.array( [ a00, a01, a11 ] )

    if rank == 0:

        threep_flat = np.array( [] )

        # Loop over tsink
        for its in range( tsinkNum ):

            threep_flat = np.append( threep_flat, 
                                     threep_to_fit_avg[ its ] )

        # End loop over tsink

        #leastSq_avg = least_squares( twoStateErrorFunction_threep, 
        #                             fitParams,
        #                             args = ( ti_flat,
        #                                      tsink_flat,
        #                                      threep_flat,
        #                                      threep_err_flat,
        #                                      E0_avg, E1_avg ),
        #                             method="lm" )
        #leastSq_avg = minimize( twoStateCostFunction_threep, 
        #                    fitParams,
        #                    args = ( ti_flat, 
        #                             tsink_flat,
        #                             threep_flat,
        #                             threep_err_flat,
        #                             E0_avg, E1_avg ),
        #                    method="BFGS" )
        leastSq_avg = differential_evolution( twoStateCostFunction_threep,
                                              fitParams, ( ti_flat, 
                                                           tsink_flat,
                                                           threep_flat,
                                                           threep_err_flat,
                                                           E0_avg, E1_avg ),
                                              tol=0.01 )

        #fitParams = leastSq_avg.x

        fitParams_avg = leastSq_avg.x

        fitParams = np.zeros( ( paramNum, 2 ) )

        # a00
        
        if fitParams_avg[ 0 ] > 0:

            fitParams[ 0 ] = [ max( fitParams_avg[ 0 ] - 10 ** -5, 0.0 ), 
                               fitParams_avg[ 0 ] + 10 ** -5 ]

        elif fitParams_avg[ 0 ] < 0:

            fitParams[ 0 ] = [ fitParams_avg[ 0 ] - 10 ** -5,
                               min( fitParams_avg[ 0 ] + 10 ** -5, 0.0 ) ]

        else: # fitParams_avg[ 0 ] == 0

            fitParams[ 0 ] = [ fitParams_avg[ 0 ] - 10 ** -5,
                               fitParams_avg[ 0 ] + 10 ** -5 ]

        # a01

        if fitParams_avg[ 1 ] > 0:

            fitParams[ 1 ] = [ max( fitParams_avg[ 1 ] - 10 ** -4, 0.0 ), 
                               fitParams_avg[ 1 ] + 10 ** -4 ]

        elif fitParams_avg[ 1 ] < 0:

            fitParams[ 1 ] = [ fitParams_avg[ 1 ] - 10 ** -4,
                               min( fitParams_avg[ 1 ] + 10 ** -4, 0.0 ) ]

        else: # fitParams_avg[ 1 ] == 0

            fitParams[ 1 ] = [ fitParams_avg[ 1 ] - 10 ** -4,
                               fitParams_avg[ 1 ] + 10 ** -4 ]

        # a11

        if fitParams_avg[ 2 ] > 0:

            fitParams[ 2 ] = [ max( fitParams_avg[ 2 ] - 10 ** -4, 0.0 ), 
                               fitParams_avg[ 2 ] + 10 ** -4 ]

        elif fitParams_avg[ 2 ] < 0:

            fitParams[ 2 ] = [ fitParams_avg[ 2 ] - 10 ** -4,
                               min( fitParams_avg[ 2 ] + 10 ** -4, 0.0 ) ]

        else: # fitParams_avg[ 2 ] == 0

            fitParams[ 2 ] = [ fitParams_avg[ 2 ] - 10 ** -4,
                               fitParams_avg[ 2 ] + 10 ** -4 ]

    comm.Bcast( fitParams, root=0 )

    # Find fit parameters for each bin

    # fit[b]

    fit_loc = np.zeros( ( binNum_loc, paramNum ) )
    chiSq_loc = np.zeros( binNum_loc )

    # Loop over bins
    for b, ib in zip( binList_loc, range( binNum_loc ) ):

        threep_flat = np.array( [] )

        # Loop over tsink
        for its in range( tsinkNum ):

            threep_flat = np.append( threep_flat, 
                                     threep_to_fit[ its ][ b, : ] )

        # End loop over tsink

        #leastSq = least_squares( twoStateErrorFunction_threep, 
        #                         fitParams,
        #                         args = ( ti_flat, tsink_flat,
        #                                  threep_flat,
        #                                  threep_err_flat,
        #                                  E0[ b ], E1[ b ] ),
        #                         method="lm" )
        #leastSq = minimize( twoStateCostFunction_threep, 
        #                    fitParams,
        #                    args = ( ti_flat, 
        #                             tsink_flat,
        #                             threep_flat,
        #                             threep_err_flat,
        #                             E0[ b ], E1[ b ] ),
        #                    method="BFGS" )
        leastSq = differential_evolution( twoStateCostFunction_threep, 
                                          fitParams,
                                          args = ( ti_flat, tsink_flat,
                                                   threep_flat,
                                                   threep_err_flat,
                                                   E0[ b ], E1[ b ] ),
                                          tol=0.0001 )

        fit_loc[ ib ] = leastSq.x
        chiSq_loc[ ib ] = leastSq.fun
        #chiSq_loc[ ib ] = leastSq.cost

    # End loop over bins

    fit = np.zeros( ( binNum, ) + fit_loc.shape[ 1: ] )
    chiSq = np.zeros( ( binNum, ) + chiSq_loc.shape[ 1: ] )

    comm.Allgatherv( fit_loc, 
                     [ fit, 
                       recvCount * np.prod( fit.shape[ 1: ] ),
                       recvOffset * np.prod( fit.shape[ 1: ] ),
                       MPI.DOUBLE ] )
    comm.Allgatherv( chiSq_loc, 
                     [ chiSq, 
                       recvCount * np.prod( chiSq.shape[ 1: ] ),
                       recvOffset * np.prod( chiSq.shape[ 1: ] ),
                       MPI.DOUBLE ] )
    
    chiSq = np.array( chiSq ) / dof
    
    return fit, chiSq


# Fit three-point functions to a two-state fit.

# threep: three-point functions to be fit
# ti_to_fit: Values of insertion time to be fit over
# tsink: list of tsink values to fit over
# E0: ground state energy value calculated from two-state function fit
# E1: first excited state energy value calculated from two-state function fit

def twoStateFit_threep_momTransfer( threep_loc, tsink, m0E0, E1,
                                    qList, qSq_where, neglect, L,
                                    dispRel, mpi_info ):

    # threep_loc[ ts, b_loc, q, ratio, t ]
    # ti_to_fit[ ts, t_to_fit ]
    # tsink[ ts ]
    # m0[ b ]
    # E1[ b, q ]
    # L
    # mpi_info

    if dispRel:

        m0 = m0E0

    else:

        m0 = m0E0[ :, 0 ]

    comm = mpi_info[ 'comm' ]
    rank = mpi_info[ 'rank' ]
    binNum = mpi_info[ 'binNum_glob' ]
    binNum_loc = mpi_info[ 'binNum_loc' ]
    binList_loc = mpi_info[ 'binList_loc' ]
    recvCount = mpi_info[ 'recvCount' ]
    recvOffset = mpi_info[ 'recvOffset' ]

    tsinkNum = len( tsink )

    assert tsinkNum == len( threep_loc ), \
        "Number of tsink's does not match " \
        + "number of three-point function datasets."

    qNum = threep_loc.shape[ 2 ]
    ratioNum = threep_loc.shape[ 3 ]

    ti_flat = np.array( [] )
    tsink_flat = np.array( [] )    

    ti_to_fit = fncs.initEmptyList( tsinkNum, 1 )
    
    # Loop over tsinks
    for its, ts in zip( range( tsinkNum ), tsink ):

        ti_to_fit[ its ] = np.arange( neglect, ts - neglect + 1 )
                
        ti = ti_to_fit[ its ]

        ti_flat = np.append( ti_flat, ti )

        tsink_flat = np.append( tsink_flat, 
                                np.repeat( ts, len( ti ) ) )

    # End loop over tsink

    paramNum = 4
    dof = len( ti_flat ) - paramNum

    # Resulting fit parameters
    # results[ b, q, r, [ A00, A01, A10, A11 ]  ]

    results = np.zeros( ( binNum, qNum, ratioNum, paramNum ) )

    m1 = E1[ :, 0 ]

    m0_avg = np.average( m0 )
    m1_avg = np.average( m1 )

    # Fit q=0 threep, where A01 = A10

    threep_q = np.zeros( ( tsinkNum, binNum, ) + threep_loc.shape[ 3: ] )

    for its in range( tsinkNum ):

        threep_loc_buffer = np.array( threep_loc[ its, :, 0, :, : ],
                                      order='c' )

        comm.Allgatherv( threep_loc_buffer,
                         [ threep_q[ its ],
                         recvCount \
                         * np.prod( threep_loc.shape[ 3: ] ),
                         recvOffset \
                         * np.prod( threep_loc.shape[ 3: ] ),
                         MPI.DOUBLE ] )

    for ir in range( ratioNum ):

        fitParams_q, dummy \
            = twoStateFit_threep( threep_q[ :, :, ir, : ],
                                  ti_to_fit, tsink,
                                  m0, m1, mpi_info )

        results[ :, 0, ir, 0 ] = fitParams_q[ :, 0 ]
        results[ :, 0, ir, 1 ] = fitParams_q[ :, 1 ]
        results[ :, 0, ir, 2 ] = fitParams_q[ :, 1 ]
        results[ :, 0, ir, 3 ] = fitParams_q[ :, 2 ]

    # Fit the rest of threeps

    for q, iq in zip( qList[ 1: ], range( 1, qNum ) ):
        
        # E0[ b ]

        if dispRel:

            E0 = pq.energy( m0, np.dot( q, q ), L )

        else:
        
            E0 = m0E0[ :, qSq_where[ iq ] ]

        E0_avg = np.average( E0 )
        E1_avg = np.average( E1[ :, qSq_where[ iq ] ] )

        threep_q = np.zeros( ( tsinkNum, binNum, ) + threep_loc.shape[ 3: ] )

        for its in range( tsinkNum ):

            threep_loc_buffer = np.array( threep_loc[ its, :, iq, :, : ],
                                          order='c' )

            comm.Allgatherv( threep_loc_buffer,
                             [ threep_q[ its ],
                               recvCount \
                               * np.prod( threep_loc.shape[ 3: ] ),
                               recvOffset \
                               * np.prod( threep_loc.shape[ 3: ] ),
                               MPI.DOUBLE ] )

        for ir in range( ratioNum ):

            # threep_to_fit[ ts, b_loc, q, r, t_to_fit ]

            threep_to_fit = fncs.initEmptyList( tsinkNum, 1 )

            # threep_avg[ts, t]

            threep_to_fit_avg = fncs.initEmptyList( tsinkNum, 1 )
            threep_to_fit_err = fncs.initEmptyList( tsinkNum, 1 )
            threep_err_flat = np.array( [] )

            # Set three-point functions to fit based on ti_to_fit

            for its, ts in zip( range( tsinkNum ), tsink ):
 
                threep_to_fit[ its ] \
                    = threep_q[ its, :, ir ].take( ti_to_fit[ its ], 
                                                   axis=-1 )
                threep_to_fit_avg[ its ] = np.average( threep_to_fit[ its ],
                                                       axis=0 )

                threep_to_fit_err[ its ] = fncs.calcError( threep_to_fit[ its ],
                                                           binNum )

                threep_err_flat = np.append( threep_err_flat,
                                             threep_to_fit_err[ its ] )

            # End loop over tsink

            # Find fit parameters of mean values to use as initial guess

            a00 = 10.0 ** -5
            a01 = 10.0 ** -6
            a10 = 10.0 ** -6
            a11 = 10.0 ** -6

            fitParams = np.array( [ a00, a01, a10, a11 ] )

            if rank == 0:

                threep_flat = np.array( [] )

                # Loop over tsink
                for its in range( tsinkNum ):

                    threep_flat = np.append( threep_flat, 
                                             threep_to_fit_avg[ its ] )

                # End loop over tsink

                #leastSq_avg = least_squares( twoStateErrorFunction_threep, 
                #                             fitParams,
                #                             args = ( ti_flat,
                #                                      tsink_flat,
                #                                      threep_flat,
                #                                      threep_err_flat,
                #                                      E0_avg, E1_avg ),
                #                             method="lm" )
                leastSq_avg \
                    = minimize( twoStateCostFunction_threep_momTransfer, 
                                fitParams,
                                args = ( ti_flat, 
                                         tsink_flat,
                                         threep_flat,
                                         threep_err_flat,
                                         m0_avg, E0_avg,
                                         m1_avg, E1_avg ),
                                method="BFGS" )
                
                fitParams = leastSq_avg.x

            # End first process

            comm.Bcast( fitParams, root=0 )

            # Find fit parameters for each bin
            
            # fit[ b ]

            fit_loc = np.zeros( ( binNum_loc, paramNum ) )
            chiSq_loc = np.zeros( binNum_loc )

            # Loop over bins
            for b, ib in zip( binList_loc, range( binNum_loc ) ):

                threep_flat = np.array( [] )

                # Loop over tsink
                for its in range( tsinkNum ):

                    threep_flat = np.append( threep_flat, 
                                             threep_to_fit[ its ][ b, : ] )

                # End loop over tsink

                #leastSq = least_squares( twoStateErrorFunction_threep, 
                #                         fitParams,
                #                         args = ( ti_flat, tsink_flat,
                #                                  threep_flat,
                #                                  threep_err_flat,
                #                                  E0[ b ], E1[ b ] ),
                #                         method="lm" )

                leastSq \
                    = minimize( twoStateCostFunction_threep_momTransfer, 
                                fitParams,
                                args = ( ti_flat, 
                                         tsink_flat,
                                         threep_flat,
                                         threep_err_flat,
                                         m0[ b ], E0[ b ],
                                         m1[ b ], E1[ b, qSq_where[ iq ] ] ),
                                method="BFGS" )

                fit_loc[ ib ] = leastSq.x
                chiSq_loc[ ib ] = leastSq.fun
                #chiSq_loc[ ib ] = leastSq.cost

            # End loop over bins

            #fit_loc = np.array( fit_loc, order='c' )

            results_buffer = np.zeros( results[ :, iq, ir, : ].shape,
                                       order='c' )

            comm.Allgatherv( fit_loc, 
                             [ results_buffer,
                               recvCount * np.prod( fit_loc.shape[ 1: ] ),
                               recvOffset * np.prod( fit_loc.shape[ 1: ] ),
                               MPI.DOUBLE ] )

            results[ :, iq, ir, : ] = results_buffer

            #mpi_fncs.mpiPrint( "Fit two-point functions to two-state fit " \
            #                   + "at q=({:+},{:+},{:+})".format( q[ 0 ],
            #                                                     q[ 1 ],
            #                                                     q[ 2 ] ),
            #                   mpi_info )
            
        # End loop over ratio
    # End loop over q
    
    return results


def twoStateCostFunction_twop( fitParams, tsink, T, twop, sigma ):

    if sigma.ndim == 1: # sigma is standard deviations

        return np.sum( twoStateErrorFunction_twop( fitParams, 
                                                   tsink, 
                                                   T, twop, 
                                                   sigma ) ** 2 )
    
    elif sigma.ndim == 2: # sigma is inverse covariant matrix

        r = twoStateResidual_twop( fitParams, tsink, T, twop )

        return r.T @ sigma @ r

    else: # Unsupported

        return

    
def twoStateCostFunction_twop_dispRel( fitParams, E0, tsink, T, twop, sigma ):

    if sigma.ndim == 1: # sigma is standard deviations

        return np.sum( twoStateErrorFunction_twop_dispRel( fitParams, 
                                                           E0,
                                                           tsink, 
                                                           T, twop, 
                                                           sigma ) ** 2 )
    
    elif sigma.ndim == 2: # sigma is inverse covariant matrix

        r = twoStateResidual_twop_dispRel( fitParams, E0, tsink, T, twop )

        return r.T @ sigma @ r

    else: # Unsupported

        return

    
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
    

def twoStateErrorFunction_twop_dispRel( fitParams, E0, tsink, T, twop, twop_err ):

    c0 = fitParams[ 0 ]
    c1 = fitParams[ 1 ]
    E1 = fitParams[ 2 ]

    twopErr = np.array( ( twoStateTwop( tsink, T, c0, c1, E0, E1 ) \
                          - twop ) / twop_err )
    
    return twopErr
    

def twoStateResidual_twop( fitParams, tsink, T, twop ):

    c0 = fitParams[ 0 ]
    c1 = fitParams[ 1 ]
    E0 = fitParams[ 2 ]
    E1 = fitParams[ 3 ]

    #return np.array( twoStateTwop( tsink, T, c0, c1, E0, E1 ) - twop )
    return np.array( twoStateTwop_forcePositive( tsink, T,
                                                 c0, c1, E0, E1 )
                     - twop )
    

def twoStateResidual_twop_dispRel( fitParams, E0, tsink, T, twop ):

    c0 = fitParams[ 0 ]
    c1 = fitParams[ 1 ]
    E1 = fitParams[ 2 ]

    return np.array( twoStateTwop_forcePositive( tsink, T, c0,
                                                 c1, E0, E1 )
                     - twop )
    

def twoStateCostFunction_effEnergy( fitParams, tsink, T, 
                                    effEnergy, sigma ):

    if sigma.ndim == 1: # sigma is standard deviations

        return np.sum( twoStateErrorFunction_effEnergy( fitParams, 
                                                        tsink, 
                                                        T, effEnergy, 
                                                        sigma ) ** 2 )

    elif sigma.ndim == 2: # sigma is inverse covariant matrix

        r = twoStateResidual_effEnergy( fitParams, tsink, T, effEnergy )

        return r.T @ sigma @ r

    else: # Unsupported

        return
    

def twoStateErrorFunction_effEnergy( fitParams, tsink, T,
                                     effEnergy, effEnergy_err ):

    c = fitParams[ 0 ]
    E0 = fitParams[ 1 ]
    E1 = fitParams[ 2 ]

    twopErr = np.array( ( twoStateEffEnergy( tsink, T, c, E0, E1 ) \
                          - effEnergy ) / effEnergy_err )
    
    return twopErr
    

def twoStateResidual_effEnergy( fitParams, tsink, T, effEnergy ):

    c = fitParams[ 0 ]
    E0 = fitParams[ 1 ]
    E1 = fitParams[ 2 ]

    return np.array( twoStateEffEnergy( tsink, T, c, E0, E1 ) 
                     - effEnergy )
    

def twoStateCostFunction_threep( fitParams, ti, tsink,
                                 threep, threep_err, E0, E1 ):

    return np.sum( twoStateErrorFunction_threep( fitParams, ti, tsink,
                                                 threep, threep_err,
                                                 E0, E1 ) ** 2 )


def twoStateCostFunction_threep_momTransfer( fitParams, ti, tsink,
                                             threep, threep_err,
                                             m0, E0, m1, E1 ):

    return np.sum( twoStateErrorFunction_threep_momTransfer( fitParams,
                                                             ti, tsink,
                                                             threep,
                                                             threep_err,
                                                             m0, E0,
                                                             m1, E1 ) ** 2 )


# Calculate the difference between three-point function values of the data 
# and calculated from the two-state fit divided by the jackknife errors
# of the data

# fitParams: Parameters of fit (a00, a01, a11 )
# ti: insertion time values to fit over
# tsink: tsink values to fit over
# threep: three-point functions to fit
# threep_err: jacckife errors associated with three-point functions
# E0: ground state energy value calculated from two-state function fit
# E1: first excited state energy value calculated from two-state function fit

def twoStateErrorFunction_threep( fitParams, ti, tsink,
                                  threep, threep_err, E0, E1):

    a00 = fitParams[ 0 ]
    a01 = fitParams[ 1 ]
    a11 = fitParams[ 2 ]

    # threepErr[ ts * ti ]

    threepErr = ( twoStateThreep( ti, tsink,
                                  a00, a01, a11,
                                  E0, E1 )
                  - threep ) / threep_err

    return np.array( threepErr )
    

def twoStateErrorFunction_threep_momTransfer( fitParams, ti, tsink,
                                              threep, threep_err,
                                              m0, E0, m1,  E1):

    a00 = fitParams[ 0 ]
    a01 = fitParams[ 1 ]
    a10 = fitParams[ 2 ]
    a11 = fitParams[ 3 ]

    # threepErr[ ts * ti ]

    threepErr = ( twoStateThreep_momTransfer( ti, tsink,
                                              a00, a01, a10, a11,
                                              m0, E0, m1, E1 )
                  - threep ) / threep_err

    return np.array( threepErr )
    

# Calculate three-point function from given two-state fit parameters and time values

# ti: insertion time value
# tsink: tsink value
# a00: amplitude of ground state term (fit parameter)
# a01: amplitude of mixed state terms (fit parameter)
# a11: amplitude of first excited state term (fit parameter)
# E0: ground state energy value calculated from two-state function fit
# E1: first excited state energy value calculated from two-state function fit

def twoStateThreep( ti, tsink, a00, a01, a11, E0, E1 ):

    return a00 * np.exp( -E0 * tsink ) \
        + a01 * np.exp( -E0 * ( tsink - ti ) - E1 * ti ) \
        + a01 * np.exp( -E1 * ( tsink - ti ) - E0 * ti ) \
        + a11 * np.exp( -E1 * tsink )


def twoStateThreep_momTransfer( ti, tsink,
                                a00, a01, a10, a11,
                                m0, E0, m1, E1 ):

    return a00 * np.exp( -m0 * ( tsink - ti ) - E0 * ti ) \
        + a01 * np.exp( -m0 * ( tsink - ti ) - E1 * ti ) \
        + a10 * np.exp( -m1 * ( tsink - ti ) - E0 * ti ) \
        + a11 * np.exp( -m1 * ( tsink - ti ) - E1 * ti )


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


def twoStateTwop_forcePositive( tsink, T, c0, c1, E0, E1 ):

    return np.abs( c0 ) * ( np.exp( -np.abs( E0 ) * tsink ) \
                  + np.exp( -np.abs( E0 ) * ( T - tsink ) ) ) \
        + np.abs( c1 ) * ( np.exp( -np.abs( E1 ) * tsink ) \
                 + np.exp( -np.abs( E1 ) * ( T - tsink ) ) )


def twoStateEffEnergy( tsink, T, c, E0, E1 ):

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


def fitMellinMoments( moments, paramNum ):

    # moments[ avgX, avgX2, ... ][ b ]
    
    binNum = moments.shape[ -1 ]

    dof = moments.shape[ 0 ] - paramNum + 1

    # Find fit parameters of mean values to use as initial guess

    moments_avg = np.average( moments, axis=-1 )

    moments_err = fncs.calcError( moments, binNum, axis=-1 )

    a = 1.0
    b = 1.0
    c = 1.0

    method = "CG"
    #method = "TNC"

    if paramNum == 2:

        fitParams_init = np.array( [ a, b ] )

        bounds = ( ( None, 0 ), ( None, None ) )

    elif paramNum == 3:

        fitParams_init = np.array( [ a, b, c ] )

        bounds = ( ( None, 0 ), ( None, None ), ( None, None ) )

    leastSq_avg = minimize( mellinMomentCostFunction, 
                            fitParams_init,
                            args = ( moments_avg,
                                     moments_err ),
                            method=method,
                            bounds=bounds )
    
    fitParams_avg = leastSq_avg.x

    # Find fit parameters for each bin

    fitParams = np.zeros( ( binNum, paramNum ) )
    chiSq = np.zeros( binNum )

    # Loop over bins
    for ib in range( binNum ):

        leastSq = minimize( mellinMomentCostFunction, 
                            fitParams_avg,
                            args = ( moments[ :, ib ],
                                     moments_err ),
                            method=method,
                            bounds=bounds )
        
        fitParams[ ib ] = leastSq.x
        chiSq[ ib ] = leastSq.fun

    # End loop over bins

    return fitParams, chiSq / dof

def mellinMomentCostFunction( fitParams,
                              moments, moments_err ):

    a = fitParams[ 0 ]
    b = fitParams[ 1 ]

    if len( fitParams ) == 3:

        c = fitParams[ 2 ]

    else:

        c = 0.

    momentsNum = len( moments )

    avgX = moments[ 0 ]
    avgX2 = moments[ 1 ]

    avgX_err = moments_err[ 0 ]
    avgX2_err = moments_err[ 1 ]

    if momentsNum >= 3:

        avgX3 = moments[ 2 ]
        avgX3_err = moments_err[ 2 ]

    if momentsNum >= 4:

       avgX4 = moments[ 3 ]
       avgX4_err = moments_err[ 3 ]
        
    if momentsNum >= 5:

        avgX5 = moments[ 4 ]
        avgX5_err = moments_err[ 4 ]

    if momentsNum >= 6:

        avgX6 = moments[ 5 ]
        avgX6_err = moments_err[ 5 ]

    avgXErrorFunction = ( avgXFit( a, b, c ) - avgX ) / avgX_err
    avgX2ErrorFunction = ( avgX2Fit( a, b, c ) - avgX2 ) / avgX2_err

    if momentsNum >= 3:

        avgX3ErrorFunction = ( avgX3Fit( a, b, c ) - avgX3 ) / avgX3_err

    if momentsNum >= 4:

        #if avgX4_err == 0.:

        avgX4ErrorFunction = ( avgX4Fit( a, b, c ) - avgX4 ) / avgX3_err
    
        #else:

        #avgX4ErrorFunction = ( avgX4Fit( a, b, c ) - avgX4 ) / avgX4_err
    
    if momentsNum >= 5:

        avgX5ErrorFunction = ( avgX5Fit( a, b, c ) - avgX5 ) / avgX5_err
    
    if momentsNum >= 6:

        avgX6ErrorFunction = ( avgX6Fit( a, b, c ) - avgX6 ) / avgX6_err
    
    errorFunction = [ avgXErrorFunction ** 2,
                      avgX2ErrorFunction ** 2 ]
    
    if momentsNum >= 3:

        errorFunction.append( avgX3ErrorFunction ** 2 )

    if momentsNum >= 4:

        errorFunction.append( avgX4ErrorFunction ** 2 )

    if momentsNum >= 5:

        errorFunction.append( avgX5ErrorFunction ** 2 )

    if momentsNum >= 6:

        errorFunction.append( avgX6ErrorFunction ** 2 )

    return np.sum( errorFunction )


def avgXFit( a, b, c ):

    return ( ( 1 + a ) * ( 3 + a + b + ( 2 + a ) * c ) ) \
        / ( ( 3 + a + b ) * ( 2 + a + b + c + a * c ) )


def avgX2Fit( a, b, c ):

    return ( ( 1 + a ) * ( 2 + a ) * ( 4 + a + b + ( 3 + a ) * c ) ) \
        / ( ( 3 + a + b ) * ( 4 + a + b ) * ( 2 + a + b + c + a * c ) )


def avgX3Fit( a, b, c ):

    return ( ( 1 + a ) * ( 2 + a ) * ( 3 + a )
             * ( 5 + a + b + ( 4 + a ) * c ) ) \
        / ( ( 3 + a + b ) * ( 4 + a + b ) * ( 5 + a + b )
            * ( 2 + a + b + c + a * c ) )


def avgX4Fit( a, b, c ):

    return ( ( 1 + a ) * ( 2 + a ) * ( 3 + a ) * ( 4 + a )
             * ( 6 + a + b + ( 5 + a ) * c ) ) \
        / ( ( 3 + a + b ) * ( 4 + a + b ) * ( 5 + a + b )
            * ( 6 + a + b ) * ( 2 + a + b + ( 1 + a ) * c ) ) 


def avgX5Fit( a, b, c ):

    return ( ( 1 + a ) * ( 2 + a ) * ( 3 + a ) * ( 4 + a )
             * ( 5 + a ) * ( 7 + a + b + ( 6 + a ) * c ) ) \
        / ( ( 3 + a + b ) * ( 4 + a + b ) * ( 5 + a + b )
            * ( 6 + a + b ) * ( 7 + a + b )
            * ( 2 + a + b + ( 1 + a ) * c ) )


def avgX6Fit( a, b, c ):

    return ( ( 1 + a ) * ( 2 + a ) * ( 3 + a )
             * ( 4 + a ) * ( 5 + a ) * ( 6 + a )
             * ( 8 + a + b + ( 7 + a ) * c ) ) \
        / ( ( 3 + a + b ) * ( 4 + a + b ) *
            ( 5 + a + b ) * ( 6 + a + b ) *
            ( 7 + a + b) * ( 8 + a + b )
            * ( 2 + a + b + ( 1 + a ) * c ) )

# Fit two-point functions to a one-state fit.

# twop: Two-point functions to be fit
# twop_rangeStart: Starting t value to include in fit range
# twop_rangeEnd: Ending t value to include in fit range
# T: Time dimension length for ensemble

def oneStateFit_twop( twop, twop_rangeStart, twop_rangeEnd, T ):

    # twop[ b, t ]

    dof = twop_rangeEnd - twop_rangeStart + 1 - 2

    twop_to_fit = twop[ :, twop_rangeStart : \
                        twop_rangeEnd + 1 ]

    binNum = twop.shape[ 0 ]

    fit = fncs.initEmptyList( binNum, 1 )

    chiSq = fncs.initEmptyList( binNum, 1 )

    twop_avg = np.average( twop_to_fit, axis=0 )

    twop_err = fncs.calcError( twop_to_fit, binNum )
    
    t = np.array( range( twop_rangeStart, \
                         twop_rangeEnd + 1 ) )

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

        chiSq[ b ] = leastSq.cost

    # End loop over bins

    chiSq = np.array( chiSq ) / dof

    return np.array( fit ), np.array( chiSq )


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
    
    return G * ( np.exp( -E * tsink ) \
                 + np.exp( -E * ( T - tsink ) ) )


def fitFormFactor( vals, vals_err, tsink, plusMinus ):

    # vals[ b, p, Q, ratio, t ]
    # vals_err[ p, Q, ratio, t ]

    fit = np.zeros( vals.shape[ :-1 ] )
    chiSq = np.zeros( vals.shape[ :-1 ] )

    fitStart = tsink // 2 - plusMinus
    fitEnd = tsink // 2 + plusMinus

    # Loop over p_fin
    for ip in range( vals.shape[ 1 ] ):
        # Loop over Q
        for iq in range( vals.shape[ 2 ] ):
            # Loop over ratio
            for ir in range( vals.shape[ 3 ] ):

                fit[ :, ip, iq, ir ], chiSq[ :, ip, iq, ir ] \
                    = fitPlateau( vals[ :, ip, iq, ir ], \
                                  vals_err[ ip, iq, ir ], \
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

    binNum = G.shape[ 0 ]

    curve = np.zeros( ( binNum, 100 ) )

    ts = np.linspace( rangeStart, \
                      rangeEnd, 100 )

    for b in range( binNum ):
        for t in range( ts.shape[ -1 ] ):
                
            curve[ b, t ] = oneStateTwop( ts[ t ], T, \
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
                                    neglect, moment ):

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

        if moment == "avgX" or moment == 1:

            preFactor = pq.avgXKineFactor( mEff[ b ], \
                                           momSq, \
                                           L ) \

        else:

            preFactor = -1.0

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
                                    * preFactor \
                                    * twoStateThreep( ti[ ts, t ], \
                                                      tsink[ ts ], \
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
                                    ZvD1, firstTs, lastTs, moment ):

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
        
        if moment == "avgX" or moment == 1:

            preFactor = pq.avgXKineFactor( mEff[ b ], \
                                           momSq, \
                                           L ) \

        else:

            preFactor = -1.0

        for ts in range( len( tsink ) ):
                
            curve[ b, ts ] = ZvD1 \
                             * preFactor \
                             * twoStateThreep( tsink[ ts ] / 2, \
                                               tsink[ ts ], \
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


def calcPDFcurve( a, b, c, xNum ):

    # a[ b ]
    # b[ b ]
    # c[ b ]
    # xNum

    binNum = a.shape[ 0 ]

    curve = np.zeros( ( binNum, xNum ) )
    x = np.linspace( 0, 1, num=xNum )
            
    for ib in range( binNum ):
        for ix in range( xNum ):
            
            curve[ ib, ix ] = x[ ix ] * x[ ix ] ** a[ ib ] \
                              * ( 1 - x[ ix ] ) ** b[ ib ] \
                              * ( 1 + c[ ib ] * x[ ix ] ) \
                              / ( beta( a[ ib ] + 1, b[ ib ] + 1 ) 
                                  + c[ ib ] * beta( a[ ib ] + 2, b[ ib ] + 1 ) )
        
    return curve, x
