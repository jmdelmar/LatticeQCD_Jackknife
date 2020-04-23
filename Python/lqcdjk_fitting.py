import numpy as np
import functions as fncs
import physQuants as pq
import mpi_functions as mpi_fncs
from mpi4py import MPI
from scipy.optimize import least_squares, minimize, \
    differential_evolution, curve_fit

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

    chiSq = chiSq / dof

    return fit, chiSq


# Wrapper for numpy.polyfit to fit a plateau line to data in parallel

def fitPlateau_parallel( data, err, start, end, mpi_confs_info ):

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

    x = range( start, \
               end + 1 )

    fit_loc = np.zeros( binNum_loc )
    chiSq_loc = np.zeros( binNum_loc )

    # Loop over bins
    for b, ib in zip( binList_loc, range( binNum_loc ) ):

        fit_loc[ ib ], chiSq_loc[ ib ], \
            dum, dum, dum = np.polyfit( x, \
                                        data[ b, \
                                              start \
                                              : end + 1 ], 0, \
                                        w=err[ start \
                                               : end + 1 ] ** -1, \
                                        full=True )
        
    # End loop over bin

    fit = np.zeros( binNum )
    chiSq = np.zeros( binNum )

    comm.Gatherv( fit_loc, [ fit, recvCount, recvOffset, MPI.DOUBLE ], 
                  root=0 )
    comm.Gatherv( chiSq_loc, [ chiSq, recvCount, recvOffset, MPI.DOUBLE ], 
                  root=0 )

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

    effEnergy_err = fncs.calcError( effEnergy, binNum )

    plat_results = []
    twop_tsf_results = []
    effEnergy_tsf_results = []

    # Loop over plateau fit range starts
    for plat_rangeStart in range( 5, rangeEnd - 5 ):

        plat_fit = np.zeros( binNum )
        plat_chiSq = np.zeros( binNum )

        # Perform the plateau fit

        plat_fit, plat_chiSq, \
            = fitPlateau_parallel( effEnergy, effEnergy_err, \
                                   plat_rangeStart, rangeEnd, 
                                   mpi_confs_info )
            
        if rank == 0:

            plat_results.append( ( plat_fit, plat_chiSq, \
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

    effEnergy_err = fncs.calcError( effEnergy, binNum )

    if "plat_t_low_range" in kwargs \
       and None not in kwargs[ "plat_t_low_range" ]:

        plat_t_low_range = kwargs[ "plat_t_low_range" ]

    else:

        if rangeEnd - 15 > 15:

            plat_t_low_range = range( 8, rangeEnd - 15 )

        else:

            plat_t_low_range = range( 7, rangeEnd - 5 )            

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

        plat_fit, chiSq = fitPlateau_parallel( effEnergy, effEnergy_err,
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

                    return 1

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
                             twop_t_low, plat_t_low )

                # End if fit is good
            # End check fit

            else: # Return fit without checking

                return ( fitParams, chiSq, plat_fit,
                         twop_t_low, plat_t_low )

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

    paramNum = 4
    dof = rangeEnd - rangeStart + 1 - paramNum
    
    # Set two-point functions to fit based on fit range start and end

    twop_to_fit = twop[ :, rangeStart : \
                        rangeEnd + 1 ]

    # fit[b]

    fit_loc = np.zeros( ( binNum_loc, paramNum ) )
    chiSq_loc = np.zeros( binNum_loc )

    # Find fit parameters of mean values to use as initial guess

    #c0 = 10 ** -3
    #c1 = 10 ** -3
    #E0 = 0.1
    #E1 = 1.0
    c0 = [ 0.0, 10**-2 ]
    c1 = [ 0.0, 10**-2 ]
    E0 = [ 0.0, E_guess + 0.2 ]
    E1 = [ E_guess + 0.2, 2.0 ]
    
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
        #                             args = ( tsink, T, twop_avg, twop_err ),
        #                             method="lm" )
        #leastSq_avg = minimize( twoStateCostFunction_twop, fitParams, \
        #                        args = ( tsink, T, twop_avg, twop_err ), \
        #                        method="BFGS" )
        leastSq_avg = differential_evolution( twoStateCostFunction_twop, 
                                              fitParams, ( tsink, T, 
                                                           twop_avg, 
                                                           twop_err ),
                                              tol=0.01 )
        
        #fitParams = leastSq_avg.x
        fitParams = np.array( [ [ max( leastSq_avg.x[ 0 ] - 10**-4, 0.0 ), 
                                  leastSq_avg.x[ 0 ] + 10**-4 ],
                                [ max( leastSq_avg.x[ 1 ] - 10**-4, 0.0 ),
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
        #leastSq = minimize( twoStateCostFunction_twop, fitParams, \
        #                    args = ( tsink, T, twop_to_fit[ b, : ], 
        #                             twop_err ), \
        #                    method="BFGS" )
        leastSq = differential_evolution( twoStateCostFunction_twop, 
                                          fitParams, ( tsink, T, 
                                                       twop_to_fit[ b, : ], 
                                                       twop_err ),
                                          tol=0.0001 )

        fit_loc[ ib ] = leastSq.x
        chiSq_loc[ ib ] = leastSq.fun
        #chiSq_loc[ ib ] = leastSq.cost

    # End loop over bins

    fit = np.zeros( ( binNum, ) + fit_loc.shape[ 1: ] )
    chiSq = np.zeros( ( binNum, ) + chiSq_loc.shape[ 1: ] )

    comm.Gatherv( fit_loc, [ fit, 
                             recvCount * np.prod( fit.shape[ 1: ] ),
                             recvOffset * np.prod( fit.shape[ 1: ] ),
                             MPI.DOUBLE ], 
                  root=0 )
    comm.Gatherv( chiSq_loc, [ chiSq, 
                               recvCount * np.prod( chiSq.shape[ 1: ] ),
                               recvOffset * np.prod( chiSq.shape[ 1: ] ),
                               MPI.DOUBLE ], 
                  root=0 )

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

    comm.Gatherv( fit_loc, [ fit, 
                             recvCount * np.prod( fit.shape[ 1: ] ),
                             recvOffset * np.prod( fit.shape[ 1: ] ),
                             MPI.DOUBLE ], 
                  root=0 )
    comm.Gatherv( chiSq_loc, [ chiSq, 
                               recvCount * np.prod( chiSq.shape[ 1: ] ),
                               recvOffset * np.prod( chiSq.shape[ 1: ] ),
                               MPI.DOUBLE ], 
                  root=0 )

    chiSq = chiSq / dof

    return fit, chiSq


# Fit three-point functions to a two-state fit.

# threep: three-point functions to be fit
# ti_to_fit: Values of insertion time to be fit over
# tsink: list of tsink values to fit over
# E0: ground state energy value calculated from two-state function fit
# E1: first excited state energy value calculated from two-state function fit
# T: Time dimension length for ensemble

def twoStateFit_threep( threep, ti_to_fit, tsink, E0, E1, T,
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
    #a00 = [ -10 ** -2, 10 ** 2 ]
    #a01 = [ -10 ** -2, 10 ** 2 ]
    #a11 = [ -10 ** -2, 10 ** 2 ]
    a00 = [ -10 ** -2, 0.0 ]
    a01 = [ -10 ** -2, 0.0 ]
    a11 = [ -10 ** -2, 0.0 ]

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
        #                                      tsink_flat, T,
        #                                      threep_flat,
        #                                      threep_err_flat,
        #                                      E0_avg, E1_avg ),
        #                             method="lm" )
        #leastSq_avg = minimize( twoStateCostFunction_threep, 
        #                    fitParams,
        #                    args = ( ti_flat, 
        #                             tsink_flat, T, 
        #                             threep_flat,
        #                             threep_err_flat,
        #                             E0_avg, E1_avg ),
        #                    method="BFGS" )
        leastSq_avg = differential_evolution( twoStateCostFunction_threep,
                                              fitParams, ( ti_flat, 
                                                           tsink_flat, T, 
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
        #                         args = ( ti_flat, tsink_flat, T,
        #                                  threep_flat,
        #                                  threep_err_flat,
        #                                  E0[ b ], E1[ b ] ),
        #                         method="lm" )
        #leastSq = minimize( twoStateCostFunction_threep, 
        #                    fitParams,
        #                    args = ( ti_flat, 
        #                             tsink_flat, T, 
        #                             threep_flat,
        #                             threep_err_flat,
        #                             E0[ b ], E1[ b ] ),
        #                    method="BFGS" )
        leastSq = differential_evolution( twoStateCostFunction_threep, 
                                          fitParams,
                                          args = ( ti_flat, tsink_flat, T, \
                                          threep_flat, \
                                          threep_err_flat, \
                                          E0[ b ], E1[ b ] ), \
                                          tol=0.0001 )

        fit_loc[ ib ] = leastSq.x
        chiSq_loc[ ib ] = leastSq.fun
        #chiSq_loc[ ib ] = leastSq.cost

    # End loop over bins

    fit = np.zeros( ( binNum, ) + fit_loc.shape[ 1: ] )
    chiSq = np.zeros( ( binNum, ) + chiSq_loc.shape[ 1: ] )

    comm.Gatherv( fit_loc, [ fit, 
                             recvCount * np.prod( fit.shape[ 1: ] ),
                             recvOffset * np.prod( fit.shape[ 1: ] ),
                             MPI.DOUBLE ], 
                  root=0 )
    comm.Gatherv( chiSq_loc, [ chiSq, 
                               recvCount * np.prod( chiSq.shape[ 1: ] ),
                               recvOffset * np.prod( chiSq.shape[ 1: ] ),
                               MPI.DOUBLE ], 
                  root=0 )

    chiSq = np.array( chiSq ) / dof
    
    return fit, chiSq


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
    

def twoStateResidual_twop( fitParams, tsink, T, twop ):

    c0 = fitParams[ 0 ]
    c1 = fitParams[ 1 ]
    E0 = fitParams[ 2 ]
    E1 = fitParams[ 3 ]

    return np.array( twoStateTwop( tsink, T, c0, c1, E0, E1 ) - twop )
    

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
    

def twoStateCostFunction_threep( fitParams, ti, tsink, T, 
                                 threep, threep_err, E0, E1 ):

    return np.sum( twoStateErrorFunction_threep( fitParams, ti, tsink, T, 
                                                 threep, threep_err,
                                                 E0, E1 ) ** 2 )


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

    threepErr = ( twoStateThreep( ti, tsink, T,
                                  a00, a01, a11,
                                  E0, E1 )
                  - threep ) / threep_err

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

    return a00 * np.exp( -E0 * tsink ) \
        + a01 * np.exp( -E0 * ( tsink - ti ) - E1 * ti ) \
        + a01 * np.exp( -E1 * ( tsink - ti ) - E0 * ti ) \
        + a11 * np.exp( -E1 * tsink )
    """
    size = len( ti )

    assert len( tsink ) == size, \
        "Error (lqcdjk_fitting.twoStateThreep): " \
        + "shape insertion current array " + str( len( ti ) ) \
        + " does not match shape of tsink array" + str( len( tsink ) )

    threep = np.zeros( size )

    # Loop over elements
    for i in range( size ):

        t = ti[ i ]
        ts = tsink[ i ]

        if t < ts:
                
            threep[ i ] = a00 * np.exp( -E0 * ts ) \
                          + a01 * np.exp( -E0 * ( ts - t ) 
                                          - E1 * t ) \
                          + a01 * np.exp( -E1 * ( ts - t )
                                          - E0 * t ) \
                          + a11 * np.exp( -E1 * ts )
        
        else: # t > ts
                
            threep[ i ] = a00 * np.exp( -E0 * ( T - ts ) ) \
                          + a01 * np.exp( -E0 * ( T - t )
                                          - E1 * ( t - ts ) ) \
                          + a01 * np.exp( -E1 * ( T - t )
                                          - E0 * ( t - ts ) ) \
                          + a11 * np.exp( -E1 * ( T - ts ) )
                        
        # End t > ts
    # End loop over elements

    return threep
    """

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

    chiSq = []

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

        #chiSq.append( min.cost )
        chiSq.append( min.fun )

    # End loop over bins

    return np.array( fit ), np.array( chiSq )


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
