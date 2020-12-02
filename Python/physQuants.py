import numpy as np
import lqcdjk_fitting as fit
import functions as fncs
import mpi_functions as mpi_fncs

# E = sqrt( m^2 + p^2 )

def energy( mEff, pSq, L ):

    return np.sqrt( mEff ** 2 + ( 2.0 * np.pi / L ) ** 2 * pSq )


# KK = sqrt( 2E ( E + m ) )

def KK_nucleon( mEff, Qsq, L ):

    return np.sqrt( 2.0 * energy( mEff, Qsq, L ) \
                    * ( energy( mEff, Qsq, L ) \
                        + mEff ) )


# KK = C_1^-1 = 2 sqrt( EE' )

def KK_meson( mEff, pSq_ini, pSq_fin, L ):

    return 2.0 * np.sqrt( energy( mEff, pSq_ini, L )
                          * energy( mEff, pSq_fin, L ) )


def twopFit( c0, E0, t ):
#def twopFit( c0, c1, E0, E1, t ):

    return c0 * np.exp( -E0 * t )
    #return c0 * np.exp( -E0 * t ) + c1 * np.exp( -E1 * t )


def kineFactor_GE_GM( ratio_err, particle, flavor, mEff, p_fin, Q, L,
                      mpi_info ):

    # ratio_err[ p, Q, r ]
    # "particle"
    # "flavor"
    # mEff[ b ]
    # p_fin[ p, pi ]
    # momList[ Q, qi ]
    # L

    finalMomentaNum = ratio_err.shape[ 0 ]
    QNum = ratio_err.shape[ 1 ]
    ratioNum = ratio_err.shape[ -1 ]
    binNum = len( mEff )

    if p_fin.shape[ 0 ] != finalMomentaNum:

        error_template = "Error (kineFactor_GE_GM): " \
                         + "final momentum dimension " \
                         + "of ratio errors {} and " \
                         + "number of final momenta {} " \
                         + "do not match. "

        mpi_fncs.mpiError( error_template.format( finalMomentaNum,
                                                  p_fin.shape[ 0 ] ),
                           mpi_info )

    if Q.shape[ 0 ] != QNum:

        error_template = "Error (kineFactor_GE_GM): " \
                         + "momentum transfer dimension " \
                         + "of ratio errors {} and " \
                         + "number of momentum transfer {} " \
                         + "do not match. "

        mpi_fncs.mpiError( error_template.format( QNum,
                                                  Q.shape[ 0 ] ),
                           mpi_info )

    # kineFactor[ b, p, Q, ratio, [GE,GM] ]

    kineFactor = np.zeros( ( binNum, finalMomentaNum,
                             QNum, ratioNum, 2 ) )

    # Loop over bins
    for b in range( binNum ):
        # Loop over p_fin
        for p, ip in fncs.zipXandIndex( p_fin ):
            # Loop over Q
            for q, iq in fncs.zipXandIndex( Q ):

                if particle == "nucleon":

                    Qsq = np.dot( q, q )
                    
                    kineFactor[ b, ip, iq ] \
                        = [ [ ( energy( mEff[ b ],
                                        Qsq, L ) \
                                + mEff[ b ] ), 0 ],
                            [ -2.0 * np.pi / L * q[ 0 ], 0 ],
                            [ -2.0 * np.pi / L * q[ 1 ], 0 ],
                            [ -2.0 * np.pi / L * q[ 2 ], 0 ],
                            [ 0, -2.0 * np.pi / L * q[ 2 ] ],
                            [ 0, 2.0 * np.pi / L * q[ 1 ] ],
                            [ 0, 2.0 * np.pi / L * q[ 2 ] ],
                            [ 0, -2.0 * np.pi / L * q[ 0 ] ],
                            [ 0, -2.0 * np.pi / L * q[ 1 ] ],
                            [ 0, 2.0 * np.pi / L * q[ 0 ] ] ] \
                        / np.repeat( ratio_err[ ip, iq ],
                                     2).reshape( ratioNum, 2 ) \
                        / KK_nucleon( mEff[ b ], Qsq, L )
                    
                else: # particle == "meson"

                    p_ini = p - q

                    pSq_ini = np.dot( p_ini, p_ini )
                    pSq_fin = np.dot( p, p )

                    chargeSign = -1.0 if particle == "kaon" \
                                 and flavor == "s" else 1.0
                    """
                    if mpi_info[ 'rank' ] == 1 and b == 0:

                        print("p_f=({},{},{}), Q=({},{},{}), p_i=({},{},{}, p_f+p_i=({},{},{}))".format( p[ 0 ],
                                                                                                         p[ 1 ],
                                                                                                         p[ 2 ],
                                                                                                         q[ 0 ],
                                                                                                         q[ 1 ],
                                                                                                         q[ 2 ],
                                                                                                         p_ini[ 0 ],
                                                                                                         p_ini[ 1 ],
                                                                                                         p_ini[ 2 ],
                                                                                                         p[0]+p_ini[0],
                                                                                                         p[1]+p_ini[1],
                                                                                                         p[2]+p_ini[2]
                                                                                                     ) )

                        print( energy( mEff[ b ],
                                       pSq_ini,
                                       L )
                               + energy( mEff[ b ],
                                         pSq_fin,
                                         L ) )

                        print( chargeSign * 2.0 * np.pi / L
                               * ( p_ini[ 0 ] + p[ 0 ] ) )
                        print( chargeSign * 2.0 * np.pi / L
                               * ( p_ini[ 1 ] + p[ 1 ] ) )
                        print( chargeSign * 2.0 * np.pi / L
                               * ( p_ini[ 2 ] + p[ 2 ] ) )
                        print( KK_meson( mEff[ b ], pSq_ini, pSq_fin, L ) )
                        print(np.repeat( ratio_err[ ip, iq ],
                                         2).reshape( ratioNum, 2 ) )
                        print([ [ energy( mEff[ b ],
                                          pSq_ini,
                                          L )
                                  + energy( mEff[ b ],
                                            pSq_fin,
                                            L ),
                                  0 ], 
                                [ chargeSign * 2.0 * np.pi / L
                                  * ( p_ini[ 0 ] + p[ 0 ] ),
                                  0 ],
                                [ chargeSign * 2.0 * np.pi / L
                                  * ( p_ini[ 1 ] + p[ 1 ] ),
                                  0 ],
                                [ chargeSign * 2.0 * np.pi / L
                                  * ( p_ini[ 2 ] + p[ 2 ] ),
                                  0 ] ] \
                              / np.repeat( ratio_err[ ip, iq ],
                                           2).reshape( ratioNum, 2 ) \
                              / KK_meson( mEff[ b ], pSq_ini, pSq_fin, L ) )
                    """
                    kineFactor[ b, ip, iq ] \
                        = [ [ energy( mEff[ b ],
                                      pSq_ini,
                                      L )
                              + energy( mEff[ b ],
                                        pSq_fin,
                                        L ),
                              0 ], 
                            [ chargeSign * 2.0 * np.pi / L
                              * ( p_ini[ 0 ] + p[ 0 ] ),
                              0 ],
                            [ chargeSign * 2.0 * np.pi / L
                              * ( p_ini[ 1 ] + p[ 1 ] ),
                              0 ],
                            [ chargeSign * 2.0 * np.pi / L
                              * ( p_ini[ 2 ] + p[ 2 ] ),
                              0 ] ] \
                        / np.repeat( ratio_err[ ip, iq ],
                                     2).reshape( ratioNum, 2 ) \
                        / KK_meson( mEff[ b ], pSq_ini, pSq_fin, L )
                    
                # End if meson                                   
            # End loop over Q
        # End loop over p_fin
    # End loop over bins

    return kineFactor


def calc_gE_gM( decomp, ratio, ratio_err, Qsq_start, Qsq_end, mpi_info ):

    # decomp[ b, p, Q[ qsq ], ratio, [ GE, GM ] ]
    # ratio[ b, p, Q, ratio ]
    # ratio_err[ p, Q, ratio ]
    # Qsq_start
    # Qsq_end

    if decomp.ndim == 5:
        """
        mpi_fncs.mpiPrint( decomp[ ..., 0 ]
                           * ratio[ :, :,
                                    Qsq_start
                                    : Qsq_end + 1, : ] \
                           / ratio_err[ :, Qsq_start
                                        : Qsq_end
                                        + 1, : ],
                           mpi_info )
        """
        gE = np.sum( decomp[ ..., 0 ]
                     * ratio[ :, :,
                              Qsq_start
                              : Qsq_end + 1, : ]
                     / ratio_err[ :, Qsq_start
                                  : Qsq_end
                                  + 1, : ],
                     axis=(1,2,3) )
                     
        gM = np.sum( decomp[ ..., 1 ]
                     * ratio[ :, :,
                              Qsq_start
                              : Qsq_end + 1, : ]
                     / ratio_err[ :, Qsq_start
                                  : Qsq_end
                                  + 1, : ],
                     axis=(1,2,3) )
                     
    elif decomp.ndim == 4:

        gE = np.sum( decomp[ ..., 0 ]
                     * ratio[ :, :,
                              Qsq_start
                              : Qsq_end + 1 ]
                     / ratio_err[ :, Qsq_start
                                 : Qsq_end
                                 + 1 ],
                             axis=(1,2) )
             
        gM = np.sum( decomp[ ..., 1 ]
                     * ratio[ :, :,
                              Qsq_start
                              : Qsq_end + 1 ]
                     / ratio_err[ :, Qsq_start
                                  : Qsq_end
                                  + 1 ],
                     axis=(1,2) )
        
    return gE, gM


def calc_GE_GM( gE, gM, mEff, Qsq, L ):

    # gE[ b ], gM[ b ], mEff[ b ]
    # Qsq, L

    GE = gE \
         + ( energy( mEff, Qsq, L ) - mEff ) \
         / ( energy( mEff, Qsq, L ) + mEff ) * gM

    GM = 2 * mEff / ( energy( mEff, Qsq, L ) + mEff ) \
         * ( gM - gE )

    return GE, GM


# Convert Q^2 from units of (2pi/L)^2 to GeV^2

# Qsq: Q^2 values to be converted
# mEff: Effective mass of particle
# a: Lattice spacing of ensemble
# L: Spacial dimension length of ensemble

def convertQsqToGeV( Qsq, mEff, a, L ):

    Qsq_GeV = 2.0 * ( 0.197 / a ) ** 2 * mEff * ( energy( mEff, Qsq, L ) \
                                                  - mEff )

    return Qsq_GeV


# Calcuate the effective mass from two-point functions which have been
# symmetrized

# twop: Symmetrized two-point functions with last dimension as time

def mEffFromSymTwop( twop ):

    halfT = twop.shape[ -1 ]

    mEff = np.zeros( twop.shape )

    for t in range( 1, halfT - 1 ):

        mEff[ ..., t ] = 1.0 / 2.0 \
                         * np.log( ( twop[ ..., t - 1 ] \
                                     + np.sqrt(twop[ ..., \
                                                     t - 1 ] ** 2 \
                                               - twop[ ..., \
                                                       halfT - 1 ] ** 2) ) \
                                   / ( twop[ ..., t + 1 ] \
                                       + np.sqrt(twop[ ..., \
                                                       t + 1 ] ** 2 \
                                                 - twop[ ..., \
                                                         halfT - 1 ] ** 2) ))

    return mEff


# Calculate the effective mass from two-point functions

# twop: Two-point functions with last dimension as time

def mEff( twop ):

    mEff = np.zeros( twop.shape )

    # Loop through timestep, excluding the last timestep

    for t in range( twop.shape[ -1 ] - 1 ):

        mEff[ ..., t ] = np.log( twop[ ..., t ] / twop[ ..., t + 1 ] )

    # Calculate effective mass at last timestep, 
    # applying boundary conditions

    mEff[ ..., -1 ] = np.log( twop[ ..., -1 ] / twop[ ..., 0 ] )

    return mEff


# avgXKineFactor = 2E/(1/2*m^2-2*E^2)

def avgXKineFactor( mEff, momSq, L ):

    return 2.0 * energy( mEff, momSq, L ) \
        / ( 0.5 * mEff ** 2 \
            - 2.0 * energy( mEff, momSq, L ) ** 2 )


def avgX2KineFactor( mEff, momSq, L ):

    return -energy( mEff, momSq, L ) / mEff


# Calculate the quark momentum fraction <x> for three-point functions with
# zero final momentum.

# threep: Three-point functions with last dimension as time
# twop_tsink: Two-point funtion at Tsink
# mEff: Effective mass of particle

def calcAvgX( threep, twop_tsink, mEff ):

    # threep[ b, t ]
    # twop_tsink[ b ]
    # mEff[ b ]

    avgX = np.zeros( threep.shape )

    for t in range( threep.shape[ 1 ] ):
           
        avgX[ :, t ] = -4.0 / 3.0 / mEff * threep[ :, t ] / twop_tsink

    return avgX


def calcAvgX2( threep, twop_tsink, mEff, momSq, L ):

    # threep[ b, t ]
    # twop_tsink[ b ]
    # mEff[ b ]

    avgX2 = np.zeros( threep.shape )

    preFactor = avgX2KineFactor( mEff, momSq, L )

    for t in range( threep.shape[ 1 ] ):
           
        avgX2[ :, t ] = preFactor * threep[ :, t ] / twop_tsink

    return avgX2


# Calculate the quark momentum fraction <x> for three-point functions with
# finite final momentum.

# threep: Three-point functions with last dimension as time
# twop_tsink: Two-point funtion at Tsink
# mEff: Effective mass of particle
# momSq: Final momentum squared
# L: Spacial dimension length of ensemble

def calcAvgX_momBoost( threep, twop_tsink, mEff, momSq, L ):

    # threep[ b, t ]
    # twop_tsink[ b ]
    # mEff[ b ]
    # momSq
    # L

    preFactor = avgXKineFactor( mEff, momSq, L )

    avgX = np.zeros( threep.shape )

    for t in range( threep.shape[ -1 ] ):

        avgX[ :, t ] = preFactor * threep[ :, t ] / twop_tsink

    return avgX


def calcMatrixElemEM_ratio( threep, twop_tsink ):

    # threep[ b, t ]
    # twop_tsink[ b ]

    ratio = np.zeros( threep.shape )

    for t in range( threep.shape[ -1 ] ):

        ratio[ :, t ] = threep[ :, t ] / twop_tsink

    return ratio


def calcMatrixElemEM_twopFit( threep, tsink, c0, E0 ):
#def calcMatrixElemEM_twopFit( threep, tsink, c0, c1, E0, E1 ):

    # threep[ b, t ]
    # twop_tsink[ b ]
    # mEff[ b ]
    # momSq
    # L

    binNum = threep.shape[ 0 ]
    T = threep.shape[ -1 ]

    c0_cp = np.repeat( c0, T ).reshape( binNum, T )
    #c1_cp = np.repeat( c1, T ).reshape( binNum, T )
    E0_cp = np.repeat( E0, T ).reshape( binNum, T )
    #E1_cp = np.repeat( E1, T ).reshape( binNum, T )

    ratio = threep / twopFit( c0_cp, E0_cp, tsink )
    #ratio = threep / twopFit( c0_cp, c1_cp, E0_cp, E1_cp, tsink )

    return ratio


def calcMellin_twopFit( threep, tsink, mEff, momSq, L, \
                        c0, E0, moment ):

    # threep[ b, t ]
    # tsink
    # mEff[ b ]
    # momSq
    # L
    # c0[ b ]
    # E0[ b ]
    
    T = threep.shape[ -1 ]

    c0_cp = np.repeat( c0, T ).reshape( threep.shape )
    E0_cp = np.repeat( E0, T ).reshape( threep.shape )

    if moment == 1 or moment == "avgX":

        preFactor = np.repeat( avgXKineFactor( mEff, momSq, L ), \
                               T ).reshape( threep.shape )

    elif moment == 2 or moment == "avgX2":

        preFactor = -1.0

    elif moment == 3 or moment == "avgX3":

        preFactor = -1.0        

    return preFactor * threep \
        / twopFit( c0_cp, E0_cp, tsink )


def calcAvgX_twopFit( threep, tsink, mEff, momSq, L, \
                      c0, E0 ):

    # threep[ b, t ]
    # tsink
    # mEff[ b ]
    # momSq
    # L
    # c0[ b ]
    # E0[ b ]
    
    binNum = threep.shape[ 0 ]
    T = threep.shape[ -1 ]

    preFactor = np.repeat( avgXKineFactor( mEff, momSq, L ), \
                           T ).reshape( binNum, T )

    c0_cp = np.repeat( c0, T ).reshape( binNum, T )
    E0_cp = np.repeat( E0, T ).reshape( binNum, T )

    avgX = preFactor * threep \
           / twopFit( c0_cp, E0_cp, tsink )

    return avgX


def calcAvgX2_twopFit( threep, tsink, mEff, momSq, L, \
                       c0, E0 ):

    # threep[ b, t ]
    # tsink
    # mEff[ b ]
    # momSq
    # L
    # c0[ b ]
    # E0[ b ]
    
    binNum = threep.shape[ 0 ]
    T = threep.shape[ -1 ]

    #preFactor = np.repeat( avgX2KineFactor( mEff, momSq, L ), \
    #                       T ).reshape( binNum, T )

    preFactor = -1.0

    c0_cp = np.repeat( c0, T ).reshape( binNum, T )
    E0_cp = np.repeat( E0, T ).reshape( binNum, T )

    avgX2 = preFactor * threep \
           / twopFit( c0_cp, E0_cp, tsink )

    return avgX2


def calcAvgX_twopTwoStateFit( threep, tsink, mEff, momSq, L, T, \
                              c0, c1, E0, E1 ):

    # threep[ b, t ]
    # tsink
    # mEff[ b ]
    # momSq
    # L
    # T
    # c0[ b ]
    # c1[ b ]
    # E0[ b ]
    # E1[ b ]
    
    binNum = threep.shape[0]

    # prefactor = E/(m(1/2*m^2-2*E))

    preFactor=1.0
    #preFactor = np.repeat( avgXKineFactor( mEff, momSq, L ), \
    #                       T ).reshape( binNum, T )

    c0_cp = np.repeat( c0, T ).reshape( binNum, T )
    c1_cp = np.repeat( c1, T ).reshape( binNum, T )
    E0_cp = np.repeat( E0, T ).reshape( binNum, T )
    E1_cp = np.repeat( E1, T ).reshape( binNum, T )

    avgX = preFactor * threep \
           / fit.twoStateTwop( tsink, T, \
                               c0_cp, c1_cp, \
                               E0_cp, E1_cp )
    return avgX


def calcAvgX_twopOneStateFit( threep, tsink, mEff, momSq, L, T, G, E ):

    # threep[ b, t ]
    # tsink
    # mEff[ b ]
    # momSq
    # L
    # T
    # G[ b ]
    # E[ b ]
    
    binNum = threep.shape[0]

    # prefactor = E/(m(1/2*m^2-2*E))

    preFactor = energy( mEff, momSq, L ) \
                / ( mEff * ( 0.5 * mEff ** 2 \
                             - 2.0 * energy( mEff, momSq, L ) ** 2 ) )

    G_cp = np.repeat( G, T ).reshape( binNum, T )
    E_cp = np.repeat( E, T ).reshape( binNum, T )
    
    avgX = preFactor * threep \
           / fit.oneStateTwop( tsink, T, \
                               G_cp, E_cp )

    return avgX


def calcAvgX_twoStateFit( a00, c0, mEff, momSq, L, ZvD1 ):

    # a00[ b ]
    # c0 [ b ]
    # mEff[ b ]
    # momSq
    # L
    # ZvD1

    return ZvD1 * avgXKineFactor( mEff, momSq, L ) \
        * a00 / c0

def calcAvgX2_twoStateFit( a00, c0, mEff, momSq, L, ZvD2 ):

    # a00[ b ]
    # c0 [ b ]
    # mEff[ b ]
    # momSq
    # L
    # ZvD1

    return ZvD2 * avgX2KineFactor( mEff, momSq, L ) \
        * a00 / c0

# Calculate the axial charge gA.

# threep: Three-point functions with last dimension as time
# twop_tsink: Two-point funtion at Tsink

def calcgA( threep, twop_tsink ):

    # threep[ b, t ]
    # twop_tsink[ b ]

    gA = np.zeros( threep.shape )

    for t in range( threep.shape[ 1 ] ):

        gA[ :, t ] = threep[ :, t ] / twop_tsink

    return gA


# Calculate the cosh form of the two-point functions

# t: 
# energy:
# tsink:

def twopCosh( t, energy, tsink ):

    return np.exp( - energy * t ) + np.exp( - energy * ( tsink - t ) )


def calcRatio_Q( threep, twop, tsink ):
    
    # threep[ ..., Q, t ]
    # twop[ ..., Q, t ]

    ratio = np.zeros( threep.shape )
    
    for q in range( threep.shape[ -2 ] ):
        for t in range( threep.shape[ -1 ] ):

            ratio[..., q, t] = threep[ ..., q, t ] / twop[ ..., 0, tsink ] \
                               * np.sqrt( np.abs( twop[ ..., q, tsink - t ] \
                                                  * twop[ ..., 0, t ] \
                                                  * twop[ ..., 0, tsink ] \
                                                  / ( twop[ ..., 0, tsink - t ] \
                                                      * twop[ ..., q, t ] \
                                                      * twop[ ..., q, tsink ] ) ) )  

    return ratio


def calcRatio_Q_twopFit( threep, c0, mEff, tsink, p_fin,
                         Q, pSq_twop, L, mpi_info ):
    
    # threep[ ..., p, Q, r, t ]
    # c0[ ..., p^2 ]
    # mEff[ ... ]
    # tsink
    # p_fin[ p, pi ]
    # Q[ q, qi ]
    # pSq_twop[ ps ]
    # L

    QNum = threep.shape[ -3 ]
    ratioNum = threep.shape[ -2 ]
    
    if QNum != len( Q ):

        error_template = "Error (physQuants.calcRatio_Q_twopFit): " \
                         + "length of threep Q dimension {} " \
                         + "does not match length of " \
                         + "Q {}."

        mpi_fncs.mpiPrintError( error_template.format( QNum,
                                                       len( Q ) ),
                                mpi_info )

    ratio = np.zeros( threep.shape[ :-1 ] + ( tsink + 1, ) )

    # Calculate twop from fit parameters
    # twop[ ..., p^2, r, t ]

    twop = np.zeros( c0.shape + ( ratioNum, tsink + 1 ) )

    # Loop over twop p^2
    for ps, ips in fncs.zipXandIndex( pSq_twop ):
        # Loop over time
        for t in range( tsink + 1 ):

            twop_tmp = twopFit( c0[ ..., ips ],
                                energy( mEff, ps, L ),
                                t )
            twop_tmp = np.repeat( twop_tmp, ratioNum )

            twop[ ..., ips, :, t ] \
                = twop_tmp.reshape( twop[ ..., ips,
                                          :, t ].shape )

        # End loop over t
    # End loop over twop p^2

    for p, ip in fncs.zipXandIndex( p_fin ):
        
        pSq_fin = np.dot( p, p )

        pSq_fin_where = np.where( pSq_twop == pSq_fin )

        for q, iq, in fncs.zipXandIndex( Q ):

            pSq_ini = np.dot( p - q, p - q )

            pSq_ini_where = np.where( pSq_twop == pSq_ini )

            for t in range( tsink + 1 ):

                ratio[..., ip, iq, :, t] \
                    = threep[ ..., ip, iq, :, t ] \
                    / twop[ ..., pSq_fin_where, :, tsink ] \
                    * np.sqrt( twop[ ..., pSq_ini_where, :, tsink - t ]
                               * twop[ ..., pSq_fin_where, :, t ]
                               * twop[ ..., pSq_fin_where, :, tsink ]
                               / ( twop[ ..., pSq_fin_where, :, tsink - t ]
                                   * twop[ ..., pSq_ini_where, :, t ]
                                   * twop[ ..., pSq_ini_where, :, tsink ] ) )
            
            # End loop over t
        # End loop ovet Q
    # End loop over p

    return ratio


# Calculate the electromagnetic form factor.

# threep:

def calcEMFF( threep, twop, Qsq, mEff, tsink, latticeDim ):

    emff = np.zeros( threep.shape )

    for q in range( threep.shape[ 0 ] ):

        energy = np.sqrt( mEff ** 2 + ( 2 * np.pi / latticeDim ) ** 2 * Qsq[ q ] )
        
        #factor = 1.0
        factor = 2 * energy / ( energy + mEff )
        #factor = 4.0 * np.sqrt( energy * mEff ) / ( energy + mEff )
        
        for t in range( threep.shape[ 2 ] ):

            emff[ q, :, t ] = factor * threep[ q, :, t ] / twop[ 0, :, tsink ] \
                             * np.sqrt( twop[ q, :, tsink - t ] * twop[ 0, :, t ] * twop[ 0, :, tsink ] \
                                        / ( twop[ 0, :, tsink - t ] * twop[ q, :, t ] * twop[ q, :, tsink ] ) )

    return emff


def calcEMFF_cosh( threep, Qsq, mEff, tsink, latticeDim ):

    emff = np.zeros( threep.shape )

    for q in range( threep.shape[ 0 ] ):

        energy = np.sqrt( mEff ** 2 + ( 2 * np.pi / latticeDim ) ** 2 * Qsq[ q ] )
        
        factor = np.sqrt( energy / mEff ) * energy / ( energy + mEff )

        for t in range( threep.shape[ 2 ] ):

            emff[ q, :, t ] = factor * threep[ q, :, t ] / threep[ 0, :, t ] \
                              * twopCosh( t, mEff, tsink ) / twopCosh( t, energy, tsink )

    return emff


def calcEMFF_ratio( threep, twop, Qsq, mEff ):

    emff = np.zeros( threep.shape )

    for q in range( threep.shape[ 0 ] ):

        for b in range( threep.shape[ 1 ] ):

            for t in range( threep.shape[ 2 ] ):

                emff[ q, b, t ] = threep[ q, b, t ] / twop[ q, b,  t ] \
                                   / threep[0,b,t] * twop[ 0, b, t ]

    return emff

